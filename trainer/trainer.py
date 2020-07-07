import torch
import os
import shutil

import sys
sys.path.append('.')

import torch.nn as nn

from tqdm import tqdm
from torchsummary import summary

from data import DataManger
from base import BaseTrainer
from models import OSNet, Baseline
from optimizers import WarmupMultiStepLR
from utils import MetricTracker, rmdir

class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        self.datamanager = DataManger(config['data'])

        # model
        self.model = Baseline(num_classes=len(self.datamanager.datasource.get_attribute()))
        self.logger.info('Model name: %s' % (self.model.__class__.__name__))

        # summary model
        summary(self.model, input_size=(3, 256, 128), batch_size=config['data']['batch_size'], device='cpu')
        
        # losses
        bce_weights = torch.tensor(self.datamanager.datasource.get_weight('train'))
        bce_weights = torch.exp(-1 * bce_weights)
        bce_weights = bce_weights.expand(config['data']['batch_size'], len(self.datamanager.datasource.get_attribute()))
        self.criterion = nn.BCEWithLogitsLoss(weight=bce_weights)

        # optimizer
        cfg_optimizer = config['optimizer']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg_optimizer['lr'],
            weight_decay=cfg_optimizer['weight_decay'])

        # learing rate scheduler
        cfg_lr_scheduler = config['lr_scheduler']
        self.lr_scheduler = WarmupMultiStepLR(
            self.optimizer,
            milestones=cfg_lr_scheduler['steps'],
            gamma=cfg_lr_scheduler['gamma'],
            warmup_factor=cfg_lr_scheduler['factor'],
            warmup_iters=cfg_lr_scheduler['iters'],
            warmup_method=cfg_lr_scheduler['method'])

        # track metric
        self.train_metrics = MetricTracker('loss', 'accuracy')
        self.valid_metrics = MetricTracker('loss', 'accuracy')

        # save best accuracy for function _save_checkpoint
        self.best_accuracy = None
        self.best_loss = None
        
        # send model to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # resume model from last checkpoint
        if config['resume'] != '':
            self._resume_checkpoint(config['resume'])

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            result = self._valid_epoch(epoch)

            # add scalars to tensorboard
            self.writer.add_scalars('Loss',
                {
                    'Train': self.train_metrics.avg('loss'),
                    'Val': self.valid_metrics.avg('loss')
                }, global_step=epoch)
            self.writer.add_scalars('Accuracy',
                {
                    'Train': self.train_metrics.avg('accuracy'),
                    'Val': self.valid_metrics.avg('accuracy')
                }, global_step=epoch)

            # logging result to console
            log = {'epoch': epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # save model
            save_best_accuracy = False
            save_best_loss = False
            if self.best_accuracy == None or self.best_accuracy < self.valid_metrics.avg('accuracy'):
                self.best_accuracy = self.valid_metrics.avg('accuracy')
                save_best_accuracy = True
            
            if self.best_loss == None or self.best_loss < self.valid_metrics.avg('loss'):
                self.best_loss = self.valid_metrics.avg('loss')
                save_best_loss = True

            self._save_checkpoint(epoch, save_best_accuracy=save_best_accuracy, save_best_loss=save_best_loss)

            # save logs
            self._save_logs(epoch)

    def _train_epoch(self, epoch):
        """ Training step
        """
        self.model.train()
        self.train_metrics.reset()
        with tqdm(total=len(self.datamanager.get_dataloader('train'))) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {epoch}')
            for batch_idx, (data, labels) in enumerate(self.datamanager.get_dataloader('train')):
                # push data to device
                data, labels = data.to(self.device), labels.to(self.device)

                # zero gradient
                self.optimizer.zero_grad()

                # forward batch
                out = self.model(data)

                # calculate loss and accuracy
                loss =  self.criterion(out, labels)
                
                # backward parameters
                loss.backward()

                # optimize
                self.optimizer.step()
                
                # caculate instabce-based accuracy
                preds = torch.sigmoid(out)
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                
                labels = labels.type(torch.BoolTensor)
                preds = preds.type(torch.BoolTensor)
                intersect = (preds & labels).type(torch.FloatTensor)
                union = (preds | labels).type(torch.FloatTensor)
                accuracy = torch.mean((torch.sum(intersect, dim=1) / torch.sum(union, dim=1)))
                
                # update loss and accuracy in MetricTracker
                self.train_metrics.update('loss', loss.item())
                self.train_metrics.update('accuracy', accuracy.item())

                # update process bar
                epoch_pbar.set_postfix({
                    'train_loss': self.train_metrics.avg('loss'),
                    'train_acc': self.train_metrics.avg('accuracy')})
                epoch_pbar.update(1)
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        """ Validation step
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            with tqdm(total=len(self.datamanager.get_dataloader('val'))) as epoch_pbar:
                epoch_pbar.set_description(f'Epoch {epoch}')
                for batch_idx, (data, labels) in enumerate(self.datamanager.get_dataloader('val')):
                    # push data to device
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    # forward batch
                    out = self.model(data)

                    # calculate loss and accuracy
                    loss =  self.criterion(out, labels)

                    # caculate instabce-based accuracy
                    preds = torch.sigmoid(out)
                    preds[preds < 0.5] = 0
                    preds[preds >= 0.5] = 1
                    
                    labels = labels.type(torch.BoolTensor)
                    preds = preds.type(torch.BoolTensor)
                    intersect = (preds & labels).type(torch.FloatTensor)
                    union = (preds | labels).type(torch.FloatTensor)
                    accuracy = torch.mean((torch.sum(intersect, dim=1) / torch.sum(union, dim=1)))

                    # update loss and accuracy in MetricTracker
                    self.valid_metrics.update('loss', loss.item())
                    self.valid_metrics.update('accuracy', accuracy.item())

                    # update process bar
                    epoch_pbar.set_postfix({
                        'val_loss': self.valid_metrics.avg('loss'),
                        'val_acc': self.valid_metrics.avg('accuracy')})
                    epoch_pbar.update(1)
        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best_accuracy=True, save_best_loss=True):
        """ Save model to file
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            # 'loss': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss
        }
        filename = os.path.join(self.checkpoint_dir, 'model_last.pth')
        self.logger.info("Saving last model: model_last.pth ...")
        torch.save(state, filename)
        if save_best_accuracy:
            filename = os.path.join(self.checkpoint_dir, 'model_best_accuracy.pth')
            self.logger.info("Saving current best: model_best_accuracy.pth ...")
            torch.save(state, filename)
        if save_best_loss:
            filename = os.path.join(self.checkpoint_dir, 'model_best_loss.pth')
            self.logger.info("Saving current best: model_best_loss.pth ...")
            torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        """ Load model from checkpoint
        """
        if not os.path.exists(resume_path):
            raise FileExistsError("Resume path not exist!")
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.map_location)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.criterion.load_state_dict(checkpoint['loss'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_loss = checkpoint['best_loss']
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _save_logs(self, epoch):
        """ Save logs from google colab to google drive
        """
        if os.path.isdir(self.logs_dir_saved):
            shutil.rmtree(self.logs_dir_saved)
        shutil.copytree(self.logs_dir, self.logs_dir_saved)
