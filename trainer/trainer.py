import torch
import os
import time

import sys
sys.path.append('.')

from torch.nn.utils import clip_grad_norm_

from base import BaseTrainer
from callbacks import Tqdm
from data import DataManger
from evaluators import plot_loss_accuracy, compute_accuracy_cuda
from losses import build_losses
from models import build_model
from optimizers import build_optimizers
from schedulers import build_lr_scheduler
from utils import MetricTracker, summary

class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        self.datamanager = DataManger(config['data'])

        # model
        self.model, params_model = build_model(
            config['model'],
            num_classes=len(self.datamanager.datasource.get_attribute()),
            device=self.device)

        # losses
        pos_ratio = torch.tensor(self.datamanager.datasource.get_weight('train'))
        self.criterion, params_loss = build_losses(config, pos_ratio=pos_ratio)

        # optimizer
        self.optimizer, params_optimizers = build_optimizers(config, self.model)

        # learing rate scheduler
        self.lr_scheduler, params_lr_scheduler = build_lr_scheduler(config, self.optimizer)

        # track metric
        self.train_metrics = MetricTracker('loss', 'accuracy', 'f1_score')
        self.valid_metrics = MetricTracker('loss', 'accuracy', 'f1_score')

        # step log loss and accuracy
        self.log_step = (len(self.datamanager.get_dataloader('train')) // 10,
                        len(self.datamanager.get_dataloader('val')) // 10)
        self.log_step = (self.log_step[0] if self.log_step[0] > 0 else 1, self.log_step[1] if self.log_step[1] > 0 else 1)
        
        # best accuracy and loss
        self.best_accuracy = None
        self.best_loss = None
        self.best_f1_score = None
        
        # print config
        self._print_config(
            params_model=params_model,
            params_loss=params_loss,
            params_optimizers=params_optimizers,
            params_lr_scheduler=params_lr_scheduler)

        # summary model
        # summary(
        #     func=self.logger.info,
        #     model=self.model,
        #     input_size=(3, self.datamanager.datasource.get_image_size()[0], self.datamanager.datasource.get_image_size()[1]),
        #     batch_size=config['data']['batch_size'],
        #     device='cpu',
        #     print_step=False)
        
        # send model to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # resume model from last checkpoint
        if config['resume'] != '':
            self._resume_checkpoint(config['resume'])

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            result = self._valid_epoch(epoch)

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(self.valid_metrics.avg('loss'))
                else:
                    self.lr_scheduler.step()
            
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
            self.writer.add_scalars('F1-Score',
                {
                    'Train': self.train_metrics.avg('f1_score'),
                    'Val': self.valid_metrics.avg('f1_score')
                }, global_step=epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], global_step=epoch)

            # logging result to console
            log = {'epoch': epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # save model
            save_best_accuracy = False
            save_best_loss = False
            save_best_f1_score = False
            if self.best_accuracy == None or self.best_accuracy < self.valid_metrics.avg('accuracy'):
                self.best_accuracy = self.valid_metrics.avg('accuracy')
                save_best_accuracy = True

            if self.best_f1_score == None or self.best_f1_score < self.valid_metrics.avg('f1_score'):
                self.best_f1_score = self.valid_metrics.avg('f1_score')
                save_best_f1_score = True
            
            if self.best_loss == None or self.best_loss > self.valid_metrics.avg('loss'):
                self.best_loss = self.valid_metrics.avg('loss')
                save_best_loss = True

            self._save_checkpoint(epoch, save_best_accuracy=save_best_accuracy, save_best_loss=save_best_loss, save_best_f1_score=save_best_f1_score)

            # save logs to drive if using colab
            if self.config['colab']:
                self._save_logs(epoch)

        # wait for tensorboard flush all metrics to file
        self.writer.flush()
        time.sleep(1*60)
        self.writer.close()
        # plot loss, accuracy and save them to plot.png in saved/logs/<run_id>/plot.png
        plot_loss_accuracy(
            dpath=self.cfg_trainer['log_dir'],
            list_dname=[self.run_id],
            path_folder=self.logs_dir_saved if self.config['colab'] == True else self.logs_dir,
            title=self.run_id + ': ' + self.config['model']['name'] + ", " + self.config['loss']['name'] + ", " + self.config['data']['name'])
      
    def _train_epoch(self, epoch):
        r""" Training step
        """
        self.model.train()
        self.train_metrics.reset()
        if self.cfg_trainer['tqdm']:
            tqdm_callback = Tqdm(epoch, len(self.datamanager.get_dataloader('train')), phase='train')
        for batch_idx, (data, labels) in enumerate(self.datamanager.get_dataloader('train')):
            # get time for log num iter per seconds
            if not self.cfg_trainer['tqdm']:
                start_time = time.time()
            # push data to device
            data, labels = data.to(self.device), labels.to(self.device)

            # zero gradient
            self.optimizer.zero_grad()

            # forward batch
            out = self.model(data)

            # calculate loss and accuracy
            loss = self.criterion(out, labels)
            
            # backward parameters
            loss.backward()

            # Clips gradient norm of an iterable of parameters.
            if self.config['clip_grad_norm_']['active']:
                clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.config['clip_grad_norm_']['max_norm'])

            # optimize
            self.optimizer.step()
            
            # calculate instance-based accuracy
            preds = torch.sigmoid(out)
            preds[preds < 0.5] = 0
            preds[preds >= 0.5] = 1
            
            accuracy, f1_score = compute_accuracy_cuda(labels, preds)

            # update loss and accuracy in MetricTracker
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('accuracy', accuracy.item())
            self.train_metrics.update('f1_score', f1_score.item())

            # update process
            if self.cfg_trainer['tqdm']:
                tqdm_callback.on_batch_end(
                    loss.item(),
                    accuracy.item(),
                    f1_score.item())
            else:
                end_time = time.time()
                if (batch_idx+1) % self.log_step[0] == 0 or (batch_idx+1) == len(self.datamanager.get_dataloader('train')):
                    self.logger.info('Train Epoch: {} {}/{} {:.1f}batch/s Loss: {:.6f} Acc: {:.6f} F1-score: {:.6f}'.format(
                        epoch,
                        batch_idx+1,
                        len(self.datamanager.get_dataloader('train')),
                        1/(end_time-start_time),
                        loss.item(),
                        accuracy.item(),
                        f1_score.item()))
        
        if self.cfg_trainer['tqdm']:
            tqdm_callback.on_epoch_end()
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        r""" Validation step
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if self.cfg_trainer['tqdm']:
                tqdm_callback = Tqdm(epoch, len(self.datamanager.get_dataloader('val')), phase='val')
            for batch_idx, (data, labels) in enumerate(self.datamanager.get_dataloader('val')):
                if not self.cfg_trainer['tqdm']:
                    start_time = time.time()
                # push data to device
                data, labels = data.to(self.device), labels.to(self.device)
                
                # forward batch
                out = self.model(data)

                # calculate loss and accuracy
                loss = self.criterion(out, labels)

                # calculate instance-based accuracy
                preds = torch.sigmoid(out)
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                
                accuracy, f1_score = compute_accuracy_cuda(labels, preds)

                # update loss and accuracy in MetricTracker
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('accuracy', accuracy.item())
                self.valid_metrics.update('f1_score', f1_score.item())

                # update process
                if self.cfg_trainer['tqdm']:
                    tqdm_callback.on_batch_end(
                        loss.item(),
                        accuracy.item(),
                        f1_score.item())
                else:
                    end_time = time.time()
                    if (batch_idx+1) % self.log_step[1] == 0 or (batch_idx+1) == len(self.datamanager.get_dataloader('val'))-1:
                        self.logger.info('Valid Epoch: {} {}/{} {:.1f}batch/s Loss: {:.6f} Acc: {:.6f} F1-score: {:.6f}'.format(
                            epoch,
                            batch_idx+1,
                            len(self.datamanager.get_dataloader('val')),
                            1/(end_time-start_time),
                            loss.item(),
                            accuracy.item(),
                            f1_score.item()))
        if self.cfg_trainer['tqdm']:
            tqdm_callback.on_epoch_end()
        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best_accuracy=False, save_best_loss=False, save_best_f1_score=False):
        r""" Save model to file
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'loss': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'best_f1_score': self.best_f1_score
        }
        filename = os.path.join(self.checkpoint_dir, 'model_last.pth')
        self.logger.info("Saving last model: model_last.pth ...")
        torch.save(state, filename)
        
        if save_best_accuracy:
            filename = os.path.join(self.checkpoint_dir, 'model_best_accuracy.pth')
            self.logger.info("Saving current best accuracy: model_best_accuracy.pth ...")
            torch.save(state, filename)
        
        if save_best_loss:
            filename = os.path.join(self.checkpoint_dir, 'model_best_loss.pth')
            self.logger.info("Saving current best loss: model_best_loss.pth ...")
            torch.save(state, filename)
        
        if save_best_f1_score:
            filename = os.path.join(self.checkpoint_dir, 'model_best_f1_score.pth')
            self.logger.info("Saving current best f1-score: model_best_f1_score.pth ...")
            torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        r""" Load model from checkpoint
        """
        if not os.path.exists(resume_path):
            raise FileExistsError("Resume path not exist!")
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.map_location)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.criterion.load_state_dict(checkpoint['loss'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_loss = checkpoint['best_loss']
        self.best_f1_score = checkpoint['best_f1_score']
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
