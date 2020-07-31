import os
import time
import torch

import sys
sys.path.append('.')

from torch.nn.utils import clip_grad_norm_

from base import BaseTrainer
from callbacks import Tqdm, FreezeLayers
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
            config,
            num_classes=len(self.datamanager.datasource.get_attribute()),
            device=self.device)

        # losses
        pos_ratio = torch.tensor(self.datamanager.datasource.get_weight('train'))
        self.criterion, params_loss = build_losses(config, pos_ratio=pos_ratio)

        # optimizer
        self.optimizer, params_optimizers = build_optimizers(config, self.model)

        # learing rate scheduler
        self.lr_scheduler, params_lr_scheduler = build_lr_scheduler(config, self.optimizer)

        # callbacks for freeze backbone
        if config['freeze']['enable']:
            self.freeze = FreezeLayers(self.model, config['freeze']['layers'], config['freeze']['epochs'])
        else:
            self.freeze = None

        # list of metrics
        self.lst_metrics = ['mA', 'accuracy', 'f1_score']

        # track metric
        self.train_metrics = MetricTracker('loss', *self.lst_metrics)
        self.valid_metrics = MetricTracker('loss', *self.lst_metrics)

        # step log loss and accuracy
        self.log_step = (len(self.datamanager.get_dataloader('train')) // 10,
                        len(self.datamanager.get_dataloader('val')) // 10)
        self.log_step = (self.log_step[0] if self.log_step[0] > 0 else 1,
                        self.log_step[1] if self.log_step[1] > 0 else 1)
        
        # best accuracy and loss
        self.best_loss = None
        self.best_metrics = dict()
        for x in self.lst_metrics:
            self.best_metrics[x] = None
        
        # print config
        self._print_config(
            params_model=params_model,
            params_loss=params_loss,
            params_optimizers=params_optimizers,
            params_lr_scheduler=params_lr_scheduler,
            freeze_layers=False if self.freeze == None else True)


        # send model to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # summary model
        summary(
            func=self.logger.info,
            model=self.model,
            input_size=(3, self.datamanager.datasource.get_image_size()[0], self.datamanager.datasource.get_image_size()[1]),
            batch_size=config['data']['batch_size'],
            device='cuda' if self.use_gpu else 'cpu',
            print_step=False)

        # resume model from last checkpoint
        if config['resume'] != '':
            self._resume_checkpoint(config['resume'])

    def train(self):
        # begin train
        for epoch in range(self.start_epoch, self.epochs + 1):
            # freeze layer
            if self.freeze != None:
                self.freeze.on_epoch_begin(epoch)

            # train
            result = self._train_epoch(epoch)
            
            # valid
            result = self._valid_epoch(epoch)

            # learning rate
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
            
            for metric in self.lst_metrics:
                self.writer.add_scalars(metric,
                    {
                        'Train': self.train_metrics.avg(metric),
                        'Val': self.valid_metrics.avg(metric)
                    }, global_step=epoch)
                    
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], global_step=epoch)

            # logging result to console
            log = {'epoch': epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # save model
            save_best_loss = False
            if self.best_loss == None or self.best_loss >= self.valid_metrics.avg('loss'):
                self.best_loss = self.valid_metrics.avg('loss')
                save_best_loss = True

            save_best = dict()
            for metric in self.lst_metrics:
                save_best[metric] = False
                if self.best_metrics[metric] == None or self.best_metrics[metric] <= self.valid_metrics.avg(metric):
                    self.best_metrics[metric] = self.valid_metrics.avg(metric)
                    save_best[metric] = True

            self._save_checkpoint(epoch, save_best_loss, save_best)

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
        if self.cfg_trainer['use_tqdm']:
            tqdm_callback = Tqdm(epoch, len(self.datamanager.get_dataloader('train')), phase='train')
        for batch_idx, (data, labels) in enumerate(self.datamanager.get_dataloader('train')):
            # get time for log num iter per seconds
            if not self.cfg_trainer['use_tqdm']:
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
            if self.config['clip_grad_norm_']['enable']:
                clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.config['clip_grad_norm_']['max_norm'])

            # optimize
            self.optimizer.step()
            
            # calculate instance-based accuracy
            preds = torch.sigmoid(out)
            
            mean_accuracy, accuracy, f1_score = compute_accuracy_cuda(labels, preds)

            # update loss and accuracy in MetricTracker
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('mA', mean_accuracy)
            self.train_metrics.update('accuracy', accuracy)
            self.train_metrics.update('f1_score', f1_score)

            # update process
            if self.cfg_trainer['use_tqdm']:
                tqdm_callback.on_batch_end({
                    'loss': loss.item(),
                    'mA': mean_accuracy,
                    'accuracy': accuracy,
                    'f1-score': f1_score})
            else:
                end_time = time.time()
                if (batch_idx+1) % self.log_step[0] == 0 or (batch_idx+1) == len(self.datamanager.get_dataloader('train')):
                    self.logger.info('Train Epoch: {} {}/{} {:.1f}batch/s Loss: {:.4f} mA: {:.4f} Acc: {:.4f} F1-score: {:.4f}'.format(
                        epoch,
                        batch_idx+1,
                        len(self.datamanager.get_dataloader('train')),
                        1/(end_time-start_time),
                        loss.item(),
                        mean_accuracy,
                        accuracy,
                        f1_score))
        if self.cfg_trainer['use_tqdm']:
            tqdm_callback.on_epoch_end()
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        r""" Validation step
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if self.cfg_trainer['use_tqdm']:
                tqdm_callback = Tqdm(epoch, len(self.datamanager.get_dataloader('val')), phase='val')
            for batch_idx, (data, labels) in enumerate(self.datamanager.get_dataloader('val')):
                if not self.cfg_trainer['use_tqdm']:
                    start_time = time.time()
                # push data to device
                data, labels = data.to(self.device), labels.to(self.device)
                
                # forward batch
                out = self.model(data)

                # calculate loss and accuracy
                loss = self.criterion(out, labels)

                # calculate instance-based accuracy
                preds = torch.sigmoid(out)
                
                mean_accuracy, accuracy, f1_score = compute_accuracy_cuda(labels, preds)

                # update loss and accuracy in MetricTracker
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('mA', mean_accuracy)
                self.valid_metrics.update('accuracy', accuracy)
                self.valid_metrics.update('f1_score', f1_score)

                # update process
                if self.cfg_trainer['use_tqdm']:
                    tqdm_callback.on_batch_end({
                        'loss': loss.item(),
                        'mA': mean_accuracy,
                        'accuracy': accuracy,
                        'f1-score': f1_score})
                else:
                    end_time = time.time()
                    if (batch_idx+1) % self.log_step[1] == 0 or (batch_idx+1) == len(self.datamanager.get_dataloader('val'))-1:
                        self.logger.info('Valid Epoch: {} {}/{} {:.1f}batch/s Loss: {:.4f} mA: {:.4f} Acc: {:.4f} F1-score: {:.4f}'.format(
                            epoch,
                            batch_idx+1,
                            len(self.datamanager.get_dataloader('val')),
                            1/(end_time-start_time),
                            loss.item(),
                            mean_accuracy,
                            accuracy,
                            f1_score))
        if self.cfg_trainer['use_tqdm']:
            tqdm_callback.on_epoch_end()
        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best_loss, save_best_metrics):
        r""" Save model to file
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'loss': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_loss': self.best_loss
        }
        for metric in self.lst_metrics:
            state.update({'best_{}'.format(metric): self.best_metrics[metric]})

        filename = os.path.join(self.checkpoint_dir, 'model_last.pth')
        self.logger.info("Saving last model: model_last.pth ...")
        torch.save(state, filename)
        
        if save_best_loss:
            filename = os.path.join(self.checkpoint_dir, 'model_best_loss.pth')
            self.logger.info("Saving current best loss: model_best_loss.pth ...")
            torch.save(state, filename)
        
        for metric in self.lst_metrics:
            if save_best_metrics[metric]:
                filename = os.path.join(self.checkpoint_dir, 'model_best_{}.pth'.format(metric))
                self.logger.info("Saving current best {}: model_best_{}.pth ...".format(metric, metric))
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
        self.best_loss = checkpoint['best_loss']
        for metric in self.lst_metrics:
            self.best_metrics[metric] = checkpoint['best_{}'.format(metric)]
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    
    def _print_config(
        self,
        params_model=None,
        params_loss=None,
        params_optimizers=None,
        params_lr_scheduler=None,
        freeze_layers=False):
        
        r""" print config into log file
        """
        def __prams_to_str(params: dict):
            if params == None:
                return ''
            row_format ="{:>4}  " * len(params)
            return row_format.format(*[key + ': ' + str(value) for key, value in params.items()])

        self.logger.info('Run id: %s' % (self.run_id))
        self.logger.info('Dataset: %s, batch_size: %d ' % (self.config['data']['name'], self.config['data']['batch_size']))
        self.logger.info('Model: %s ' % (self.config['model']['name']) + __prams_to_str(params_model))
        if freeze_layers:
            self.logger.info('Freeze layer: %s ,at first epoch %d' % (str(self.config['freeze']['layers']), self.config['freeze']['epochs']))
        self.logger.info('Loss: %s ' % (self.config['loss']['name']) + __prams_to_str(params_loss))
        self.logger.info('Optimizer: %s ' % (self.config['optimizer']['name']) + __prams_to_str(params_optimizers))
        if params_lr_scheduler != None:
            self.logger.info('Lr scheduler: %s ' % (self.config['lr_scheduler']['name']) + __prams_to_str(params_lr_scheduler))

