import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import time
import torch
import logging

from base import BaseTrainer
from callbacks import Tqdm, FreezeLayers
from data import build_datamanager
from evaluators import plot_loss_accuracy, log_test, recognition_metrics
from losses import build_losses
from models import build_model
from optimizers import build_optimizers
from schedulers import build_lr_scheduler
from utils import MetricTracker, summary

class Trainer(BaseTrainer):
    r""" Trainer for person attribute recognition
    """
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        # Datamanager
        self.datamanager, params_data = build_datamanager(config['type'], config['data'])

        # model
        self.model, params_model = build_model(
            config,
            num_classes=len(self.datamanager.datasource.get_attribute()),
            device=self.device)
        
        # losses
        pos_ratio = torch.tensor(self.datamanager.datasource.get_weight('train'))
        self.criterion, params_loss = build_losses(config, pos_ratio=pos_ratio, num_attribute=len(self.datamanager.datasource.get_attribute()))

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
        self.log_step = (len(self.datamanager.get_dataloader('train')) // 5,
                        len(self.datamanager.get_dataloader('val')) // 5)
        self.log_step = (self.log_step[0] if self.log_step[0] > 0 else 1,
                        self.log_step[1] if self.log_step[1] > 0 else 1)
        
        # best accuracy and loss
        self.best_loss = None
        self.best_metrics = dict()
        for x in self.lst_metrics:
            self.best_metrics[x] = None
        
        # print config
        self._print_config(
            params_data=params_data,
            params_model=params_model,
            params_loss=params_loss,
            params_optimizers=params_optimizers,
            params_lr_scheduler=params_lr_scheduler,
            freeze_layers=False if self.freeze == None else True,
            clip_grad_norm_=self.config['clip_grad_norm_']['enable'])

        # send model to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # summary model
        summary(
            model=self.model,
            input_data=torch.zeros((
                self.datamanager.get_batch_size(), 
                3, 
                self.datamanager.get_image_size()[0], 
                self.datamanager.get_image_size()[1])),
            batch_dim=None,
            device='cuda' if self.use_gpu else 'cpu',
            print_func=self.logger.info,
            print_step=False)

        # resume model from last checkpoint
        if config['resume'] != '':
            self._resume_checkpoint(config['resume'], config['only_model'])
    
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
                if self.config['lr_scheduler']['start'] <= epoch:
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
                self._save_logs()

        # wait for tensorboard flush all metrics to file
        self.writer.flush()
        # time.sleep(1*60)
        self.writer.close()
        # save logs to drive if using colab
        if self.config['colab']:
            self._save_logs()
        # plot loss, accuracy and save them to plot.png in saved/logs/<run_id>/plot.png
        plot_loss_accuracy(
            dpath=self.cfg_trainer['log_dir'],
            list_dname=[self.run_id],
            path_folder=self.logs_dir_saved if self.config['colab'] == True else self.logs_dir,
            title=self.run_id + ': ' + self.config['model']['name'] + ", " + self.config['loss']['name'] + ", " + self.config['data']['name'])
      
    def _train_epoch(self, epoch):
        r""" Training step
        """
        raise NotImplementedError
    
    def _valid_epoch(self, epoch):
        r""" Validation step
        """
        raise NotImplementedError
    
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
    
    def _resume_checkpoint(self, resume_path, only_model=False):
        r""" Load model from checkpoint
        """
        if not os.path.exists(resume_path):
            raise FileExistsError("Resume path not exist!")
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.map_location)
        self.model.load_state_dict(checkpoint['state_dict'])
        if only_model:
            self.logger.info("Pretrained-model loaded!")
            return
        self.start_epoch = checkpoint['epoch'] + 1
        self.criterion.load_state_dict(checkpoint['loss'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.best_loss = checkpoint['best_loss']
        for metric in self.lst_metrics:
            self.best_metrics[metric] = checkpoint['best_{}'.format(metric)]
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _print_config(
        self,
        params_data=None,
        params_model=None,
        params_loss=None,
        params_optimizers=None,
        params_lr_scheduler=None,
        freeze_layers=False,
        clip_grad_norm_=False):
        
        r""" print config into log file
        """
        def __prams_to_str(params: dict):
            if params == None:
                return ''
            row_format ="{:>4},  " * len(params)
            return row_format.format(*[key + ': ' + str(value) for key, value in params.items()])

        self.logger.info('Run id: %s' % (self.run_id))
        self.logger.info('Data: ' + __prams_to_str(params_data))
        self.logger.info('Model: %s ' % (self.config['model']['name']) + __prams_to_str(params_model))
        if freeze_layers:
            self.logger.info('Freeze layer: %s, at first epoch %d' % (str(self.config['freeze']['layers']), self.config['freeze']['epochs']))
        self.logger.info('Loss: %s ' % (self.config['loss']['name']) + __prams_to_str(params_loss))
        self.logger.info('Optimizer: %s ' % (self.config['optimizer']['name']) + __prams_to_str(params_optimizers))
        if params_lr_scheduler != None:
            self.logger.info('Lr scheduler: %s ' % (self.config['lr_scheduler']['name']) + __prams_to_str(params_lr_scheduler))
        if clip_grad_norm_:
            self.logger.info('clip_grad_norm_, max_norm: %f' % self.config['clip_grad_norm_']['max_norm'])
