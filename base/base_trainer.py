import os
import pytz
import torch
import logging
import shutil
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from logger import setup_logging
from utils import config_to_str

class BaseTrainer(object):
    def __init__(self, config):
        self.config = config

        self.cfg_trainer = config['trainer_colab'] if config['colab'] == True else config['trainer']

        # self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self.run_id = datetime.now(pytz.timezone(config['timezone'])).strftime(r'%m%d_%H%M%S')
        
        self.checkpoint_dir = os.path.join(self.cfg_trainer['checkpoint_dir'], self.run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logs_dir = os.path.join(self.cfg_trainer['log_dir'], self.run_id)
        os.makedirs(self.logs_dir, exist_ok=True)

        if self.config['colab']:
            self.logs_dir_saved = os.path.join(self.cfg_trainer['log_dir_saved'], self.run_id)

        setup_logging(self.logs_dir)
        self.logger = logging.getLogger('train')
        
        self.use_gpu = self.cfg_trainer['n_gpu'] > 0 and torch.cuda.is_available()
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.map_location = torch.device('cuda:0' if self.use_gpu else 'cpu')

        self.epochs = self.cfg_trainer['epochs']
        self.start_epoch = 1
        self.writer = SummaryWriter(self.logs_dir)
    
    def _print_config(self, params_model=None, params_loss=None, params_optimizers=None, params_lr_scheduler=None, freeze_layers=False):
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

    def _save_logs(self, epoch):
        r""" Save logs from google colab to google drive
        """
        if os.path.isdir(self.logs_dir_saved):
            shutil.rmtree(self.logs_dir_saved)
        shutil.copytree(self.logs_dir, self.logs_dir_saved)

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
