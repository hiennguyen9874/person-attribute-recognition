import torch
import os
import logging
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from logger import setup_logging
from utils import config_to_str

class BaseTrainer(object):
    def __init__(self, config):
        self.config = config

        self.cfg_trainer = config['trainer_colab'] if config['colab'] == True else config['trainer']

        self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        
        self.checkpoint_dir = os.path.join(self.cfg_trainer['checkpoint_dir'], self.run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logs_dir = os.path.join(self.cfg_trainer['log_dir'], self.run_id)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.logs_dir_saved = os.path.join(self.cfg_trainer['log_dir_saved'], self.run_id)

        setup_logging(self.logs_dir)
        self.logger = logging.getLogger('train')
        
        self.use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.map_location = torch.device('cuda:0' if self.use_gpu else 'cpu')

        self.epochs = self.cfg_trainer['epochs']
        self.start_epoch = 1
        self.writer = SummaryWriter(self.logs_dir)
    
    def _print_config(self, params_model=None, params_loss=None, params_optimizers=None, params_lr_scheduler=None):
        def __prams_to_str(params: dict):
            if params == None:
                return ''
            row_format ="{:>4}  " * len(params)
            return row_format.format(*[key + ': ' + str(value) for key, value in params.items()])

        self.logger.info('Run id: %s' % (self.run_id))
        self.logger.info('Dataset: %s, batch_size: %d ' % (self.config['data']['name'], self.config['data']['batch_size']))
        self.logger.info('Model: %s ' % (self.config['model']['name']) + __prams_to_str(params_model))
        self.logger.info('Loss: %s ' % (self.config['loss']['name']) + __prams_to_str(params_loss))
        self.logger.info('Optimizer: %s ' % (self.config['optimizer']['name']) + __prams_to_str(params_optimizers))
        self.logger.info('Lr scheduler: %s ' % (self.config['lr_scheduler']['name']) + __prams_to_str(params_lr_scheduler))

