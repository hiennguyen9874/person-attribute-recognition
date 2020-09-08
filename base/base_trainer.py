import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import pytz
import torch
import logging
import shutil
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from logger import setup_logging
from utils import *

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

    def _save_logs(self, epoch):
        r""" Save logs from google colab to google drive
        """
        if os.path.isdir(self.logs_dir_saved):
            rmdir(self.logs_dir_saved, remove_parent=False)
        shutil.copytree(self.logs_dir, self.logs_dir_saved)