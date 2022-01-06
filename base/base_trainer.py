import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import pytz
import torch
import shutil
import logging
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from logger import setup_logging
from utils import *


class BaseTrainer(object):
    def __init__(self, config):
        self.config = config

        self.cfg_trainer = (
            config["trainer"]
        )

        # self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self.run_id = datetime.now(pytz.timezone(config["timezone"])).strftime(
            r"%m%d_%H%M%S"
        )

        self.checkpoint_dir = os.path.join(
            self.cfg_trainer["checkpoint_dir"], self.run_id
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logs_dir = os.path.join(self.cfg_trainer["log_dir"], self.run_id)
        os.makedirs(self.logs_dir, exist_ok=True)



        setup_logging(self.logs_dir)
        self.logger = logging.getLogger("train")

        self.use_gpu = self.cfg_trainer["n_gpu"] > -1 and torch.cuda.is_available()
        if self.cfg_trainer["n_gpu"] >= torch.cuda.device_count():
            raise KeyError("n_gpu not in cuda visible!")
        if not self.use_gpu:
            self.device = torch.device(type="cpu")
        else:
            self.device = torch.device(type="cuda", index=self.cfg_trainer["n_gpu"])
        torch.cuda.set_device(self.cfg_trainer["n_gpu"])

        if self.use_gpu:
            torch.backends.cudnn.benchmark = True

        self.start_epoch = 1
        self.epochs = self.cfg_trainer["epochs"]
        self.writer = SummaryWriter(self.logs_dir)

