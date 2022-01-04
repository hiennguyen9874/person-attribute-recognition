import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import argparse

from trainer import Trainer_Epoch, Trainer_Episode
from utils import read_config

def main(config):
    if config['type'].lower() == 'epoch':
        trainer = Trainer_Epoch(config)
    elif config['type'].lower() == 'episode':
        trainer = Trainer_Episode(config)
    else:
        raise KeyError('type error')
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='config/base_epoch.yml', type=str, help='path to config file')
    parser.add_argument('--resume', default='', type=str, help='path to model_pretrained.pth file')
    parser.add_argument('--only_model', default=False, type=lambda x: (str(x).lower() == 'true'), help='only resume model')
    parser.add_argument('--colab', default=False, type=lambda x: (str(x).lower() == 'true'), help='train on colab or no')
    args = parser.parse_args()
    
    config = read_config(args.config)

    config.update({'resume': args.resume})
    config.update({'colab': args.colab})
    config.update({'only_model': args.only_model})
    main(config)
