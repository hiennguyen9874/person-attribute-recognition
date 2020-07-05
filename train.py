import argparse
import os

from trainer import Trainer
from utils import read_json, write_json

def main(config):
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cfg', '--config', default='config.json', type=str, help='config file path (default: ./config.json)')
    parser.add_argument('-r', '--resume', default='', type=str, help='config file path (default: ./config.json)')
    parser.add_argument('-co', '--colab', default=False, type=lambda x: (str(x).lower() == 'true'), help='train on colab (default: false)')
    args = parser.parse_args()
    config = read_json(args.config)
    config.update({'resume': args.resume})
    config.update({'colab': args.colab})
    main(config)
