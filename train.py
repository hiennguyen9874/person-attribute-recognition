import argparse

from trainer import Trainer
from utils import read_json

def main(config):
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='config.json', type=str, help='config file path (default: ./config.json)')
    parser.add_argument('--resume', default='', type=str, help='config file path (default: ./config.json)')
    parser.add_argument('--colab', default=False, type=lambda x: (str(x).lower() == 'true'), help='train on colab (default: false)')
    args = parser.parse_args()
    config = read_json(args.config)
    config.update({'resume': args.resume})
    config.update({'colab': args.colab})
    main(config)
