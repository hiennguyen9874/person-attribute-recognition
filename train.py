import argparse

from trainer import Trainer_Epoch, Trainer_Episode
from utils import read_config

def main(config):
    if config['type'].lower() == 'epoch':
        trainer = Trainer_Epoch(config)
        trainer.train()
    elif config['type'].lower() == 'episode':
        trainer = Trainer_Episode(config)
        trainer.train()
    else:
        raise KeyError('type error')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='config/base_epoch.yml', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--colab', default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    
    config = read_config(args.config)

    config.update({'resume': args.resume})
    config.update({'colab': args.colab})
    main(config)
