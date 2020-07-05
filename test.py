import argparse
import os
import logging
import torch

from torchsummary import summary

from models import OSNet
from data import DataManger
from logger import setup_logging
from utils import read_json, write_json
from evaluators import plot_loss, show_image

def main(config):
    setup_logging(os.getcwd())
    logger = logging.getLogger('test')

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    datamanager = DataManger(config['data'], phase='test')
    
    model = OSNet(num_classes=datamanager.datasource.get_num_classes('train'), is_training=False)
    model = model.eval()

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.json',
                        type=str, help='config file path (default: ./config.json)')
    parser.add_argument('-r', '--resume', default='',
                        type=str, help='resume file path (default: .)')
    parser.add_argument('-e', '--extract', default=True, type=lambda x: (
        str(x).lower() == 'true'), help='extract feature (default: true')
    args = parser.parse_args()

    config = read_json(args.config)
    config.update({'resume': args.resume})
    config.update({'extract': args.extract})

    main(config)
