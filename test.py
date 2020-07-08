import argparse
from os import pread
import data
import os
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary
from tqdm import tqdm

from models import OSNet, BaselineReid, BaselineAttribute
from data import DataManger
from logger import setup_logging
from utils import read_json, write_json, rmdir
from evaluators import plot_loss, show_image, recognition_metrics

def main(config):
    # (os.path.exists(config['testing']['output_dir']) or os.makedirs(config['testing']['output_dir'], exist_ok=True)) and rmdir(config['testing']['output_dir'], remove_parent=False)
    os.path.exists(os.path.join(config['testing']['output_dir'], 'info.log')) and os.remove(os.path.join(config['testing']['output_dir'], 'info.log'))
    setup_logging(config['testing']['output_dir'])
    logger = logging.getLogger('test')

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    datamanager = DataManger(config['data'], phase='test')
    
    model = OSNet(num_classes=len(datamanager.datasource.get_attribute()))

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    
    preds = []
    labels = []

    with tqdm(total=len(datamanager.get_dataloader('test'))) as epoch_pbar:
        with torch.no_grad():
            for batch_idx, (data, _labels) in enumerate(datamanager.get_dataloader('test')):
                data, _labels = data.to(device), _labels.to(device)

                out = model(data)

                _preds = torch.sigmoid(out)
                preds.append(_preds)
                labels.append(_labels)
                epoch_pbar.update(1)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    result_label, result_instance = recognition_metrics(labels, preds)
        
    logger.info('instance-based metrics:')
    logger.info('accuracy: %0.4f' % result_instance.accuracy)
    logger.info('precision: %0.4f' % result_instance.precision)
    logger.info('recall: %0.4f' % result_instance.recall)
    logger.info('f1_score: %0.4f' % result_instance.f1_score)
    
    result = np.stack([result_label.accuracy, result_label.precision, result_label.recall, result_label.f1_score], axis=0)
    fig, ax = plt.subplots(1, 1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table = ax.table(
        cellText=np.around(result*100, 2),
        rowLabels=['accuracy', 'precision', 'recall', 'f1_score'],
        colLabels=datamanager.datasource.get_attribute(),
        loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    # table.scale(1,4)
    ax.axis('off')
    # fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: ./config.json)')
    parser.add_argument('-r', '--resume', default='', type=str, help='resume file path (default: .)')
    parser.add_argument('-e', '--extract', default=True, type=lambda x: (str(x).lower() == 'true'), help='extract feature (default: true')
    args = parser.parse_args()

    config = read_json(args.config)
    config.update({'resume': args.resume})
    config.update({'extract': args.extract})

    main(config)
