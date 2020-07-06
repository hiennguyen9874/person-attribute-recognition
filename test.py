import argparse
from os import pread
import data
import os
import logging
import torch
import torch.nn as nn
import numpy as np

from torchsummary import summary
from tqdm import tqdm

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
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    datamanager = DataManger(config['data'], phase='test')
    
    model = OSNet(num_classes=len(datamanager.datasource.get_attribute()))

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    
    # TODO: mean Accuracy (mA), four instance-based metrics, Accuracy (Acc), Precision (Prec), Recall (Rec) and F1-score (F1).
    accuracy_all = torch.zeros(1, 6)
    count = 0
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

    # accuracy
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    accuracy = np.equal(preds, labels).astype(float)
    accuracy = np.sum(accuracy, axis=0) / preds.shape[0]
    preds = preds.astype(bool)
    labels = labels.astype(bool)
    tp = np.sum((preds & labels).astype(float), axis=0)
    fp = np.sum((preds & (~labels)).astype(float), axis=0)
    tn = np.sum(((~preds) & (~labels)).astype(float), axis=0)
    fn = np.sum(((~preds) & labels).astype(float), axis=0)
    precision = tp / np.add(tp, fp)
    recall = tp /np.add(tp, fn)
    f1_score = 2 * np.multiply(precision, recall) / np.add(precision, recall)

    print('           ', 'hard_hat   none_hard_hat    safety_vest    none_safety_vest    no_hat     no_vest')
    print('accuracy:  ', '%0.4f,     %0.4f,           %0.4f,         %0.4f,          %0.4f,    %0.4f' % (*accuracy,))
    print('precision: ', '%0.4f,     %0.4f,           %0.4f,         %0.4f,          %0.4f,    %0.4f' % (*precision,))
    print('recall:    ', '%0.4f,     %0.4f,           %0.4f,         %0.4f,          %0.4f,    %0.4f' % (*recall,))
    print('f1 score:  ', '%0.4f,     %0.4f,           %0.4f,         %0.4f,          %0.4f,    %0.4f' % (*f1_score,))

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
