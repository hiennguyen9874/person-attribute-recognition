import argparse
import os
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from models import build_model
from data import DataManger
from logger import setup_logging
from utils import read_json, rmdir, summary
from evaluators import recognition_metrics

def main(config):
    cfg_testing = config['testing']
    run_id = config['resume'].split('/')[-2]
    file_name = config['resume'].split('/')[-1].split('.')[0]
    output_dir = os.path.join(cfg_testing['output_dir'], run_id, file_name)
    (os.path.exists(output_dir) or os.makedirs(output_dir, exist_ok=True)) and rmdir(output_dir, remove_parent=False)
    setup_logging(output_dir)
    logger = logging.getLogger('test')

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    datamanager = DataManger(config['data'], phase='test')
    
    model = build_model(cfg_testing['model'], num_classes=len(datamanager.datasource.get_attribute()))

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
    
    logger.info('class-based metrics:')
    result = np.stack([result_label.accuracy, result_label.mean_accuracy, result_label.precision, result_label.recall, result_label.f1_score], axis=0)
    result = np.around(result*100, 2)
    result = result.transpose()
    row_format ="{:>17}" * 6
    logger.info(row_format.format('attribute', 'accuracy', 'mA', 'precision', 'recall', 'f1_score'))
    logger.info(row_format.format(*['-']*6))
    for i in range(len(datamanager.datasource.get_attribute())):
        logger.info(row_format.format(datamanager.datasource.get_attribute()[i], *result[i].tolist()))
    
    logger.info(row_format.format(*['-']*6))
    logger.info(row_format.format(
        'mean',
        round(np.mean(result_label.accuracy)*100, 2),
        round(np.mean(result_label.mean_accuracy)*100, 2),
        round(np.mean(result_label.precision)*100, 2),
        round(np.mean(result_label.recall)*100, 2),
        round(np.mean(result_label.f1_score)*100, 2)))

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
