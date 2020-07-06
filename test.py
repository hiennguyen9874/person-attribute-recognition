import argparse
import data
import os
import logging
import torch
import torch.nn as nn

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
    map_location = (lambda storage, loc: storage) if use_gpu else None

    datamanager = DataManger(config['data'])
    attribute_name = datamanager.datasource.get_attribute()
    
    model = OSNet(num_classes=len(datamanager.datasource.get_attribute()))
    model = model.eval()

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])
    
    accuracy_all = torch.zeros(1, 6)
    count = 0
    with tqdm(total=len(datamanager.get_dataloader('val'))) as epoch_pbar:
        for batch_idx, (data, labels) in enumerate(datamanager.get_dataloader('val')):
            data, labels = data.to(device), labels.to(device)

            out = model(data)

            preds = torch.sigmoid(out)
            preds[preds < 0.5] = 0
            preds[preds >= 0.5] = 1

            labels = labels.type(torch.BoolTensor)
            preds = preds.type(torch.BoolTensor)
            # intersect = (preds & labels).type(torch.FloatTensor)
            # union = (preds | labels).type(torch.FloatTensor)
            # accuracy = torch.sum(intersect, dim=0) / torch.sum(union, dim=0)
            accuracy = torch.sum(torch.eq(labels, preds).type(torch.FloatTensor), dim=0)
            accuracy_all += accuracy
            count += data.size(0)
            epoch_pbar.update(1)
    accuracy_all /= count
    # print('accuracy: %.4f', (accuracy_all))
    for i in range(len(attribute_name)):
        print(list(attribute_name)[i], ': ', accuracy_all[i])

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
