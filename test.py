import argparse
import os
import logging
import torch

from torchsummary import summary

from models import Baseline
from data import DataManger
from logger import setup_logging
from utils import read_json, write_json
from evaluators import top_k, mAP, compute_distance_matrix, cmc_rank, feature_extractor, plot_loss, show_image

def main(config):
    setup_logging(os.getcwd())
    logger = logging.getLogger('test')

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    datamanager = DataManger(config['data'], phase='test')
    
    model = Baseline(num_classes=datamanager.datasource.get_num_classes(
        'train'), is_training=False)
    model = model.eval()

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

    if config['extract']:
        logger.info('Extract feature from query set...')
        query_feature, query_label = feature_extractor(
            model, datamanager.get_dataloader('query'), device)

        logger.info('Extract feature from gallery set...')
        gallery_feature, gallery_label = feature_extractor(
            model, datamanager.get_dataloader('gallery'), device)

        gallery_embeddings = (gallery_feature, gallery_label)
        query_embeddings = (query_feature, query_label)

        os.makedirs(config['testing']['ouput_dir'], exist_ok=True)

        with open(os.path.join(config['testing']['ouput_dir'], 'gallery_embeddings.pt'), 'wb') as f:
            torch.save(gallery_embeddings, f)

        with open(os.path.join(config['testing']['ouput_dir'], 'query_embeddings.pt'), 'wb') as f:
            torch.save(query_embeddings, f)

    gallery_feature, gallery_label = torch.load(os.path.join(
        config['testing']['ouput_dir'], 'gallery_embeddings.pt'), map_location='cpu')
    query_feature, query_label = torch.load(os.path.join(
        config['testing']['ouput_dir'], 'query_embeddings.pt'), map_location='cpu')

    distance = compute_distance_matrix(query_feature, gallery_feature)

    top1 = top_k(distance, output=gallery_label, target=query_label, k=1)
    top5 = top_k(distance, output=gallery_label, target=query_label, k=5)
    top10 = top_k(distance, output=gallery_label, target=query_label, k=10)
    m_ap = mAP(distance, output=gallery_label, target=query_label, k='all')

    logger.info('Datasets: {}, without spatial-temporal: top1: {}, top5: {}, top10: {}, mAP: {}'.format(
        datamanager.datasource.get_name_dataset(), top1, top5, top10, m_ap))


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
