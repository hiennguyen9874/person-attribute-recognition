import argparse
import os
import logging
import torch
import numpy as np

from tqdm import tqdm

from models import build_model
from data import DataManger_Epoch, DataManger_Episode
from logger import setup_logging
from utils import read_config, rmdir, summary
from evaluators import recognition_metrics, log_test

def main(config):
    cfg_trainer = config['trainer_colab'] if config['colab'] == True else config['trainer']
    run_id = config['resume'].split('/')[-2]
    file_name = config['resume'].split('/')[-1].split('.')[0]
    output_dir = os.path.join(cfg_trainer['output_dir'], run_id, file_name)
    (os.path.exists(output_dir) or os.makedirs(output_dir, exist_ok=True)) and rmdir(output_dir, remove_parent=False)
    setup_logging(output_dir)
    logger = logging.getLogger('test')

    use_gpu = cfg_trainer['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    if config['type'].lower() == 'epoch':
        datamanager = DataManger_Epoch(config['data'])
    elif config['type'].lower() == 'episode':
        datamanager = DataManger_Episode(config['data'])
    else:
        raise KeyError
    
    model, _ = build_model(config, num_classes=len(datamanager.datasource.get_attribute()))

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

    # # get best threshold
    # from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # precision = dict()
    # recall = dict()
    # thresholds_pr = dict()
    # pr_auc = dict()
    # best_threshold = dict()

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # thresholds_roc = dict()

    # for i in range(len(datamanager.datasource.get_attribute())):
    #     precision[i], recall[i], thresholds_pr[i] = precision_recall_curve(labels[:, i], preds[:, i])
    #     pr_auc[i] = auc(recall[i], precision[i])
    #     best_threshold[i] = np.argmax((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
        
    #     fpr[i], tpr[i], thresholds_roc[i] = roc_curve(labels[:, i], preds[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    #     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    #     ax1.plot(recall[i], precision[i], label='Precision-Recall Curve, mAP: %f' % pr_auc[i])
    #     ax1.scatter(
    #         recall[i][best_threshold[i]],
    #         precision[i][best_threshold[i]],
    #         marker='o',
    #         color='black',
    #         label='Best threshold %f' % (thresholds_pr[i][best_threshold[i]]))

    #     ax1.set_xlabel('Recall')
    #     ax1.set_ylabel('Precision')
    #     ax1.set_title('Attribute: %s' % datamanager.datasource.get_attribute()[i])
    #     # ax1.legend(loc="lower right")

    #     fig, ax2 = plt.subplots(122)
    #     ax2.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % (roc_auc[i]))
    #     ax2.plot([0, 1], [0, 1], 'k--')
    #     ax2.scatter(fpr[i][best_threshold[i]], tpr[i][best_threshold[i]], marker='o', color='black', label='Best threshold %f' % (thresholds[i][best_threshold[i]]))
    #     ax2.set_xlim([0.0, 1.0])
    #     ax2.set_ylim([0.0, 1.05])
    #     ax2.set_xlabel('False Positive Rate')
    #     ax2.set_ylabel('True Positive Rate')
    #     ax2.set_title('Attribute: %s' % datamanager.datasource.get_attribute()[i])
    #     # ax2.legend(loc="lower right")
    
    # plt.show()

    result_label, result_instance = recognition_metrics(labels, preds)
    log_test(logger.info, datamanager.datasource.get_attribute(), datamanager.datasource.get_weight('test'), result_label, result_instance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='base/config.json', type=str, help='config file path (default: base/config.json)')
    parser.add_argument('--resume', default='', type=str, help='resume file path (default: .)')
    parser.add_argument('--colab', default=False, type=lambda x: (str(x).lower() == 'true'), help='train on colab (default: false)')
    args = parser.parse_args()

    config = read_config(args.config)
    config.update({'resume': args.resume})
    config.update({'colab': args.colab})
    
    main(config)

