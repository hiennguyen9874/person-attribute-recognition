import sys
sys.path.append('.')

import argparse
import os
import logging
import torch
import numpy as np

from tqdm import tqdm
from easydict import EasyDict

from models import build_model
from data import DataManger_Epoch, DataManger_Episode
from logger import setup_logging
from utils import read_config, rmdir, summary, array_interweave
from evaluators import recognition_metrics, log_test

__all__ = ['recognition_metrics', 'compute_accuracy_cuda', 'log_test']

def recognition_metrics(labels, preds, threshold=0.5, eps = 1e-20):
    r""" https://en.wikipedia.org/wiki/Confusion_matrix
    https://arxiv.org/pdf/1603.07054.pdf
    Args:
        labels (num_sampler, num_attribute): 2d numpy array binary
        preds (num_sampler, num_attribute): 2d numpy array float
        threshold (float): 
        eps (float):
    """
    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0
    preds = preds.astype(bool)
    labels = labels.astype(bool)

    # label metrics
    result_label = EasyDict()
    # positive
    tp_fn = np.sum(labels.astype(float), axis=0)
    # negative
    tn_fp = np.sum((~labels).astype(float), axis=0)
    # true positive
    tp = np.sum((preds & labels).astype(float), axis=0)
    # false positive
    fp = np.sum((preds & (~labels)).astype(float), axis=0)
    # true negative
    tn = np.sum(((~preds) & (~labels)).astype(float), axis=0)
    fn = np.sum(((~preds) & labels).astype(float), axis=0)
    result_label.precision = tp / (np.add(tp, fp) + eps)
    result_label.recall = tp / (np.add(tp, fn) + eps)
    result_label.f1_score = 2 * np.multiply(result_label.precision, result_label.recall) / (np.add(result_label.precision, result_label.recall) + eps)
    # mean accuracy or balanced accuracy
    result_label.mean_accuracy = ((tp / (tp_fn + eps)) + (tn / (tn_fp + eps))) / 2
    result_label.accuracy = (tp + tn) / ((tp + tn + fp + fn) + eps)
    # result_label.accuracy = tp / ((tp + fp + fn) + eps)
    
    # instance metrics
    result_instance = EasyDict()
    _positive = np.sum(preds.astype(float), axis=1)
    _true = np.sum(labels.astype(float), axis=1)
    intersect = np.sum((preds & labels).astype(float), axis=1)
    union = np.sum((preds | labels).astype(float), axis=1)

    _accuracy = intersect / (union + eps)
    _precision = intersect / (_positive + eps)
    _recall = intersect / (_true + eps)
    _f1_score = 2 * np.multiply(_precision, _recall) / (np.add(_precision, _recall) + eps)

    result_instance.accuracy = np.mean(_accuracy)
    result_instance.precision = np.mean(_precision)
    result_instance.recall = np.mean(_recall)
    result_instance.f1_score = np.mean(_f1_score)
    
    return result_label, result_instance

def compute_accuracy_cuda(labels, preds, threshold=0.5, eps=1e-20):
    r""" compute mean accuracy (class-based), accuracy (instance-based), f1-score (instance-based).
    Args:
        labels (tensor 2d (float 0, 1) (num_classes, num_attribute)): tensor 2d 0, 1;
        preds (tensor 2d (float)(num_classes, num_attribute)): output after sigmoid layer
        threshold (float)
        eps (float): epsilon for avoid divide by zero
    Return:
        mean_accuracy (float)
        accuracy (float)
        f1-score (float)
    """
    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0
    labels = labels.type(torch.BoolTensor)
    preds = preds.type(torch.BoolTensor)

    # class-based metrics
    tp_fn = torch.sum(labels.type(torch.FloatTensor), dim=0)
    tn_fp = torch.sum((~labels).type(torch.FloatTensor), dim=0)
    tp = torch.sum((preds & labels).type(torch.FloatTensor), dim=0)
    tn = torch.sum(((~preds) & (~labels)).type(torch.FloatTensor), dim=0)
    mean_accuracy = ((tp / (tp_fn + eps)) + (tn / (tn_fp + eps))) / 2

    # instance-based metrics
    _positive = preds.type(torch.FloatTensor)
    _true = labels.type(torch.FloatTensor)
    intersect = (preds & labels).type(torch.FloatTensor)
    union = (preds | labels).type(torch.FloatTensor)

    _accuracy = (torch.sum(intersect, dim=1) / (torch.sum(union, dim=1) + eps))
    _precision = (torch.sum(intersect, dim=1) / (torch.sum(_positive, dim=1) + eps))
    _recall = (torch.sum(intersect, dim=1) / (torch.sum(_true, dim=1) + eps))
    _f1_score = 2 * torch.mul(_precision, _recall) / (torch.add(_precision, _recall) + eps)

    return torch.mean(mean_accuracy).item(), torch.mean(_accuracy).item(), torch.mean(_f1_score).item()

def log_test(logger_func, attribute_name: list, result_label, result_instance):
    r""" log test from result
    """
    logger_func('instance-based metrics:')
    logger_func('accuracy: %0.4f' % result_instance.accuracy)
    logger_func('precision: %0.4f' % result_instance.precision)
    logger_func('recall: %0.4f' % result_instance.recall)
    logger_func('f1_score: %0.4f' % result_instance.f1_score)

    logger_func('class-based metrics:')
    result = np.stack([result_label.accuracy, result_label.mean_accuracy, result_label.precision, result_label.recall, result_label.f1_score], axis=0)
    result = np.around(result*100, 2)
    result = result.transpose()
    row_format ="{:>20}" * 6
    
    logger_func(row_format.format('attribute', 'accuracy', 'mA', 'precision', 'recall', 'f1_score'))
    
    logger_func(row_format.format(*['-']*6))
    
    for i in range(len(attribute_name)):
        logger_func(row_format.format(attribute_name[i], *result[i].tolist()))

    logger_func(row_format.format(*['-']*6))
    
    logger_func(row_format.format(
        'mean',
        round(np.mean(result_label.accuracy)*100, 2),
        round(np.mean(result_label.mean_accuracy)*100, 2),
        round(np.mean(result_label.precision)*100, 2),
        round(np.mean(result_label.recall)*100, 2),
        round(np.mean(result_label.f1_score)*100, 2)))


def compare_class_based(logger_func, attribute_name, result_label1, result_label2):
    
    logger_func('class-based metrics:')
    result1 = np.stack([result_label1.accuracy, result_label1.mean_accuracy, result_label1.precision, result_label1.recall, result_label1.f1_score], axis=0)
    result1 = np.around(result1*100, 2)
    result1 = result1.transpose()

    result2 = np.stack([result_label2.accuracy, result_label2.mean_accuracy, result_label2.precision, result_label2.recall, result_label2.f1_score], axis=0)
    result2 = np.around(result2*100, 2)
    result2 = result2.transpose()

    row_format = "{:>22}"*6
    logger_func(row_format.format('attribute', 'accuracy', 'mA', 'precision', 'recall', 'f1_score'))
    # logger_func(row_format.format('attribute', 'accuracy', 'accuracy', 'mA', 'mA', 'precision', 'precision', 'recall', 'recall', 'f1_score', 'f1_score'))
    
    logger_func(row_format.format(*['-']*6))

    row_format = "{:>20}" + "{:>20}|{:>5}"*5
    for i in range(len(attribute_name)):
        logger_func(row_format.format(attribute_name[i], *array_interweave(result1[i], result2[i]).tolist()))

    logger_func(row_format.format(*['-']*11))
    
    logger_func(row_format.format(
        'mean',
        round(np.mean(result_label1.accuracy)*100, 2),
        round(np.mean(result_label2.accuracy)*100, 2),
        round(np.mean(result_label1.mean_accuracy)*100, 2),
        round(np.mean(result_label2.mean_accuracy)*100, 2),
        round(np.mean(result_label1.precision)*100, 2),
        round(np.mean(result_label2.precision)*100, 2),
        round(np.mean(result_label1.recall)*100, 2),
        round(np.mean(result_label2.recall)*100, 2),
        round(np.mean(result_label1.f1_score)*100, 2),
        round(np.mean(result_label2.f1_score)*100, 2)))

def test(config, datamanager, logger_func):
    cfg_trainer = config['trainer_colab'] if config['colab'] == True else config['trainer']

    use_gpu = cfg_trainer['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')
    
    model, _ = build_model(config, num_classes=len(datamanager.datasource.get_attribute()))

    logger_func('Loading checkpoint: {} ...'.format(config['resume']))
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

    return result_label, result_instance

if __name__ == "__main__":
    config1 = "config/baseline_peta.yml"
    config2 = "config/episode_peta.yml"

    resume1 = "/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0731_232453/model_best_accuracy.pth"
    resume2 = "/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0804_125045/model_best_accuracy.pth"

    config1 = read_config(config1)
    config1.update({'resume': resume1})
    config1.update({'colab': True})

    config2 = read_config(config2)
    config2.update({'resume': resume2})
    config2.update({'colab': True})
    
    datamanager1 = DataManger_Epoch(config1['data'])
    datamanager2 = DataManger_Episode(config2['data'])

    # model1
    result_label1, result_instance1 = test(config1, datamanager1, print)

    # model 2
    result_label2, result_instance2 = test(config2, datamanager2, print)

    compare_class_based(print, datamanager1.datasource.get_attribute(), result_label1, result_label2)

