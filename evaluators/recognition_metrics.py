import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from easydict import EasyDict

from models import build_model
from data import build_datamanager
from logger import setup_logging
from utils import read_config, rmdir, summary, array_interweave, array_interweave3, COLOR
from evaluators import recognition_metrics

__all__ = ['recognition_metrics', 'compute_accuracy_cuda', 'log_test']

def recognition_metrics(labels, preds, threshold=0.5, eps = 1e-20):
    r""" 
    https://arxiv.org/pdf/1603.07054.pdf
    https://en.wikipedia.org/wiki/Confusion_matrix
    https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition/blob/97e33338432d277596dcb1958292b070facfd6ff/tools/function.py#L81
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

def log_test(logger_func, attribute_name: list, weight, result_label, result_instance):
    r""" log test from result
    Args:
        logger_func: logger.info or print
        attribute_name: list of attribute in dataset
        weight: weight of test dataset
        result_label, result_instance: from recognition_metrics
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
    row_format ="{:>20}" + "{:>10}"*6
    
    logger_func(row_format.format('attribute', 'weight', 'accuracy', 'mA', 'precision', 'recall', 'f1_score'))
    
    logger_func(row_format.format(*['-']*7))
    
    for i in range(len(attribute_name)):
        logger_func(row_format.format(attribute_name[i], np.around(weight[i]*100, 2), *result[i].tolist()))

    logger_func(row_format.format(*['-']*7))
    
    logger_func(row_format.format(
        'mean',
        '-',
        round(np.mean(result_label.accuracy)*100, 2),
        round(np.mean(result_label.mean_accuracy)*100, 2),
        round(np.mean(result_label.precision)*100, 2),
        round(np.mean(result_label.recall)*100, 2),
        round(np.mean(result_label.f1_score)*100, 2)))

def compare_class_based(logger_func, attribute_name, weight, result_label1, result_label2, color=COLOR.BOLD):
    r""" log result and the difference between result_label1 and result_label2
    Args:
        logger_func: logger.info or print
        attribute_name: list of attribute
        weight: weight of each attribute in testset
        result_label1, result_label2
        color: result higher will colored
    """
    # logger_func('class-based metrics:')
    result1 = np.stack([result_label1.accuracy, result_label1.mean_accuracy, result_label1.precision, result_label1.recall, result_label1.f1_score], axis=0)
    result1 = np.around(result1*100, 2)
    result1 = result1.transpose()

    result2 = np.stack([result_label2.accuracy, result_label2.mean_accuracy, result_label2.precision, result_label2.recall, result_label2.f1_score], axis=0)
    result2 = np.around(result2*100, 2)
    result2 = result2.transpose()

    row_format = "{:>20}{:>12}{:>15}{:>12}{:>20}{:>14}{:>18}"
    logger_func(row_format.format('attribute', 'weight', 'accuracy', 'mA', 'precision', 'recall', 'f1_score'))
    
    logger_func(row_format.format(*['-']*7))

    for i in range(len(attribute_name)):
        row_format = "{:>20}" + "{:>12}"
        for j in range(5):
            if result1[i][j] > result2[i][j]:
                row_format += color + "{:>10}" + COLOR.END +"|{:>5}"
            elif result1[i][j] < result2[i][j]:
                row_format += "{:>10}|" + color + "{:>5}" + COLOR.END
            else:
                row_format += color + "{:>10}" + COLOR.END + "|" + color + "{:>5}" + COLOR.END
        logger_func(row_format.format(attribute_name[i], np.around(weight[i]*100, 2), *array_interweave(result1[i], result2[i]).tolist()))

    row_format = "{:>20}" + "{:>12}" + "{:>10}|{:>5}"*5
    logger_func(row_format.format(*['-']*12))
    
    mean_result1 = np.around(np.mean(np.array([result_label1.accuracy, result_label1.mean_accuracy, result_label1.precision, result_label1.recall, result_label1.f1_score]), axis=1)*100, 2)
    
    mean_result2 = np.around(np.mean(np.array([result_label2.accuracy, result_label2.mean_accuracy, result_label2.precision, result_label2.recall, result_label2.f1_score]), axis=1)*100, 2)
    
    row_format = "{:>20}" + "{:>12}"
    for i in range(5):
        if mean_result1[i] > mean_result2[i]:
            row_format += color + "{:>10}" + COLOR.END +"|{:>5}"
        elif mean_result1[i] < mean_result2[i]:
            row_format += "{:>10}|" + color + "{:>5}" + COLOR.END
        else:
            row_format += color + "{:>10}" + COLOR.END + "|" + color + "{:>5}" + COLOR.END

    logger_func(row_format.format(
        'mean',
        '-',
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
def compare_class_based3(logger_func, attribute_name, weight, result_label1, result_label2, result_label3, color=COLOR.BOLD):
    r""" log result and the difference between result_label1, result_label2 and result_label3
    Args:
        logger_func: logger.info or print
        attribute_name: list of attribute
        weight: weight of each attribute in testset
        result_label1, result_label2
        color: result higher will colored
    """
    # logger_func('class-based metrics:')
    result1 = np.stack([result_label1.accuracy, result_label1.mean_accuracy, result_label1.precision, result_label1.recall, result_label1.f1_score], axis=0)
    result1 = np.around(result1*100, 2)
    result1 = result1.transpose()

    result2 = np.stack([result_label2.accuracy, result_label2.mean_accuracy, result_label2.precision, result_label2.recall, result_label2.f1_score], axis=0)
    result2 = np.around(result2*100, 2)
    result2 = result2.transpose()

    result3 = np.stack([result_label3.accuracy, result_label3.mean_accuracy, result_label3.precision, result_label3.recall, result_label3.f1_score], axis=0)
    result3 = np.around(result3*100, 2)
    result3 = result3.transpose()

    row_format = "{:>20}{:>12}{:>17}{:>20}{:>25}{:>20}{:>24}"
    logger_func(row_format.format('attribute', 'weight', 'accuracy', 'mA', 'precision', 'recall', 'f1_score'))
    
    logger_func(row_format.format(*['-']*7))

    for i in range(len(attribute_name)):
        row_format = "{:>20}" + "{:>12}"
        for j in range(5):
            if np.argmax([result1[i][j], result2[i][j], result3[i][j]]) == 0:
                row_format += color + "{:>10}" + COLOR.END + "|{:>5}" + "|{:>5}"
            elif np.argmax([result1[i][j], result2[i][j], result3[i][j]]) == 1:
                row_format += "{:>10}|" + color + "{:>5}" + COLOR.END + "|{:>5}"
            elif np.argmax([result1[i][j], result2[i][j], result3[i][j]]) == 2:
                row_format += "{:>10}|" + "{:>5}|" + color + "{:>5}" + COLOR.END
        logger_func(row_format.format(attribute_name[i], np.around(weight[i]*100, 2), *array_interweave3(result1[i], result2[i], result3[i]).tolist()))

    row_format = "{:>20}{:>12}{:>17}{:>20}{:>25}{:>20}{:>24}"
    logger_func(row_format.format(*['-']*7))
    
    mean_result1 = np.around(np.mean(np.array([result_label1.accuracy, result_label1.mean_accuracy, result_label1.precision, result_label1.recall, result_label1.f1_score]), axis=1)*100, 2)
    
    mean_result2 = np.around(np.mean(np.array([result_label2.accuracy, result_label2.mean_accuracy, result_label2.precision, result_label2.recall, result_label2.f1_score]), axis=1)*100, 2)
    
    mean_result3 = np.around(np.mean(np.array([result_label3.accuracy, result_label3.mean_accuracy, result_label3.precision, result_label3.recall, result_label3.f1_score]), axis=1)*100, 2)

    row_format = "{:>20}" + "{:>12}"
    for i in range(5):
        if np.argmax([mean_result1[i], mean_result2[i], mean_result3[i]]) == 0:
            row_format += color + "{:>10}" + COLOR.END + "|{:>5}" + "|{:>5}"
        elif np.argmax([mean_result1[i], mean_result2[i], mean_result3[i]]) == 1:
            row_format += "{:>10}|" + color + "{:>5}" + COLOR.END + "|{:>5}"
        elif np.argmax([mean_result1[i], mean_result2[i], mean_result3[i]]) == 2:
            row_format += "{:>10}|" + "{:>5}|" + color + "{:>5}" + COLOR.END

    logger_func(row_format.format(
        'mean',
        '-',
        round(np.mean(result_label1.accuracy)*100, 2),
        round(np.mean(result_label2.accuracy)*100, 2),
        round(np.mean(result_label3.accuracy)*100, 2),
        round(np.mean(result_label1.mean_accuracy)*100, 2),
        round(np.mean(result_label2.mean_accuracy)*100, 2),
        round(np.mean(result_label3.mean_accuracy)*100, 2),
        round(np.mean(result_label1.precision)*100, 2),
        round(np.mean(result_label2.precision)*100, 2),
        round(np.mean(result_label3.precision)*100, 2),
        round(np.mean(result_label1.recall)*100, 2),
        round(np.mean(result_label2.recall)*100, 2),
        round(np.mean(result_label3.recall)*100, 2),
        round(np.mean(result_label1.f1_score)*100, 2),
        round(np.mean(result_label2.f1_score)*100, 2),
        round(np.mean(result_label3.f1_score)*100, 2)))

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
    resume2 = "/content/drive/Shared drives/REID/HIEN/Models/person_attribute_recognition/checkpoints/0809_231322/model_best_accuracy.pth"

    config1 = read_config(config1)
    config1.update({'resume': resume1})
    config1.update({'colab': True})

    config2 = read_config(config2)
    config2.update({'resume': resume2})
    config2.update({'colab': True})
    
    datamanager1, _ = build_datamanager(config1['type'], config1['data'])
    datamanager2, _ = build_datamanager(config2['type'], config2['data'])

    weight = datamanager1.datasource.get_weight('test')

    # model1
    result_label1, result_instance1 = test(config1, datamanager1, print)

    # model 2
    result_label2, result_instance2 = test(config2, datamanager2, print)

    compare_class_based(print, datamanager1.datasource.get_attribute(), weight, result_label1, result_label2, COLOR.BLUE)

