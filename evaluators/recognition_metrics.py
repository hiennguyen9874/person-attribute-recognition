import sys
sys.path.append('.')

import torch
import numpy as np
from easydict import EasyDict

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

def log_test(logger_func, attribute_name: list, labels, preds):
    r""" log test from result
    """
    result_label, result_instance = recognition_metrics(labels, preds)

    logger_func('instance-based metrics:')
    logger_func('accuracy: %0.4f' % result_instance.accuracy)
    logger_func('precision: %0.4f' % result_instance.precision)
    logger_func('recall: %0.4f' % result_instance.recall)
    logger_func('f1_score: %0.4f' % result_instance.f1_score)

    logger_func('class-based metrics:')
    result = np.stack([result_label.accuracy, result_label.mean_accuracy, result_label.precision, result_label.recall, result_label.f1_score], axis=0)
    result = np.around(result*100, 2)
    result = result.transpose()
    row_format ="{:>17}" * 6
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


if __name__ == "__main__":
    pass