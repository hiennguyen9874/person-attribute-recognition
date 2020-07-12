import torch
import numpy as np
from easydict import EasyDict

def recognition_metrics(labels, preds, threshold=0.5, eps = 1e-20):
    ''' https://en.wikipedia.org/wiki/Confusion_matrix
    Args:
        labels (num_sampler, num_attribute): 2d numpy array binary
        preds (num_sampler, num_attribute): 2d numpy array float
    '''
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


def compute_accuracy_cuda(labels, preds, eps=1e-20):
    labels = labels.type(torch.BoolTensor)
    preds = preds.type(torch.BoolTensor)
    intersect = (preds & labels).type(torch.FloatTensor)
    union = (preds | labels).type(torch.FloatTensor)
    return torch.mean((torch.sum(intersect, dim=1) / (torch.sum(union, dim=1) + eps)))
