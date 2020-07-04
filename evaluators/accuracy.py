import numpy as np

def top_k(distance, output, target, k=5):
    """ Computes top-k accuracy.
    Args:
        distance (torch.Tensor):    distance between query feature and output feature (query_size, output_size)
        output (torch.Tensor):      prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor):  ground truth labels with shape (batch_size).
        topk (tuple, optional):     accuracy at top-k will be computed.
    Returns:
        list: accuracy at top-k.
    """
    top = np.asarray(distance.topk(k=k, dim=1, largest=False).indices)
    predict_labels = output.take(top)
    target = target.reshape(-1, 1)
    r = target == predict_labels
    return np.any(r, axis=1).mean()


def AP(top_predict_result):
    """ Computes AP.
    https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    Args:
        top_predict_result (1d array of boolean):
    Returns:
        float: AP
    """
    hit = 0
    total = 0
    for i, r in zip(range(len(top_predict_result)), top_predict_result):
        if r:
            total += 1
            hit += total / (i+1)
    if total == 0:
        return 0
    return hit / total


def mAP(distance, output, target, k=5):
    """ Computes top-k accuracy.
    https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    Args:
        distance (torch.Tensor):    distance between query feature and output feature (quey_size, output_size)
        output (torch.Tensor):      prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor):  ground truth labels with shape (batch_size).
        k (int): topk, 'all'
    Returns:
        list: accuracy at top-k.
    """
    if type(k) != int and k.lower() == 'all':
        k = len(output)
    top = np.asarray(distance.topk(k=k, dim=1, largest=False).indices)
    predict_labels = output.take(top)
    target = target.reshape(-1, 1)
    r = target == predict_labels
    r = np.apply_along_axis(AP, 1, r)
    return r.mean()
    # total_ap = 0
    # for idx_query in range(len(target)):
    #     hit, total = 0, 0
    #     for idx_predict in range(len(predict_labels[idx_query])):
    #         if target[idx_query] == predict_labels[idx_query][idx_predict]:
    #             total += 1
    #             hit += total/(idx_predict+1)
    #     if total == 0:
    #         total_ap += 0
    #     else:
    #         total_ap += float(hit)/total
    # return total_ap/len(target)
