import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_distance_matrix(input1, input2, metrix='euclidean'):
    """Computes distance of two matrix.
    Args:
        output (torch.Tensor): 2d feature matrix.  (a, feature_dim)
        target (torch.Tensor): 2d feature matrix.  (b, feature_dim)
        metrix (str, optional): 'euclidean', 'cosine'. Default is 'euclidean'   => (a, b)
    Returns:
        tensor: distance matrix
    """
    if metrix.lower() == 'euclidean':
        return euclidean_squared_distance(input1, input2)
    elif metrix.lower() == 'cosine':
        return cosine_distance(input1, input2)
    else:
        raise SyntaxError('Metrix')


def cosine_distance(input1, input2):
    """Computes cosine distance.
    Args:
        output (torch.Tensor): 2d feature matrix.
        target (torch.Tensor): 2d feature matrix.
    Returns:
        tensor: distance matrix
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean distance.
    Args:
        output (torch.Tensor): 2d feature matrix.
        target (torch.Tensor): 2d feature matrix.
    Returns:
        tensor: distance matrix
    """
    return torch.cdist(input1, input2)


def pairwise_distances(mat1, mat2=None):
    """Bo con distance.
    Args:
        output (torch.Tensor): 2d feature matrix.
        target (torch.Tensor): 2d feature matrix.
    Returns:
        tensor: distance matrix
    """
    if mat2 is None:
        mat2 = mat1

    a = torch.mm(mat1, mat1.t()).diag().view(-1, 1)
    b = torch.mm(mat2, mat2.t()).diag().view(1, -1)
    c = torch.mm(mat1, mat2.t())

    r = a + b - 2*c
    return torch.sqrt(r)
