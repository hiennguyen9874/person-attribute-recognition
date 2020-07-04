import torch
import torch.nn as nn

import sys
sys.path.append('.')

class CrossEntropyLabelSmooth(nn.Module):
    """ Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) # one-hot encoding
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(dim=0).sum()
        return loss

if __name__ == "__main__":
    loss = CrossEntropyLabelSmooth(num_classes=3, use_gpu=False)
    input = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.05, 0.05, 0.9]])
    target = torch.tensor([0, 1, 2], dtype=torch.int32).type(torch.LongTensor)
    # output = loss(input, target)
    # print(output)
    print(torch.nn.functional.softmax(input, dim=1))
    print(torch.nn.functional.log_softmax(input, dim=1))
    print(torch.nn.functional.cross_entropy(input, target))
    print(loss(input, target))