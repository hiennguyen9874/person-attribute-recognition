# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import math
import torch
import unittest
import torch.nn as nn
import matplotlib.pyplot as plt
from bisect import bisect_right


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    ''' Bag of tricks: https://arxiv.org/pdf/1903.07071.pdf
    '''
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    r""" Copy from here: https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/solver/lr_scheduler.py
    """
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iters: int,
            delay_iters: int = 0,
            eta_min_lr: int = 0,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch=-1,
            **kwargs
    ):
        self.max_iters = max_iters
        self.delay_iters = delay_iters
        self.eta_min_lr = eta_min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        assert self.delay_iters >= self.warmup_iters, "Scheduler delay iters must be larger than warmup iters"
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor,)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif self.last_epoch <= self.delay_iters:
            return self.base_lrs

        else:
            return [self.eta_min_lr + (base_lr - self.eta_min_lr) * (1 + math.cos(math.pi * (self.last_epoch - self.delay_iters) / (self.max_iters - self.delay_iters))) / 2 for base_lr in self.base_lrs]

def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

if __name__ == "__main__":
    x = []
    y = []
    net = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.00035)
    lr_scheduler = WarmupCosineAnnealingLR(optimizer, max_iters=120, delay_iters=30, eta_min_lr=0.00000077, warmup_factor=0.01, warmup_iters=10, warmup_method="linear")
    for i in range(120):
        lr_scheduler.step()
        for j in range(3):
            # print(i, lr_scheduler.get_lr()[0])
            x.append(i)
            y.append(lr_scheduler.get_lr()[0])
            optimizer.step()
    plt.plot(x, y)
    plt.show()
    
