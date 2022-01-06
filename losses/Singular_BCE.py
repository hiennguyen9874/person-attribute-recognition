import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F


class Singular_BCE(nn.Module):
    r"""Binary cross entropy only on idx_attribute"""

    def __init__(self, num_attribute, reduction="sum"):
        super(Singular_BCE, self).__init__()
        assert reduction in ["sum", "mean"]
        self.reduction = "mean" if reduction == "attribute" else reduction
        self.num_attribute = num_attribute if reduction == "attribute" else 1

    def forward(self, logits, targets, idx_attribute):
        return (
            F.binary_cross_entropy_with_logits(
                logits.gather(1, idx_attribute.unsqueeze(1)).squeeze(),
                targets.gather(1, idx_attribute.unsqueeze(1)).squeeze(),
                reduction=self.reduction,
            )
            * self.num_attribute
        )
