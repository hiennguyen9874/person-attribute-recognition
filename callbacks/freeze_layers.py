import torch
import torch.nn as nn

class FreezeLayers(object):
    r""" freeze layers when training
    Args:
        model (nn.Module)
        freeze_layers (list): list of layer will frozen
        freeze_epochs (int): num of epoch will frozen
    """
    def __init__(self, model, freeze_layers, freeze_epochs):
        self.model = model
        self.freeze_layers = freeze_layers
        self.freeze_epochs = freeze_epochs

    def _freeze(self):
        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                for param in module.parameters():
                    param.requires_grad = False
    
    def _unfreeze(self):
        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                for param in module.parameters():
                    param.requires_grad = True

    def on_epoch_begin(self, epoch):
        if epoch == 1:
            self._freeze()
        if epoch == self.freeze_epochs:
            self._unfreeze()
