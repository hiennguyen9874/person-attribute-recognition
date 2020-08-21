import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import torch

from utils import imread

__all__ = ['ImageDataset']

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        if isinstance(index, int):
            img_path, label = self.data[index]
            img = imread(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        elif isinstance(index, tuple):
            index, attribute_idx = index
            img_path, label = self.data[index]
            img = imread(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, label, attribute_idx

    def __len__(self):
        return len(self.data)