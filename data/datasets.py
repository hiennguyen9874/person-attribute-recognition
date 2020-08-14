import sys
sys.path.append('.')
import torch
import torchvision.datasets as datasets

from PIL import Image

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