import os
import torch
import json
# import cv2
import random
import torch
import numpy as np
import torchvision.datasets as datasets

from tqdm import tqdm
from PIL import Image

def imread(path):
    image = Image.open(path)
    return image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = imread(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)
