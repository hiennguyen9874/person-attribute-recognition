import sys
from typing import Iterator
sys.path.append('.')

import copy
import torch
import random
import math
import numpy as np

from PIL import Image
from torchvision.transforms import *
import torchvision.transforms.functional as F

from data.image.ppe import PPE
from data.image.pa_100k import PA_100K

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class RemoveZero(object):
    def __init__(self, interpolation=Image.BILINEAR):
        self.interpolation = interpolation

    def __call__(self, img):
        img_origin = copy.deepcopy(img)
        img = np.array(F.to_grayscale(img))
        y_sum = np.where(np.sum(img, axis=0) == 0)
        x_sum = np.where(np.sum(img, axis=1) == 0)

        temp1 = np.split(y_sum, np.where(np.diff(y_sum) != 1)[0]+1)
        begin1 = temp1[0][0][1]
        end1 = temp1[0][0][-1]


        img = img.astype(bool)
        return img

class Resize(object):
    def __init__(self, height: int, width: int, interpolation=Image.BILINEAR):
        assert isinstance(width, int) and isinstance(height, int)
        self.width = width
        self.height = height
        self.interpolation = interpolation
    
    def __call__(self, img):
        w, h = img.size
        if w <= h:
            ow = self.width
            oh = int(self.width * h / w)
            img = img.resize((ow, oh), self.interpolation)
        else:
            oh = self.height
            ow = int(self.height * w / h)
            img = img.resize((ow, oh), self.interpolation)
        return img

class Fill(object):
    def __init__(self, height: int, width: int, fill=0, interpolation=Image.BILINEAR):
        assert isinstance(width, int) and isinstance(height, int)
        self.width = width
        self.height = height
        self.fill = fill
        self.interpolation = interpolation
    
    def __call__(self, img):
        w, h = img.size
        w_pad = ((self.width-w)//2, self.width - w - (self.width-w)//2)
        h_pad = ((self.height-h)//2, self.height - h - (self.height-h)//2)
        return F.pad(img, (w_pad[0], h_pad[0], w_pad[1], h_pad[1]), fill=self.fill)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    datasource = PPE(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    # img_path, label = datasource.get_data('train')[100]
    # image = Image.open(img_path)
    # print(image.size[0], image.size[1])
    # plt.imshow(np.asarray(image))
    # plt.show()
    # image = RemoveZero()(image)
    # print(image.size[0], image.size[1])
    # plt.imshow(np.asarray(image))
    # plt.show()
    # image = Resize(256, 128, 0)(image)
    # print(image.size[0], image.size[1])
    # plt.imshow(np.asarray(image))
    # plt.show()
    # image = Fill(256, 128, 0)(image)
    # print(image.size[0], image.size[1])
    # plt.imshow(np.asarray(image))
    # plt.show()
    # pass

    for img_path, label in datasource.get_data('train'):
        if Image.open(img_path).size[0] != Image.open(img_path).size[1]:
            print(Image.open(img_path).size[0], "-", Image.open(img_path).size[1], "-", img_path)