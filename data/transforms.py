import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import random
import math
import numpy as np

from collections import deque
import torchvision.transforms.functional as F

from data.image.ppe import PPE
from data.image.pa_100k import PA_100K

__all__ = ['RandomErasing']

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
       
    def __call__(self, image, **kwargs):

        if random.uniform(0, 1) > self.probability:
            return image

        for attempt in range(100):
            area = image.size()[1] * image.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < image.size()[2] and h < image.size()[1]:
                x1 = random.randint(0, image.size()[1] - h)
                y1 = random.randint(0, image.size()[2] - w)
                if image.size()[0] == 3:
                    image[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    image[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    image[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    image[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return image

        return image

