import os
import json
import torchvision
import sys
sys.path.append('.')

import numpy as np

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .datasets import ImageDataset
from .samplers import SubsetIdentitySampler, RandomIdentitySampler
from .image import PA_100K, Penta, PPE
from .transforms import RandomErasing

class DataManger(object):
    __datasets = {'pa100k': PA_100K, 'ppe': PPE}
    def __init__(self, config, phase='train', data_name='pa100k'):
        super().__init__()

        assert data_name in list(self.__datasets.keys())
        self.data_name = data_name

        self.datasource = self.__datasets[self.data_name](
            root_dir=config['data_dir'],
            download=config['download'],
            extract=config['extract'])
        
        transform = dict()
        transform['train'] = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.Pad(padding=10, fill=0, padding_mode='constant'),
            transforms.RandomCrop(size=(256, 128), padding=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform['test'] = transforms.Compose([
            transforms.Resize(size=(256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = dict()
        dataset['train'] = ImageDataset(self.datasource.get_data('train'), transform=transform['train'])
        dataset['val'] = ImageDataset(self.datasource.get_data('val'), transform=transform['test'])
        
        if data_name in ['pa100k']:
            dataset['test'] = ImageDataset(self.datasource.get_data('test'), transform=transform['test'])

        if phase == 'train':
            self.train_loader = DataLoader(
                dataset=dataset['train'],
                batch_size=config['batch_size'],
                shuffle=config['shuffle'],
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                drop_last=config['drop_last']
            )

            self.val_loader = DataLoader(
                dataset=dataset['val'],
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                drop_last=config['drop_last']
            )
        elif phase == 'test':
            if data_name in ['pa100k']:
                self.test_loader = DataLoader(dataset['test'], batch_size=32, shuffle=False, drop_last=False)
        else:
            raise ValueError("phase == train or phase == test")

    def get_dataloader(self, dataset):
        if dataset not in ['train', 'val', 'test']:
            raise ValueError("Error dataset paramaster, dataset in [train, val, test]")
        if dataset == 'train':
            return self.train_loader
        elif dataset == 'val':
            return self.val_loader
        elif dataset == 'test':
            return self.test_loader