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
    __datasets = {'pa_100k': PA_100K, 'penta': Penta, 'ppe': PPE}

    def __init__(self, config, phase='train'):
        super().__init__()

        assert config['name'] in list(self.__datasets.keys())
        self.data_name = config['name']

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
        
        transform['val'] = transforms.Compose([
            transforms.Resize(size=(256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if 'test' in self.datasource.get_list_phase():
            transform['test'] = transforms.Compose([
                transforms.Resize(size=(256, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        dataset = dict()
        for _phase in self.datasource.get_list_phase():
            dataset[_phase] = ImageDataset(self.datasource.get_data(_phase), transform=transform[_phase])

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
            if 'test' in self.datasource.get_list_phase():
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