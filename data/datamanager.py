import os
import json
import torchvision
import sys
sys.path.append('.')

import numpy as np

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from .datasets import ImageDataset
from .image import build_datasource
from .transforms import RandomErasing, RandomPatch

class DataManger(object):
    def __init__(self, config, phase='train'):
        super().__init__()
        self.data_name = config['name']

        self.datasource = build_datasource(
            name=self.data_name,
            root_dir=config['data_dir'],
            download=config['download'],
            extract=config['extract'],
            use_tqdm=False if 'tqdm' not in config else config['tqdm'])
        
        transform = dict()
        transform['train'] = transforms.Compose([
            transforms.Resize(size=self.datasource.get_image_size()),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            RandomPatch(),
            transforms.Pad(padding=10, fill=0, padding_mode='constant'),
            transforms.RandomCrop(size=self.datasource.get_image_size(), padding=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        
        transform['val'] = transforms.Compose([
            transforms.Resize(size=self.datasource.get_image_size()),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform['test'] = transforms.Compose([
            transforms.Resize(size=self.datasource.get_image_size()),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = dict()
        for _phase in ['train', 'val', 'test']:
            dataset[_phase] = ImageDataset(self.datasource.get_data(_phase), transform=transform[_phase])
        
        self.dataloader = dict()
        self.dataloader['train'] = DataLoader(
            dataset=dataset['train'],
            batch_size=config['batch_size'],
            shuffle=config['shuffle'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            drop_last=config['drop_last']
        )

        self.dataloader['val'] = DataLoader(
            dataset=dataset['val'],
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            drop_last=config['drop_last']
        )
        self.dataloader['test'] = DataLoader(dataset['test'], batch_size=32, shuffle=False, drop_last=False)

    def get_dataloader(self, phase):
        if phase not in ['train', 'val', 'test']:
            raise ValueError("Error phase paramaster, phase in [train, val, test]")
        return self.dataloader[phase]