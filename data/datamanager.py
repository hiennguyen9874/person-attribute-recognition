import os
import json
import torchvision

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from .datasets import ImageDataset
from .samplers import SubSetSampler, RandomIdentitySampler
from .image import PA_100K, Penta
from .transforms import RandomErasing

class DataManger(object):
    def __init__(self, config, phase='train'):
        super().__init__()
        self.datasource = PA_100K(
            root_dir=config['data_dir'],
            download=config['download'],
            extract=config['extract'])

        transform_train = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.Pad(padding=10, fill=0, padding_mode='constant'),
            transforms.RandomCrop(size=(256, 128), padding=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(size=(256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        training_set = ImageDataset(self.datasource.get_data('train'), transform=transform_train)
        val_set = ImageDataset(self.datasource.get_data('val'), transform=transform_test)
        test_set = ImageDataset(self.datasource.get_data('test'), transform=transform_test)

        if phase == 'train':
            self.train_loader = DataLoader(
                dataset=training_set,
                batch_size=config['batch_size'],
                shuffle=config['shuffle'],
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                drop_last=config['drop_last']
            )

            self.val_loader = DataLoader(
                dataset=val_set,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                drop_last=config['drop_last']
            )
        elif phase == 'test':
            self.test_loader = DataLoader(test_set['query'], batch_size=32, shuffle=False, drop_last=False)
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
