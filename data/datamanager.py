import sys
sys.path.append('.')

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from data.datasets import Epoch_ImageDataset, Episode_ImageDataset
from data.image import build_datasource
from data.transforms import RandomErasing
from data.samplers import RandomBalanceBatchSamplerAttribute

__all__ = ['DataManger_Epoch', 'DataManger_Episode']

class DataManger_Epoch(object):
    def __init__(self, config, phase='train'):
        super().__init__()
        self.data_name = config['name']

        self.datasource = build_datasource(
            name=self.data_name,
            root_dir=config['data_dir'],
            download=config['download'],
            extract=config['extract'],
            use_tqdm=config['use_tqdm'])
        
        transform = dict()
        transform['train'] = transforms.Compose([
            transforms.Resize(size=self.datasource.get_image_size()),
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
        for _phase in self.datasource.get_phase():
            dataset[_phase] = Epoch_ImageDataset(self.datasource.get_data(_phase), transform=transform[_phase])
        
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

        self.dataloader['test'] = DataLoader(
            dataset['test'],
            batch_size=32,
            shuffle=False,
            drop_last=False
        )

    def get_dataloader(self, phase):
        if phase not in self.datasource.get_phase():
            raise ValueError("Error phase paramaster, phase in %s" % str(self.datasource.get_phase()))
        return self.dataloader[phase]


class DataManger_Episode(object):
    def __init__(self, config):
        super().__init__()
        self.data_name = config['name']

        self.datasource = build_datasource(
            name=self.data_name,
            root_dir=config['data_dir'],
            download=config['download'],
            extract=config['extract'],
            use_tqdm=config['use_tqdm'])
        
        transform = dict()
        transform['train'] = transforms.Compose([
            transforms.Resize(size=self.datasource.get_image_size()),
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
        for _phase in self.datasource.get_phase():
            dataset[_phase] = Episode_ImageDataset(
                self.datasource.get_data(_phase),
                self.datasource.get_attribute(),
                transform=transform[_phase])

        sampler = dict()
        sampler['train'] = RandomBalanceBatchSamplerAttribute(
            self.datasource.get_data('train'),
            self.datasource.get_attribute(),
            num_attribute=config['train']['num_attribute'],
            num_instance=config['train']['num_instance'],
            num_iterator=config['train']['num_iterator'],
            selected_ratio=config['train']['selected_ratio']
        )

        sampler['val'] = RandomBalanceBatchSamplerAttribute(
            self.datasource.get_data('val'),
            self.datasource.get_attribute(),
            num_attribute=config['val']['num_attribute'],
            num_instance=config['val']['num_instance'],
            num_iterator=config['val']['num_iterator'],
            selected_ratio=config['val']['selected_ratio']
        )

        self.dataloader = dict()
        self.dataloader['train'] = DataLoader(
            dataset=dataset['train'],
            batch_sampler=sampler['train'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        self.dataloader['val'] = DataLoader(
            dataset=dataset['val'],
            batch_sampler=sampler['val'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
        )

        self.dataloader['test'] = DataLoader(
            dataset['test'],
            batch_size=32,
            shuffle=False,
            drop_last=False
        )

    def get_dataloader(self, phase):
        if phase not in self.datasource.get_phase():
            raise ValueError("Error phase paramaster, phase in %s" % str(self.datasource.get_phase()))
        return self.dataloader[phase]
