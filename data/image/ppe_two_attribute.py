import os
import numpy as np

from collections import defaultdict

import sys
sys.path.append('.')

from base import BaseDataSource

class PPE_Two(BaseDataSource):
    map_folder = {'ppe_200617': 'train', 'ppe': 'train', 'ppe_test': 'test'}
    attribute_name = ['hard_hat', 'safety_vest']
    
    def __init__(self, root_dir='datasets', download=True, extract=True, use_tqdm=True, validation_split=0.1):
        dataset_dir = 'ppe'
        file_name = 'ppe.zip'
        super(PPE_Two, self).__init__(root_dir, dataset_dir, file_name, image_size = (256, 256))
        if download:
            self._download(use_tqdm=use_tqdm)
        if extract:
            self._extract(use_tqdm=use_tqdm)

        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed', 'ppe')
        data = self._processes_dir(data_dir)
        
        # split data
        idx_full = np.arange(len(data['train']))
        np.random.shuffle(idx_full)
        len_valid = int(len(data['train']) * validation_split)
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        self.data = dict()
        self.data['train'] = [data['train'][idx] for idx in train_idx.tolist()]
        self.data['val'] = [data['train'][idx] for idx in valid_idx.tolist()]
        self.data['test'] = data['test']

        self.weight = dict()
        # compute weight
        for phase in ['train', 'val', 'test']:
            self.weight[phase] = np.zeros((len(self.attribute_name)))
            for _, _attribute_label in self.data[phase]:
                self.weight[phase] += _attribute_label
            self.weight[phase] = np.divide(self.weight[phase], int(len(self.data[phase])))

    def _processes_dir(self, data_dir):
        all_attribute = set()
        all_data = defaultdict(list)
        # print('processing detection_ppe.txt')
        with open(os.path.join(data_dir, 'detection_ppe.txt'), 'r') as f:
            # for i, line in enumerate(tqdm(f, total=7678)):
            for line in f:
                splited_line = line.split(',')
                splited_line[-1] = splited_line[-1][0:-1]
                file_folder = splited_line[0].split('/')[0]
                file_path = os.path.join(data_dir, splited_line[0])
                id = int(splited_line[1])
                j = 2
                dict_attribute = dict()
                while j < len(splited_line):
                    all_attribute.add(splited_line[j])
                    dict_attribute[splited_line[j]] = list(map(float, splited_line[j+1:j+5]))
                    j+=5
                all_data[file_folder].append((file_path, list(dict_attribute.keys())))
        # all_attribute.add('no_hat')
        # all_attribute.add('no_vest')
        # if len(all_attribute.difference(set(self.attribute_name))) != 0:
        #     raise KeyError('attribute wrong')
        
        data = defaultdict(list)
        for key, value in all_data.items():
            for _sampler in value:
                attribute_label = dict()
                for _attribute in self.attribute_name:
                    if _attribute in _sampler[1]:
                        attribute_label[_attribute] = 1
                    else:
                        attribute_label[_attribute] = 0
                # if attribute_label['hard_hat'] == 0 and attribute_label['none_hard_hat'] == 0:
                #     attribute_label['no_hat'] = 1
                # if attribute_label['safety_vest'] == 0 and attribute_label['none_safety_vest'] == 0:
                #     attribute_label['none_safety_vest'] = 1
                data[self.map_folder[key]].append((_sampler[0], np.array(list(attribute_label.values())).astype(np.float32)))
        return data
    
    def get_attribute(self):
        return self.attribute_name

    def get_data(self, phase='train'):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        return self.data[phase]
    
    def get_weight(self, phase = 'train'):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        return self.weight[phase]

    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'ppe', 'ppe')) \
            and os.path.exists(os.path.join(extract_dir, 'ppe', 'ppe_200617')) \
            and os.path.exists(os.path.join(extract_dir, 'ppe', 'ppe_test')) \
            and os.path.exists(os.path.join(extract_dir, 'ppe', 'detection_ppe.txt')):
            return True
        return False

if __name__ == "__main__":
    datasource = PPE_Two(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    # print('num image train:', len(datasource.get_data('train')))
    # print('num image val', len(datasource.get_data('val')))
    # print('num image test', len(datasource.get_data('test')))

    print(datasource.get_weight('train'))
    print(datasource.get_weight('val'))
    print(datasource.get_weight('test'))
    
