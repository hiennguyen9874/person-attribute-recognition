import os
import pickle
import numpy as np

from collections import defaultdict

import sys
sys.path.append('.')

from base import BaseDataSource
from utils import read_json, neq

class Wider(BaseDataSource):
    dataset_id = '1whFSGBMLm-92SQ6JXEAJwM49XrmQhUd5'

    def __init__(self, root_dir='datasets', download=True, extract=True, use_tqdm=True, validation_split=0.1):
        dataset_dir = 'wider'
        file_name = 'Wider-data.zip'
        super(Wider, self).__init__(root_dir, dataset_dir, file_name, image_size = (256, 256))
        if download:
            self._download(self.dataset_id, use_tqdm=use_tqdm)
        if extract:
            self._extract(use_tqdm=use_tqdm)

        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed', 'Wider-data')
        
        self.data = dict()
        self.weight = dict()
        self.data['train'], attribute_train, self.weight['train'] = self._processes_dir(os.path.join(data_dir, 'wiger_attribute_train.json'), data_dir)
        self.data['val'], attribute_val, self.weight['val'] = self._processes_dir(os.path.join(data_dir, 'wiger_attribute_val.json'), data_dir)
        self.data['test'], attribute_test, self.weight['test'] = self._processes_dir(os.path.join(data_dir, 'wiger_attribute_test.json'), data_dir)

        assert neq(len(attribute_train), len(attribute_val), len(attribute_test))
        for i in range(len(attribute_train)):
            if not neq(list(attribute_train.keys())[i], list(attribute_val.keys())[i], list(attribute_test.keys())[i]):
                raise ValueError
            if not neq(list(attribute_train.values())[i], list(attribute_val.values())[i], list(attribute_test.values())[i]):
                raise ValueError

        self.attribute_name = list(attribute_train.values())
        
    def _processes_dir(self, path_json_file, data_dir):
        folder_name = path_json_file.split('/')[-1].split('.')[0].split('_')[-1]
        wider_attribute = read_json(path_json_file)
        attribute_name = wider_attribute['attribute_id_map']
        data = list()
        weight = np.zeros(len(attribute_name))
        for value in wider_attribute['images']:
            path = value['file_name'].split('/')[-1]
            path = os.path.join(data_dir, folder_name, path)
            if not os.path.exists(path):
                raise FileExistsError('{}'.format(path))
            attribute = np.array(value['attribute'])
            attribute[attribute == -1] = 0
            attribute = attribute.astype(np.float32)
            weight += attribute
            data.append((path, attribute))
        return data, attribute_name, np.divide(weight, int(len(data)))
    
    def get_attribute(self):
        return self.attribute_name

    def get_data(self, phase='train'):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        return self.data[phase]
    
    def get_weight(self, phase = 'train'):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        return self.weight[phase]

    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'Wider-data', 'train')) \
            and os.path.exists(os.path.join(extract_dir, 'Wider-data', 'val')) \
            and os.path.exists(os.path.join(extract_dir, 'Wider-data', 'test')) \
            and os.path.exists(os.path.join(extract_dir, 'Wider-data', 'wiger_attribute_train.json')) \
            and os.path.exists(os.path.join(extract_dir, 'Wider-data', 'wiger_attribute_val.json')) \
            and os.path.exists(os.path.join(extract_dir, 'Wider-data', 'wiger_attribute_test.json')):
            return True
        return False
    
    def save_attribute(self, path='attribute.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.get_attribute(), f)

if __name__ == "__main__":
    datasource = Wider(root_dir='/datasets', download=False, extract=True)