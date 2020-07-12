import os
from numpy.core.fromnumeric import partition
import scipy.io
import numpy as np

import sys
sys.path.append('.')

from collections import defaultdict

from base import BaseDataSource

class Peta(BaseDataSource):
    ''' http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
    '''
    dataset_id = '13UvQ4N-sY67htGnK6qheb027XuMx9Jbr'
    group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]
  
    def __init__(self, root_dir='datasets', download=True, extract=True, validation_split=0.1):
        super(Peta, self).__init__(root_dir, dataset_dir = 'peta', file_name = 'PETA-New.zip', image_size = (256, 192))
        if download:
            print("Downloading!")
            self._download(dataset_id=self.dataset_id)
            print("Downloaded!")
        if extract:
            print("Extracting!")
            self._extract()
            print("Extracted!")
        
        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        f = scipy.io.loadmat(os.path.join(data_dir, 'PETA.mat'))
        
        raw_attr_name = [i[0][0] for i in f['peta'][0][0][1]]
        # raw_img_name = f['peta'][0][0][0][:, 0]
        raw_label = f['peta'][0][0][0][:, 4:]
        
        label = raw_label[:, :35][:, np.array(self.group_order)].astype(np.float32)
        self.attribute_name = [raw_attr_name[:35][i] for i in self.group_order]

        self.data = defaultdict(list)
        self.weight_train = []
        # self.weight_trainval = []

        for idx in range(5):
            _train = f['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
            _val = f['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
            _test = f['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
            # _trainval = np.concatenate((_train, _val), axis=0)

            self.data['train'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _train])
            self.data['val'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _val])
            # self.data['trainval'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _trainval])
            self.data['test'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _test])

            self.weight_train.append(np.mean(label[_train], axis=0))
            # self.weight_trainval.append(np.mean(label[_trainval], axis=0))

    def get_data(self, phase='train', partition=0):
        if phase == 'train':
            return self.data['train'][partition]
        elif phase == 'val':
            return self.data['val'][partition]
        elif phase == 'test':
            return self.data['test'][partition]
        else:
            raise ValueError('phase error, phase in [train, val, test]')

    def get_attribute(self, phase = 'train'):
        return self.attribute_name
    
    def get_weight(self, phase = 'train', partition=0):
        if phase == 'train':
            return self.weight_train[partition]
        raise ValueError('phase error, phase in [train]')
    
    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'images')) \
            and os.path.exists(os.path.join(extract_dir, 'README')) \
            and os.path.exists(os.path.join(extract_dir, 'PETA.mat')):
            return True
        return False

if __name__ == "__main__":
    datasource = Peta(root_dir='/home/hien/Documents/datasets')
    path = datasource.get_data()[0]
