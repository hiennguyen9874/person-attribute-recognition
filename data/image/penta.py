import sys
sys.path.append('.')

import os
import scipy.io
import numpy as np

from base import BaseDataSource

class Penta(BaseDataSource):
    ''' http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
    '''
    dataset_id = '13UvQ4N-sY67htGnK6qheb027XuMx9Jbr'
    group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]
  
    def __init__(self, root_dir='datasets', download=True, extract=True):
        dataset_dir = 'penta'
        file_name = 'PETA-New.zip'
        list_phases = ['train', 'val', 'test']
        super(Penta, self).__init__(root_dir, dataset_dir, file_name, list_phases)
        if download:
            print("Downloading!")
            self._download()
            print("Downloaded!")
        if extract:
            print("Extracting!")
            self._extract()
            print("Extracted!")
        
        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        peta_data = scipy.io.loadmat(os.path.join(data_dir, 'PETA.mat'))
        
        raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
        raw_label = peta_data['peta'][0][0][0][:, 4:]
        
        label = raw_label[:, :35][:, np.array(self.group_order)].astype(np.float32)
        self.attr_name = [raw_attr_name[:35][i] for i in self.group_order]

        self.train = []
        self.val = []
        self.trainval = []
        self.test = []
        self.weight_train = []
        self.weight_trainval = []

        for idx in range(5):
            _train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
            _val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
            _test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
            _trainval = np.concatenate((_train, _val), axis=0)

            self.train.append([(os.path.join(data_dir, 'images', '%05d.png'%(idx)), label[idx]) for idx in _train])
            self.val.append([(os.path.join(data_dir, 'images', '%05d.png'%(idx)), label[idx]) for idx in _val])
            self.trainval.append([(os.path.join(data_dir, 'images', '%05d.png'%(idx)), label[idx]) for idx in _trainval])
            self.test.append([(os.path.join(data_dir, 'images', '%05d.png'%(idx)), label[idx]) for idx in _test])

            self.weight_train.append(np.mean(label[_train], axis=0))
            self.weight_trainval.append(np.mean(label[_trainval], axis=0))

    def get_data(self, phase='train'):
        if phase == 'train':
            return self.train[0]
        elif phase == 'val':
            return self.val[0]
        elif phase == 'train_val':
            return self.trainval[0]
        elif phase == 'test':
            return self.test[0]
        raise ValueError('phase error, phase in [train, val, train_val, test]')

    def get_attribute(self, phase = 'train'):
        return self.attr_name
    
    def get_weight(self, phase = 'train'):
        if phase == 'train':
            return self.weight_train[0]
        elif phase == 'train_val':
            return self.weight_trainval[0]
        raise ValueError('phase error, phase in [train, val]')
    
    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'images')) \
            and os.path.exists(os.path.join(extract_dir, 'README')) \
            and os.path.exists(os.path.join(extract_dir, 'PETA.mat')):
            return True
        return False

if __name__ == "__main__":
    datasource = Penta(root_dir='/home/hien/Documents/datasets')
    path = datasource.get_data()[0]
