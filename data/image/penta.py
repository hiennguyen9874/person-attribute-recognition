import sys
sys.path.append('.')

import os
import requests
import tarfile
import zipfile
import re
import glob
import scipy.io
import numpy as np
from tqdm import tqdm, tnrange  
from collections import defaultdict

from utils import download_with_url

class Penta(object):
    dataset_dir = 'penta'
    dataset_id = '13UvQ4N-sY67htGnK6qheb027XuMx9Jbr'
    file_name = 'PETA-New.zip'
    list_phases = ['train', 'val', 'test']
    google_drive_api = 'AIzaSyAVfS-7Dy34a3WjWgR509o-u_3Of59zizo'
    group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]
  
    def __init__(self, root_dir='datasets', download=True, extract=True):
        self.root_dir = root_dir
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
    
    def get_list_phase(self):
        return self.list_phases

    def _download(self):
        os.makedirs(os.path.join(self.root_dir, self.dataset_dir, 'raw'), exist_ok=True)
        download_with_url(self.google_drive_api, self.dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), self.file_name)

    def _extract(self):
        file_path = os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)
        extract_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        if self._exists(extract_dir):
            return
        try:
            tar = tarfile.open(file_path)
            os.makedirs(extract_dir, exist_ok=True)
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                tar.extract(member=member, path=extract_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(file_path, 'r')
            for member in tqdm(iterable=zip_ref.infolist(), total=len(zip_ref.infolist())):
                zip_ref.extract(member=member, path=extract_dir)
            zip_ref.close()

    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'images')) \
            and os.path.exists(os.path.join(extract_dir, 'README')) \
            and os.path.exists(os.path.join(extract_dir, 'PETA.mat')):
            return True
        return False

if __name__ == "__main__":
    datasource = Penta(root_dir='/home/hien/Documents/datasets')
    path = datasource.get_data()[0]
