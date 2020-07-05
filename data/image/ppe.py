
import numpy as np
import scipy.io
import glob
import re
import zipfile
import tarfile
import requests
import os
import sys
sys.path.append('.')

from tqdm import tqdm, tnrange
from collections import defaultdict

class PPE(object):
    dataset_dir = 'ppe'
    file_name = 'ppe.zip'
    
    def __init__(self, root_dir='datasets', download=True, extract=True, validation_split=0.1):
        self.root_dir = root_dir
        if download:
            print("Downloading!")
            self._download()
            print("Downloaded!")
        if extract:
            print("Extracting!")
            self._extract()
            print("Extracted!")

        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed', 'ppe')
        data, self.attr_name = self._processes_dir(data_dir)

        # split data
        idx_full = np.arange(len(data))
        np.random.shuffle(idx_full)
        len_valid = int(len(data) * validation_split)
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        self.train = [data[idx] for idx in train_idx.tolist()]
        self.val = [data[idx] for idx in valid_idx.tolist()]
        self.weight_train = np.asarray(self.train[0][1])
        for _, _attribute_label in self.train[1:]:
            self.weight_train += np.asarray(_attribute_label)
        self.weight_train = np.divide(self.weight_train, int(len(self.train)))
        pass

    def _processes_dir(self, data_dir):
        all_attribute = set()
        all_data = []
        with open(os.path.join(data_dir, 'detection_ppe.txt'), 'r') as f:
            for i, line in enumerate(tqdm(f, total=7679)):
                temp = line.split(',')
                temp[-1] = temp[-1][0:-1]
                file_path = os.path.join(data_dir, temp[0])
                id = int(temp[1])
                j = 2
                dict_attribute = dict()
                while j < len(temp):
                    all_attribute.add(temp[j])
                    dict_attribute[temp[j]] = list(map(float, temp[j+1:j+5]))
                    j+=5
                all_data.append((file_path, list(dict_attribute.keys())))
        data = []
        for _sampler in all_data:
            attribute_label = dict()
            for _attribute in all_attribute:
                if _attribute in _sampler[1]:
                    attribute_label[_attribute] = 1
                else:
                    attribute_label[_attribute] = 0
            data.append((_sampler[0], np.array(list(attribute_label.values())).astype(np.float32)))
        return data, all_attribute

    def get_data(self, mode='train'):
        if mode == 'train':
            return self.train
        elif mode == 'val':
            return self.val
        else:
            raise ValueError('mode error')
        
    def get_attribute(self):
        return self.attr_name
    
    def get_weight(self, mode = 'train'):
        if mode == 'train':
            return self.weight_train
        raise ValueError('mode error, mode in [train]')

    def _download(self):
        os.makedirs(os.path.join(self.root_dir, self.dataset_dir, 'raw'), exist_ok=True)
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)):
            raise FileExistsError('please download file into %s' % (os.path.join(self.root_dir, self.dataset_dir, 'raw')))

    def _extract(self):
        file_path = os.path.join(
            self.root_dir, self.dataset_dir, 'raw', self.file_name)
        extract_dir = os.path.join(
            self.root_dir, self.dataset_dir, 'processed')
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
        if os.path.exists(os.path.join(extract_dir, 'ppe', 'ppe')) \
                and os.path.exists(os.path.join(extract_dir, 'ppe', 'ppe_200617')) \
                and os.path.exists(os.path.join(extract_dir, 'ppe', 'ppe_test')) \
                and os.path.exists(os.path.join(extract_dir, 'ppe', 'detection_ppe.txt')):
            return True
        return False

# if __name__ == "__main__":
#     datasource = PPE(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
#     pass