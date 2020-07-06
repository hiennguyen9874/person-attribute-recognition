from shutil import Error
import zipfile
import tarfile
import os
import numpy as np

from collections import defaultdict

import sys
sys.path.append('.')

from tqdm import tqdm

class PPE(object):
    dataset_dir = 'ppe'
    file_name = 'ppe.zip'
    list_phases = ['train', 'val', 'test']
    map_folder = {'train': 'ppe_200617', 'val': 'ppe', 'test': 'ppe_test'}

    attribute_name = ['hard_hat', 'none_hard_hat', 'safety_vest', 'none_safety_vest', 'no_hat', 'no_vest']
    
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
        self.data = self._processes_dir(data_dir)        
        self.weight_train = np.zeros((len(self.attribute_name)))
        for _, _attribute_label in self.data[self.map_folder['train']]:
            self.weight_train += _attribute_label
        self.weight_train = np.divide(self.weight_train, int(len(self.data[self.map_folder['train']])))

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
        all_attribute.add('no_hat')
        all_attribute.add('no_vest')
        if len(all_attribute.difference(set(self.attribute_name))) != 0:
            raise Error('attribute wrong')
        
        data = defaultdict(list)
        for key, value in all_data.items():
            for _sampler in value:
                attribute_label = dict()
                for _attribute in self.attribute_name:
                    if _attribute in _sampler[1]:
                        attribute_label[_attribute] = 1
                    else:
                        attribute_label[_attribute] = 0
                if attribute_label['hard_hat'] == 0 and attribute_label['none_hard_hat'] == 0:
                    attribute_label['no_hat'] = 1
                if attribute_label['safety_vest'] == 0 and attribute_label['none_safety_vest'] == 0:
                    attribute_label['no_vest'] = 1
                data[key].append((_sampler[0], np.array(list(attribute_label.values())).astype(np.float32)))
        return data

    def get_data(self, phase='train'):
        if phase == 'train':
            return self.data[self.map_folder['train']]
        elif phase == 'val':
            return self.data[self.map_folder['val']]
        elif phase == 'test':
            return self.data[self.map_folder['test']]
        else:
            raise ValueError('phase error, phase in [train, val, test]')
        
    def get_attribute(self):
        return self.attribute_name
    
    def get_weight(self, phase = 'train'):
        if phase == 'train':
            return self.weight_train
        raise ValueError('phase error, phase in [train]')
    
    def get_list_phase(self):
        return self.list_phases

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

if __name__ == "__main__":
    datasource = PPE(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    pass