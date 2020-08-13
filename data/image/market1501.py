import sys
sys.path.append('.')

import os
import requests
import tarfile
import zipfile
import re
import glob
import gdown
import scipy.io
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils import download_file_from_google_drive, download_with_url

class Market1501_Attribute(object):
    dataset_dir = 'market1501_attribute'
    dataset_id = '13GJogmW1neVqSOHxZ_EUo9KfoWuwsAQM'
    file_name = 'Market-1501-v15.09.15-Attribute.zip'
    google_drive_api = 'AIzaSyAVfS-7Dy34a3WjWgR509o-u_3Of59zizo'

    def __init__(self, root_dir='datasets', download=True, extract=True, re_label_on_train=True):
        self.root_dir = root_dir
        if download:
            print("Downloading!")
            self._download()
            print("Downloaded!")
        if extract:
            print("Extracting!")
            self._extract()
            print("Extracted!")

        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed', 'Market-1501-v15.09.15-Attribute')

        train_dir = os.path.join(data_dir, 'bounding_box_train')
        query_dir = os.path.join(data_dir, 'query')
        gallery_dir = os.path.join(data_dir, 'bounding_box_test')

        self.pid_container = dict()
        self.camid_containter = dict()
        self.frames_container = dict()
        pid2label = dict()

        print("Processing on train directory!")
        self.train, self.pid_container['train'], self.camid_containter['train'], self.frames_container['train'], pid2label['train'] = self._process_dir(
            train_dir, relabel=re_label_on_train)

        print("Processing on query directory!")
        self.query, self.pid_container['query'], self.camid_containter['query'], self.frames_container['query'], pid2label['query'] = self._process_dir(
            query_dir, relabel=False)

        print("Processing on gallery directory!")
        self.gallery, self.pid_container['gallery'], self.camid_containter['gallery'], self.frames_container['gallery'], pid2label['gallery'] = self._process_dir(
            gallery_dir, relabel=False)

        f = scipy.io.loadmat(os.path.join(data_dir, 'attribute', 'market_attribute.mat'))
        
        # print("Get attribute...")
        self.dict_attribute = dict()
        self.dict_attribute_label = dict()
        self.dict_attribute['train'], self.dict_attribute_label['train'] = self._get_dict_attribute(f, 'train', relabel=re_label_on_train, pid2label=pid2label['train'])
        self.dict_attribute['test'], self.dict_attribute_label['test'] = self._get_dict_attribute(f, 'test', relabel=False)
        
    def get_data(self, mode='train'):
        if mode == 'train':
            return self.train
        elif mode == 'query':
            return self.query
        elif mode == 'gallery':
            return self.gallery
        else:
            raise ValueError('mode error')
    
    def get_attribute(self, mode='train'):
        if mode == 'train':
            return self.dict_attribute['train'], self.dict_attribute_label['train']
        elif mode == 'query':
            return self.dict_attribute['test'], self.dict_attribute_label['test']
        elif mode == 'gallery':
            return self.dict_attribute['test'], self.dict_attribute_label['test']
        else:
            raise ValueError('mode error')

    def _process_dir(self, path, relabel):
        data = []
        pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_([-\d]+)')

        with tqdm(total=len(os.listdir(path)*2)) as pbar:
            pid_container = set()
            camid_containter = set()
            frames_container = set()

            for img in os.listdir(path):
                name, ext = os.path.splitext(img)
                if ext == '.jpg':
                    img_path = os.path.join(path, img)
                    person_id, camera_id, seq, frame = map(
                        int, pattern.search(name).groups())
                    # if person_id == -1:
                    #     pbar.update(1)
                    #     continue
                    pid_container.add(person_id)
                    camid_containter.add(camera_id)
                    frames_container.add(self._re_frame(camera_id, seq, frame))
                pbar.update(1)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img in os.listdir(path):
                name, ext = os.path.splitext(img)
                if ext == '.jpg':
                    img_path = os.path.join(path, img)
                    person_id, camera_id, seq, frame = map(
                        int, pattern.search(name).groups())
                    # if person_id == -1:
                    #     pbar.update(1)
                    #     continue
                    if relabel:
                        person_id = pid2label[person_id]
                    data.append((img_path, person_id, camera_id))
                pbar.update(1)
        return data, pid_container, camid_containter, frames_container, pid2label

    def _download(self):
        os.makedirs(os.path.join(self.root_dir,
                                 self.dataset_dir, 'raw'), exist_ok=True)
        download_with_url(self.google_drive_api, self.dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), self.file_name)
        

    def _extract(self):
        file_path = os.path.join(
            self.root_dir, self.dataset_dir, 'raw', self.file_name)
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
        if os.path.exists(os.path.join(extract_dir, 'Market-1501-v15.09.15-Attribute', 'bounding_box_train')) \
            and os.path.exists(os.path.join(extract_dir, 'Market-1501-v15.09.15-Attribute', 'bounding_box_test')) \
            and os.path.exists(os.path.join(extract_dir, 'Market-1501-v15.09.15-Attribute', 'query')):
            return True
        return False

    def get_num_classes(self, dataset: str):
        if dataset not in ['train', 'query', 'gallery']:
            raise ValueError(
                "Error dataset paramaster, dataset in [train, query, gallery]")
        return len(self.pid_container[dataset])

    def get_num_camera(self, dataset: str):
        if dataset not in ['train', 'query', 'gallery']:
            raise ValueError(
                "Error dataset paramaster, dataset in [train, query, gallery]")
        return len(self.camid_containter[dataset])

    def _re_frame(self, cam, seq, frame):
        """ Re frames on market1501.
            more info here: https://github.com/Wanggcong/Spatial-Temporal-Re-identification/issues/10
        """
        if seq == 1:
            return frame
        dict_cam_seq_max = {
            11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
            21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
            31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
            41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
            51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
            61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}
        
        re_frame = 0
        for i in range(1, seq):
            re_frame += dict_cam_seq_max[int(str(cam) + str(i))]
        return re_frame + frame

    def get_name_dataset(self):
        return self.file_name.split('.zip')[0]
    
    def _get_dict_attribute(self, f, test_train, relabel = True, pid2label=None):
        train_label=[
            'age',
            'backpack',
            'bag',
            'handbag',
            'downblack',
            'downblue',
            'downbrown',
            'downgray',
            'downgreen',
            'downpink',
            'downpurple',
            'downwhite',
            'downyellow',
            'upblack',
            'upblue',
            'upgreen',
            'upgray',
            'uppurple',
            'upred',
            'upwhite',
            'upyellow',
            'clothes',
            'down',
            'up',
            'hair',
            'hat',
            'gender']
    
        test_label=[
            'age',
            'backpack',
            'bag',
            'handbag',
            'clothes',
            'down',
            'up',
            'hair',
            'hat',
            'gender',
            'upblack',
            'upwhite',
            'upred',
            'uppurple',
            'upyellow',
            'upgray',
            'upblue',
            'upgreen',
            'downblack',
            'downwhite',
            'downpink',
            'downpurple',
            'downyellow',
            'downgray',
            'downblue',
            'downgreen',
            'downbrown']
            
        attribute_dict = defaultdict(list)

        pid_container_sorted = sorted(self.pid_container[test_train if test_train == 'train' else 'query'])

        test_train = 0 if test_train == 'test' else (1 if test_train == 'train' else None)

        for attribute_id in range(len(f['market_attribute'][0][0][test_train][0][0])):
            if isinstance(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):
                continue
            for person_id in range(len(f['market_attribute'][0][0][test_train][0][0][attribute_id][0])):
                # print(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])
                attribute_dict[pid_container_sorted[person_id]].append(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][person_id] - 1)
        
        unified_attribute = {}
        for k, v in attribute_dict.items():
            temp_atr = [0]*len(test_label)
            for i in range(len(test_label)):
                temp_atr[i]=v[train_label.index(test_label[i])]
            pid = k
            if relabel:
                pid = pid2label[k]
            unified_attribute[pid] = temp_atr
        
        for id in unified_attribute:
            if unified_attribute[id][0] == 0:
                unified_attribute[id].pop(0)
                unified_attribute[id].insert(0, 1)
                unified_attribute[id].insert(1, 0)
                unified_attribute[id].insert(2, 0)
                unified_attribute[id].insert(3, 0)
            elif unified_attribute[id][0] == 1:
                unified_attribute[id].pop(0)
                unified_attribute[id].insert(0, 0)
                unified_attribute[id].insert(1, 1)
                unified_attribute[id].insert(2, 0)
                unified_attribute[id].insert(3, 0)
            elif unified_attribute[id][0] == 2:
                unified_attribute[id].pop(0)
                unified_attribute[id].insert(0, 0)
                unified_attribute[id].insert(1, 0)
                unified_attribute[id].insert(2, 1)
                unified_attribute[id].insert(3, 0)
            elif unified_attribute[id][0] == 3:
                unified_attribute[id].pop(0)
                unified_attribute[id].insert(0, 0)
                unified_attribute[id].insert(1, 0)
                unified_attribute[id].insert(2, 0)
                unified_attribute[id].insert(3, 1)

        test_label.pop(0)
        test_label.insert(0,'young')
        test_label.insert(1,'teenager')
        test_label.insert(2,'adult')
        test_label.insert(3,'old')
        return unified_attribute, test_label

if __name__ == "__main__":
    market1501 = Market1501_Attribute(root_dir='/home/hien/Documents/datasets', download=True, extract=True, re_label_on_train=True)
    print('Train: len: {}, num_class: {}, num_camera: {}'.format(len(market1501.train), market1501.get_num_classes('train'), market1501.get_num_camera('train')))
    print('Query: len: {}, num_class: {}, num_camera: {}'.format(len(market1501.query), market1501.get_num_classes('query'), market1501.get_num_camera('query')))
    print('Gallery: len: {}, num_class: {}, num_camera: {}'.format(len(market1501.gallery), market1501.get_num_classes('gallery'), market1501.get_num_camera('gallery')))