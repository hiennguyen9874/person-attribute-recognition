import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

import pickle
import scipy.io
import numpy as np
from shutil import copy2

from collections import defaultdict

from base import BaseDataSource

class Peta(BaseDataSource):
    r""" Peta Dataset
        http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
        https://github.com/dangweili/pedestrian-attribute-recognition-pytorch
        https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition/blob/master/dataset/preprocess/format_peta.py
    """
    url = {
        'PETA-New.zip': '1Z2o5RyyCXBBGdEUey-Wi1ImFDFUTIVEo'
    }
    file_path = {
        'PETA-New.zip': '/content/drive/Shared drives/REID/HIEN/Datasets/PETA-New.zip',
    }
    group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]

    def __init__(self, root_dir='datasets', download=True, extract=True, use_tqdm=True, **kwargs):
        super(Peta, self).__init__(
            root_dir, 
            dataset_dir = 'peta', 
            image_size = (256, 192))
        if download:
            for key, value in self.url.items():
                try:
                    self._download(file_name=key, file_path=self.file_path[key], use_tqdm=use_tqdm)
                except:
                    self._download(file_name=key, dataset_id=value, use_tqdm=use_tqdm)
        if extract: 
            for key, value in self.url.items():
                self._extract(file_name=key, use_tqdm=use_tqdm)
        
        self.data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        if os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'processed', 'PETA-New')):
            self.data_dir = os.path.join(self.data_dir, 'PETA-New')
        
        f = scipy.io.loadmat(os.path.join(self.data_dir, 'PETA.mat'))
        
        raw_attr_name = [i[0][0] for i in f['peta'][0][0][1]]
        raw_label = f['peta'][0][0][0][:, 4:]
        
        label = raw_label[:, :35][:, np.array(self.group_order)].astype(np.float32)
        self.attribute_name = [raw_attr_name[:35][i] for i in self.group_order]

        self.data = defaultdict(list)
        self.weight = defaultdict(list)

        for idx in range(5):
            _train = f['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
            _val = f['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
            _test = f['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
            # _trainval = np.concatenate((_train, _val), axis=0)

            self.data['train'].append([(os.path.join(self.data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _train])
            self.data['val'].append([(os.path.join(self.data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _val])
            # self.data['trainval'].append([(os.path.join(self.data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _trainval])
            self.data['test'].append([(os.path.join(self.data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _test])

            self.weight['train'].append(np.mean(label[_train], axis=0))
            self.weight['val'].append(np.mean(label[_val], axis=0))
            self.weight['test'].append(np.mean(label[_test], axis=0))
            # self.weight_trainval.append(np.mean(label[_trainval], axis=0))
        self._check_file_exits()

    def get_data(self, phase='train', partition=0):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        assert partition < 5, 'partition must in [0-5]'
        return self.data[phase][partition]

    def get_weight(self, phase = 'train', partition=0):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        assert partition < 5, 'partition must in [0-5]'
        return self.weight[phase][partition]

    def get_attribute(self):
        return self.attribute_name

    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'images')) \
            and os.path.exists(os.path.join(extract_dir, 'README')) \
            and os.path.exists(os.path.join(extract_dir, 'PETA.mat')):
            return True
        return False

    def save_attribute(self, path='attribute.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.get_attribute(), f)
    
    def parser_folder(self, phase='train'):
        from tqdm.auto import tqdm
        des_dir = os.path.join(self.data_dir, phase)
        os.makedirs(des_dir, exist_ok=True)
        os.makedirs(os.path.join(des_dir, 'images'), exist_ok=True)
        copy2(os.path.join(self.data_dir, 'PETA.mat'), des_dir)
        for file_path, labels in tqdm(self.get_data(phase)):
            copy2(file_path, os.path.join(des_dir, 'images'))
    
    def pase_data(self, phase):
        pos_dict = defaultdict(list)
        neg_dict = defaultdict(list)

        for idx, (_, label) in enumerate(self.get_data(phase)):
            for i, attribute in enumerate(self.get_attribute()):
                if label[i] == 1.0:
                    pos_dict[attribute].append(idx)
                else:
                    neg_dict[attribute].append(idx)
        return pos_dict, neg_dict

    
    def summary_count(self):
        print('num image in training set: ', len(self.get_data('train')))
        print('num image in valid set: ', len(self.get_data('val')))
        print('num image in test set: ', len(self.get_data('test')))
    
    def summary_weight(self):
        row_format = "{:>5}" + "{:>25}" + "{:>10}"*3
        print(row_format.format('-', 'attribute', 'train', 'val', 'test'))
        print(row_format.format('-', '-', '-', '-', '-'))
        for idx in range(len(self.get_attribute())):
            print(row_format.format(
                idx+1,
                self.get_attribute()[idx], 
                round(self.get_weight('train')[idx]*100, 2),
                round(self.get_weight('val')[idx]*100, 2),
                round(self.get_weight('test')[idx]*100,2)))

if __name__ == "__main__":
    from utils import read_config
    config = read_config('config/base_epoch.yml', False)
    datasource = Peta(root_dir=config['data']['data_dir'], download=False, extract=True)

    datasource.summary_weight()
    
    # show some image by attribute
    # import cv2
    # import matplotlib.pyplot as plt
    
    # from shutil import copy2
    # from utils import imread
    # from tqdm.auto import tqdm

    # for attribute in tqdm(datasource.get_attribute()):
    #     pos_dict, neg_dict = datasource.pase_data(phase='train')
        
    #     list_idx_by_attribute = pos_dict[attribute]
    #     for idx in list_idx_by_attribute:
    #         path, _ = datasource.get_data('train')[idx]
    #         os.makedirs(os.path.join(datasource.data_dir, 'positive', attribute), exist_ok=True)
    #         copy2(path, os.path.join(datasource.data_dir, 'positive', attribute))
        
    #     list_idx_by_attribute = neg_dict[attribute]
    #     for idx in list_idx_by_attribute:
    #         path, _ = datasource.get_data('train')[idx]
    #         os.makedirs(os.path.join(datasource.data_dir, 'neg', attribute), exist_ok=True)
    #         copy2(path, os.path.join(datasource.data_dir, 'neg', attribute))

    # datasource.summary()
    # datasource.parser_folder('train')
    # print(len(datasource.get_attribute()))
    # print(np.around(np.stack((datasource.get_weight('train'), datasource.get_weight('test')), axis=1)*100, 2))
    # datasource.save_attribute('peta_attribute.pkl')
    pass

r'''
'accessoryHat',
'accessoryMuffler',
'accessoryNothing',
'accessorySunglasses',
'hairLong',
'upperBodyCasual',
'upperBodyFormal',
'upperBodyJacket',
'upperBodyLogo',
'upperBodyPlaid',
'upperBodyShortSleeve',
'upperBodyThinStripes',
'upperBodyTshirt',
'upperBodyOther',
'upperBodyVNeck',
'lowerBodyCasual',
'lowerBodyFormal',
'lowerBodyJeans',
'lowerBodyShorts',
'lowerBodyShortSkirt',
'lowerBodyTrousers',
'footwearLeatherShoes',
'footwearSandals',
'footwearShoes',
'footwearSneaker',
'carryingBackpack',
'carryingOther',
'carryingMessengerBag',
'carryingNothing',
'carryingPlasticBags',
'personalLess30',
'personalLess45',
'personalLess60',
'personalLarger60',
'personalMale']
'''

