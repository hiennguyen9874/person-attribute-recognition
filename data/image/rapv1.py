import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import pickle
import scipy.io
import numpy as np
from shutil import copy2

from collections import defaultdict

from base import BaseDataSource

class RAPV1(BaseDataSource):
    r""" RAP dataset, http://www.rapdataset.com/
    """
    url = {
        'RAP_v1_dataset.zip': '1kJ93jBF8Tc2jI6sLzuABhdxE0qVRM4Rw',
        'RAP_v1_annotation.zip': '1zuZTSjAm4hCpFvp2mss7vLK4MCmijR2q'
    }
    file_path = {
        'RAP_v1_dataset.zip': '/content/drive/My Drive/Colab/Datasets/RAP_v1_dataset.zip',
        'RAP_v1_annotation.zip': '/content/drive/My Drive/Colab/Datasets/RAP_v1_annotation.zip',
    }
    group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]

    def __init__(
        self, 
        root_dir='datasets', 
        download=True, 
        extract=True, 
        use_tqdm=True, 
        **kwargs):

        super(RAPV1, self).__init__(
            root_dir, 
            dataset_dir = 'rapv1', 
            image_size = (256, 192))
        
        if download:
            for key, value in self.url.items():
                try:
                    self._download(file_name=key, file_path=self.file_path[key], use_tqdm=use_tqdm)
                except:
                    self._download(file_name=key, dataset_id=value, use_tqdm=use_tqdm)
        if extract: 
            for key, value in self.url.items():
                self._extract(file_name=key, use_tqdm=use_tqdm, pwd='casia_cripac_isee_5610')
        
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
    datasource = RAPV1(root_dir='/home/hien/Documents/datasets', download=True, extract=True, use_tqdm=True)
    datasource.summary_weight()
    datasource.summary_count()
