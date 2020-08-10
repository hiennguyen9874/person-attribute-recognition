import os
import scipy.io
import numpy as np

import sys
sys.path.append('.')

from collections import defaultdict

from base import BaseDataSource

class Peta(BaseDataSource):
    r''' 
        http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
        https://github.com/dangweili/pedestrian-attribute-recognition-pytorch
    '''
    dataset_id = '1Z2o5RyyCXBBGdEUey-Wi1ImFDFUTIVEo'
    group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]

    def __init__(self, root_dir='datasets', download=True, extract=True, use_tqdm=True, validation_split=0.1):
        super(Peta, self).__init__(
            root_dir, 
            dataset_dir = 'peta', 
            file_name = 'PETA-New.zip', 
            image_size = (256, 192))
        if download:
            self._download(dataset_id=self.dataset_id, use_tqdm=use_tqdm)
        if extract:
            self._extract(use_tqdm=use_tqdm)
        
        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        
        f = scipy.io.loadmat(os.path.join(data_dir, 'PETA.mat'))
        
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

            self.data['train'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _train])
            self.data['val'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _val])
            # self.data['trainval'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _trainval])
            self.data['test'].append([(os.path.join(data_dir, 'images', '%05d.png'%(idx+1)), label[idx]) for idx in _test])

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

if __name__ == "__main__":
    from utils import read_config
    config = read_config('config/base_epoch.yml')
    datasource = Peta(root_dir=config['data']['data_dir'], download=True, extract=True)
    print(len(datasource.get_attribute()))
    print(np.expand_dims(datasource.get_weight('test'), axis=1))
    pass

r'''
['accessoryHat',
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