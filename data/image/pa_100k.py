
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import pickle
import numpy as np

import scipy.io

from base import BaseDataSource

class PA_100K(BaseDataSource):
    r''' https://github.com/xh-liu/HydraPlus-Net/blob/master/README.md,
    '''
    url = {
        'PA-100K.zip': '1Jfb3I8BK4oOX3eepaiYyd4fHoMNkgTaz'
    }
    file_path = {
        'PA-100K.zip': '/content/drive/Shared drives/HIEN/Datasets/PA-100K.zip',
    }
    group_order = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]

    def __init__(
        self,
        root_dir='datasets',
        download=True,
        extract=True,
        use_tqdm=True):

        super(PA_100K, self).__init__(
            root_dir,
            dataset_dir = 'pa_100k',
            image_size=(256, 128))
        
        if download:
            for key, value in self.url.items():
                try:
                    self._download(file_name=key, file_path=self.file_path[key], use_tqdm=use_tqdm)
                except:
                    self._download(file_name=key, dataset_id=value, use_tqdm=use_tqdm)
        if extract: 
            for key, value in self.url.items():
                self._extract(file_name=key, use_tqdm=use_tqdm)
        
        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        if os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'processed', 'PA-100K')):
            data_dir = os.path.join(data_dir, 'PA-100K')

        f = scipy.io.loadmat(os.path.join(data_dir, 'annotation.mat'))
        image_name = dict()
        label = dict()
        
        image_name['train'] = [os.path.join(data_dir, 'images', f['train_images_name'][i][0][0]) for i in range(80000)]
        label['train'] = f['train_label'][:, np.array(self.group_order)].astype(np.float32)
        
        image_name['val'] = [os.path.join(data_dir, 'images',  f['val_images_name'][i][0][0]) for i in range(10000)]
        label['val'] = f['val_label'][:, np.array(self.group_order)].astype(np.float32)
        
        image_name['test'] = [os.path.join(data_dir, 'images', f['test_images_name'][i][0][0]) for i in range(10000)]
        label['test'] = f['test_label'][:, np.array(self.group_order)].astype(np.float32)

        self.attribute_name = [f['attributes'][i][0][0] for i in range(26)]
        self.data = dict()
        self.weight = dict()
        for phase in ['train', 'val', 'test']:
            self.data[phase] = list(zip(image_name[phase], label[phase]))
            self.weight[phase] = np.mean(label[phase], axis=0).astype(np.float32)
        
        self._check_file_exits()

    def get_data(self, phase='train'):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        return self.data[phase]
    
    def get_weight(self, phase = 'train'):
        assert phase in ['train', 'val', 'test'], 'phase must in [train, val, test]'
        return self.weight[phase]
    
    def get_attribute(self):
        return self.attribute_name

    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'images')) \
                and os.path.exists(os.path.join(extract_dir, 'README_0.txt')) \
                and os.path.exists(os.path.join(extract_dir, 'README_1.txt')) \
                and os.path.exists(os.path.join(extract_dir, 'annotation.mat')):
            return True
        return False

    def get_list_attribute_random(self):
        import itertools
        arr = list()
        arr.append([[0], [1]]) #1
        arr.append([[0, 0, 1], [0, 1, 0], [1,1, 0]]) #2
        arr.append([[0, 0, 1], [0, 1, 0], [1,1, 0]]) #3
        for _ in range(6): # 4-> 9
            arr.append([[0], [1]])
        arr.append([[0, 1], [1, 0]]) # 10
        for _ in range(7): # 11->17
            arr.append([[0], [1]])
        arr.append([[0, 0, 1], [0, 1, 0], [1, 1, 0]]) #18
        arr.append([[0], [1]]) # 19
        return [list(itertools.chain(*ele)) for ele in itertools.product(*arr)]

    def save_attribute(self, path='attribute.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.get_attribute(), f)

    def summary_count(self):
        print('num image in training set: ', len(self.get_data('train')))
        print('num image in valid set: ', len(self.get_data('val')))
        print('num image in test set: ', len(self.get_data('test')))
    
    def summary_weight(self):
        row_format = "{:>5}" + "{:>20}" + "{:>10}"*3
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
    datasource = PA_100K(root_dir='/datasets', download=True, extract=True)
    datasource.summary_weight()
    # print(np.expand_dims(datasource.get_weight('train'), axis=1))
    # print(np.around(np.stack((datasource.get_weight('train'), datasource.get_weight('test')), axis=1)*100, 2))

    # datasource.save_attribute('pa100k_attribute.pkl')
    pass

'''
['Female']:1

['AgeOver60']: 2
['Age18-60']: 2
['AgeLess18']: 2

['Front']: 3
['Side']: 3
['Back']: 3

['Hat']: 4
['Glasses']: 5
['HandBag']: 6
['ShoulderBag']: 7
['Backpack']: 8
['HoldObjectsInFront']: 9

['ShortSleeve']: 10
['LongSleeve']: 10

['UpperStride']: 11
['UpperLogo']: 12
['UpperPlaid']: 13
['UpperSplice']: 14

['LowerStripe']: 15
['LowerPattern']: 16

['LongCoat']: 17

['Trousers']: 18
['Shorts']: 18
['Skirt&Dress']: 18

['boots']: 19
'''