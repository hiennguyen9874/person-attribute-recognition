
import numpy as np
import scipy.io
import os
import sys
sys.path.append('.')

from base import BaseDataSource

class PA_100K(BaseDataSource):
    ''' https://github.com/xh-liu/HydraPlus-Net/blob/master/README.md
    '''
    dataset_id = '13UjvKJQlkNXAmvsPG6h5dwOlhJQA_TcT'
    group_order = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]

    def __init__(self, root_dir='datasets', download=True, extract=True):
        file_name = 'PA-100K.zip'
        dataset_dir = 'pa_100k'
        super(PA_100K, self).__init__(root_dir, dataset_dir, file_name, image_size=(256, 128))
        if download:
            print("Downloading!")
            self._download(self.dataset_id)
            print("Downloaded!")
        if extract:
            print("Extracting!")
            self._extract()
            print("Extracted!")
            
        f = scipy.io.loadmat(os.path.join(self.data_dir, 'annotation.mat'))
        image_name = dict()
        label = dict()
        
        image_name['train'] = [os.path.join(self.data_dir, 'images', f['train_images_name'][i][0][0]) for i in range(80000)]
        label['train'] = f['train_label'][:, np.array(self.group_order)].astype(np.float32)
        
        image_name['val'] = [os.path.join(self.data_dir, 'images',  f['val_images_name'][i][0][0]) for i in range(10000)]
        label['val'] = f['val_label'][:, np.array(self.group_order)].astype(np.float32)
        
        image_name['test'] = [os.path.join(self.data_dir, 'images', f['test_images_name'][i][0][0]) for i in range(10000)]
        label['test'] = f['test_label'][:, np.array(self.group_order)].astype(np.float32)

        self.attr_name = [f['attributes'][i][0][0] for i in range(26)]
        self.train = list(zip(image_name['train'], label['train']))
        self.val = list(zip(image_name['val'], label['val']))
        self.test = list(zip(image_name['test'], label['test']))
        self.weight_train = np.mean(label['train'], axis=0).astype(np.float32)

    def get_data(self, phase='train'):
        if phase == 'train':
            return self.train
        elif phase == 'val':
            return self.val
        elif phase == 'test':
            return self.test
        else:
            raise ValueError('phase error, phase in [train, val, test]')
        
    def get_attribute(self, mode = 'train'):
        return self.attr_name
    
    def get_weight(self, phase = 'train'):
        if phase == 'train':
            return self.weight_train
        raise ValueError('phase error, phase in [train]')

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

if __name__ == "__main__":
    datasource = PA_100K(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    all_data = datasource.get_data('train')[0] + datasource.get_data('val')[0] + datasource.get_data('test')[0]
    datasource.get_list_attribute_random()
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