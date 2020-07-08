import os
import zipfile
import tarfile

import sys
sys.path.append('.')

from utils import download_with_url

class BaseDataSource(object):
    google_drive_api = 'AIzaSyAVfS-7Dy34a3WjWgR509o-u_3Of59zizo'
    list_phases = ['train', 'val', 'test']
    
    def __init__(self, root_dir, dataset_dir, file_name, image_size = (256, 128), list_phases=['train', 'val', 'test']):
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        self.file_name = file_name
        self.list_phases = list_phases
        self.data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        self.image_size = image_size
        
    def _exists(self, extract_dir):
        raise NotImplementedError
    
    def _extract(self):
        file_path = os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)
        extract_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        if self._exists(extract_dir):
            return
        try:
            tar = tarfile.open(file_path)
            os.makedirs(extract_dir, exist_ok=True)
            tar.extractall(path=extract_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(path=extract_dir)
            zip_ref.close()
    
    def _download(self, dataset_id=None):
        os.makedirs(os.path.join(self.root_dir, self.dataset_dir, 'raw'), exist_ok=True)
        if dataset_id == None:
            if not os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)):
                raise FileExistsError('please download file into %s' % (os.path.join(self.root_dir, self.dataset_dir, 'raw')))
        else:
            download_with_url(self.google_drive_api, dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), self.file_name)

    def get_list_phase(self):
        return self.list_phases
    
    def get_data(self, phase='train'):
        raise NotImplementedError

    def get_image_size(self):
        return self.image_size