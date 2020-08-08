import os
import zipfile
import tarfile

import sys
sys.path.append('.')
from tqdm import tqdm

from utils import download_file_from_google_drive, download_with_url

class BaseDataSource(object):
    google_drive_api = 'AIzaSyBEp1hj-WxRxAezSd5sGfPmWnLbuxuxSvI'
    
    def __init__(
        self,
        root_dir,
        dataset_dir,
        file_name,
        image_size=(256, 128),
        phase=['train', 'val', 'test']):
        
        self.phase = phase
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        self.file_name = file_name
        self.data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        self.image_size = image_size
        
    def _exists(self, extract_dir):
        raise NotImplementedError
    
    def _extract(self, use_tqdm=True):
        r""" extract compressed file
        Args:
            use_tqdm (boolean): use tqdm process bar when extracting
        """
        file_path = os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)
        extract_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
        if self._exists(extract_dir):
            return
        print("Extracting...")
        try:
            tar = tarfile.open(file_path)
            os.makedirs(extract_dir, exist_ok=True)
            if use_tqdm:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member, path=extract_dir)
            else:
                tar.extractall(path=extract_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(file_path, 'r')
            if use_tqdm:
                for member in tqdm(iterable=zip_ref.infolist(), total=len(zip_ref.infolist())):
                    zip_ref.extract(member=member, path=extract_dir)
            else:
                zip_ref.extractall(path=extract_dir)
            zip_ref.close()
        print("Extracted!")
    
    def _download(self, dataset_id=None, use_tqdm=True):
        r""" download file from google drive.
        Args:
            dataset_id (str): id of file on google drive. guide to get it (https://www.wonderplugin.com/wordpress-tutorials/how-to-apply-for-a-google-drive-api-key/)
            use_tqdm (boolean): use tqdm process bar when downloading
        """
        os.makedirs(os.path.join(self.root_dir, self.dataset_dir, 'raw'), exist_ok=True)
        if dataset_id == None:
            if not os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)):
                raise FileExistsError('please download file %s into %s' % (self.file_name, os.path.join(self.root_dir, self.dataset_dir, 'raw')))
        else:
            print("Downloading...")
            download_with_url(self.google_drive_api, dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), self.file_name, use_tqdm)
            # download_file_from_google_drive(dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), use_tqdm)
            print("Downloaded!")
    
    def get_data(self, phase='train'):
        r""" get data, must return list of (image_path, label)
        """
        raise NotImplementedError

    def get_image_size(self):
        r""" get size of image to resize when training
        """
        return self.image_size
    
    def get_phase(self):
        return self.phase
    
    def _check_file_exits(self):
        for phase in self.phase:
            for path, label in self.get_data(phase):
                if not os.path.exists(path):
                    raise FileExistsError
                
