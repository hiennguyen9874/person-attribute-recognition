import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import zipfile
import tarfile

from tqdm import tqdm
from shutil import copyfile

from utils import download_file_from_google_drive, download_with_url

class BaseDataSource(object):    
    def __init__(
        self,
        root_dir,
        dataset_dir,
        phase=['train', 'val', 'test'],
        **kwargs):
        
        self.phase = phase
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        
    def _exists(self, extract_dir):
        raise NotImplementedError
    
    def _extract(self, file_name, use_tqdm=True):
        r""" extract compressed file
        Args:
            use_tqdm (boolean): use tqdm process bar when extracting
        """
        file_path = os.path.join(self.root_dir, self.dataset_dir, 'raw', file_name)
        extract_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed', ''.join(file_name.split('.')[:-1]))
        if self._exists(extract_dir):
            return
        print("Extracting...")
        try:
            tar = tarfile.open(file_path)
            os.makedirs(extract_dir, exist_ok=True)
            if use_tqdm:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc=file_name):
                    tar.extract(member=member, path=extract_dir)
            else:
                tar.extractall(path=extract_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(file_path, 'r')
            if use_tqdm:
                for member in tqdm(iterable=zip_ref.infolist(), total=len(zip_ref.infolist()), desc=file_name):
                    zip_ref.extract(member=member, path=extract_dir)
            else:
                zip_ref.extractall(path=extract_dir)
            zip_ref.close()
        print("Extracted!")
    
    def _download(self, file_name, url=None, dataset_id=None, file_path=None, use_tqdm=True):
        r""" download file from google drive.
        Args:
            dataset_id (str): id of file on google drive. guide to get it (https://www.wonderplugin.com/wordpress-tutorials/how-to-apply-for-a-google-drive-api-key/)
            use_tqdm (boolean): use tqdm process bar when downloading
        """
        os.makedirs(os.path.join(self.root_dir, self.dataset_dir, 'raw'), exist_ok=True)
        if dataset_id != None:
            print("Downloading...")
            try:
                try:
                    download_file_from_google_drive(dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), use_tqdm)
                except:
                    url = "https://www.googleapis.com/drive/v3/files/" + dataset_id + "?alt=media&key=AIzaSyBEp1hj-WxRxAezSd5sGfPmWnLbuxuxSvI"
                    download_with_url(url, os.path.join(self.root_dir, self.dataset_dir, 'raw'), file_name, use_tqdm)
            except:
                try:
                    if os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'raw', file_name)):
                        os.remove(os.path.join(self.root_dir, self.dataset_dir, 'raw', file_name))
                    download_file_from_google_drive(dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), use_tqdm)
                except:
                    if os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'raw', file_name)):
                        os.remove(os.path.join(self.root_dir, self.dataset_dir, 'raw', file_name))
                    url = "https://www.googleapis.com/drive/v3/files/" + dataset_id + "?alt=media&key=AIzaSyBEp1hj-WxRxAezSd5sGfPmWnLbuxuxSvI"
                    download_with_url(url, os.path.join(self.root_dir, self.dataset_dir, 'raw'), file_name, use_tqdm)
            print("Downloaded!")
        elif url != None:
            download_with_url(url, os.path.join(self.root_dir, self.dataset_dir, 'raw'), file_name, use_tqdm)
        elif file_path != None:
            print('Copying data...')
            copyfile(file_path, os.path.join(self.root_dir, self.dataset_dir, 'raw', file_name))
            print("Copied!")
        else:
            if not os.path.exists(os.path.join(self.root_dir, self.dataset_dir, 'raw', file_name)):
                raise FileExistsError('please download file %s into %s' % (file_name, os.path.join(self.root_dir, self.dataset_dir, 'raw')))

    def get_data(self, phase='train'):
        r""" get data, must return list of (image_path, label)
        """
        raise NotImplementedError
    
    def get_phase(self):
        r""" get list of phase.
        """
        return self.phase
    
    def _check_file_exits(self):
        r""" check all image in datasource exists
        """
        for phase in self.phase:
            for path, label in self.get_data(phase):
                if not os.path.exists(path):
                    raise FileExistsError
