import os
import shutil
import argparse

from tqdm  import tqdm

from utils import rmdir

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

if __name__ == "__main__":
    log_dir = '/content/drive/Shared drives/REID/HIEN/Models/person_attribute_recognition/logs'
    saved_folder = '/content/drive/Shared drives/REID/HIEN/Models/person_attribute_recognition/saved'

    all_saved_info = list()
    for f1 in [f.name for f in os.scandir(log_dir) if f.is_dir()]:
        if not os.path.exists(os.path.join(log_dir, f1, 'info.log')):
            continue
        info_log = open(os.path.join(log_dir, f1, 'info.log'), 'r')
        if os.path.exists(os.path.join(log_dir, f1, 'info.log.gdoc')):
            os.remove(os.path.join(log_dir, f1, 'info.log.gdoc'))
        saved_info = dict()
        saved_info['id'] = f1
        saved_info['path'] = os.path.join(log_dir, f1)
        saved_info['info'] = dict()
        saved = False
        for i, line in enumerate(info_log):
            head = line[41:].split(':')
            if head[0] not in ['Run id', 'Dataset', 'Model', 'Freeze layer', 'Loss', 'Optimizer', 'Lr scheduler']:
                break
            if head[0] == 'Dataset':
                saved_info['info']['Dataset'] = head[1].split(',')[0].replace(' ', '')
            if head[0] == 'Model':
                saved_info['info']['Model'] = head[2].split(' ')[1].replace(' ', '')
            if head[0] == 'Freeze layer':
                saved_info['info']['Freeze layer'] = 1
            if head[0] == 'Loss':
                saved_info['info']['Loss'] = head[1].split(' ')[1].replace(' ', '')
            if head[0] == 'Optimizer':
                saved_info['info']['Optimizer'] = head[1].split(' ')[1].replace(' ', '')
            if head[0] == 'Lr scheduler':
                saved_info['info']['Lr scheduler'] = head[1].split(' ')[1].replace(' ', '')
            saved = True
        info_log.close()
        if saved == True:
            if 'Freeze layer' not in saved_info['info']:
                saved_info['info']['Freeze layer'] = 0
            all_saved_info.append(saved_info)
    
    # rmdir(saved_folder)
    for x in tqdm(all_saved_info):
        source = x['path']
        des = os.path.join(saved_folder, x['info']['Dataset'], x['info']['Model'],  x['info']['Loss'], 'Freeze layer'.replace(' ', '') + '-' + str(x['info']['Freeze layer']), x['info']['Optimizer'], x['info']['Lr scheduler'], x['id'])
        if not os.path.exists(des):
            os.makedirs(des, exist_ok=True)
            copytree(source, des)

