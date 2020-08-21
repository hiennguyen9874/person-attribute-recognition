import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import torch
import pickle
import argparse

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import build_model
from utils import read_config, imread

def extractor(path_config, image, return_type=0):
    r"""
    Args:
        path_config (str): path to config image
        image (numpy image):
        return_type: type of return
            0: return list of binary
            1: return dict
            2: return list of attribute
    """
    config = read_config(path_config, False)

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    attribute_name = pickle.load(open(config['data']['path_attribute'], 'rb'))
    
    model, _ = build_model(config, num_classes=len(attribute_name))
    checkpoint = torch.load(config['resume'], map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    tranform_extract = A.Compose([
        A.Resize(config['data']['size'][0], config['data']['size'][1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    image = tranform_extract(image=image)
    image = torch.unsqueeze(image, 0)

    out = model(image)
    out = torch.squeeze(out)
    out = torch.sigmoid(out)

    out = out.cpu().detach().numpy()

    out[out>0.5]=1
    out[out<=0.5]=0
    out = out.astype(int)
    
    if return_type == 0:        
        out = out.tolist()
        return out
    elif return_type == 1:
        return dict(zip(attribute_name, out.tolist()))
    elif return_type == 2:
        return np.array(attribute_name)[out.astype(bool)].tolist()
    
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='config/base_extraction.yml', type=str)
    args = parser.parse_args()

    path_image = "D:/datasets/peta/processed/images/00001.png"
    image = imread(path_image)
    
    result = extractor(args.config, image, 2)
    print(result)
