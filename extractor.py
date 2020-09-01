import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import torch
import pickle
import argparse

import numpy as np

from data import datamanager

from PIL import Image
from torchvision import transforms

from models import build_model
from utils import read_config

def imread(path):
    return Image.open(path)

def extractor(path_config, path_attribute, path_model, image, return_type=0):
    r""" 

    Args:
        path_config ([type]): [description]
        path_attribute ([type]): [description]
        path_model ([type]): [description]
        image ([type]): [description]
        return_type (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    config = read_config(path_config, False)

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    attribute_name = pickle.load(open(path_attribute, 'rb'))
    
    model, _ = build_model(config, num_classes=len(attribute_name))
    checkpoint = torch.load(path_model, map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    image_processing = transforms.Compose([
        transforms.Resize(size=(256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = image_processing(image)
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

    path_image = "/datasets/peta/processed/PETA-New/images/00009.png"
    image = imread(path_image)
    
    # result = extractor(args.config, image, 2)
    result = extractor(path_config=args.config, path_attribute='peta_attribute.pkl', path_model="/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0731_232453/model_best_accuracy.pth", image=image, return_type=0)
    print(result)

