from matplotlib.pyplot import broken_barh
import numpy
import torch
import argparse
import sys
sys.path.append('.')
import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from utils import read_json

from data.image import build_datasource
from models import build_model

def imread(path):
    image = Image.open(path)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: ./config.json)')
    parser.add_argument('-r', '--resume', default='', type=str, help='resume file path (default: .)')
    args = parser.parse_args()

    config = read_json(args.config)
    config.update({'resume': args.resume})

    datasource = build_datasource(
        name=config['data']['name'],
        root_dir=config['data']['data_dir'],
        download=config['data']['download'],
        extract=config['data']['extract'],
        use_tqdm=False if 'tqdm' not in config else config['data']['tqdm'])
    
    transform = transforms.Compose([
        transforms.Resize(size=datasource.get_image_size()),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model, _ = build_model(config['model'], num_classes=len(datasource.get_attribute()))

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    print('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    width, height = datasource.get_image_size()

    attribute_name = datasource.get_attribute()

    for img_path, label in datasource.get_data('train'):
        label = label.astype(bool)
        img_orig = imread(img_path)
        img = transform(img_orig)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)

        output, heatmaps = model.get_heat_maps(img)
        output = torch.sigmoid(output)
        output = torch.squeeze(output).cpu().detach().numpy()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        output = output.astype(bool)
        heatmaps = torch.squeeze(heatmaps).cpu().detach().numpy()
        
        # RGB image
        img_orig = cv2.cvtColor(numpy.array(img_orig), cv2.COLOR_RGB2BGR)
        img_orig = cv2.resize(img_orig, (width, height))
        
        list_image = [img_orig]
        title = ''
        for i, (heatmap, attribute) in enumerate(zip(heatmaps, attribute_name)):
            if i == 5:
                break
            # heatmaps
            am = cv2.resize(heatmap, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_orig*0.35 + am*0.65
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)
            
            list_image.append(overlapped)
            title += attribute + ': pred-' + str(output[i]) + ', label-' + str(label[i]) +'   |    '

            # GRID_SPACING = 10
            # # save images in a single figure (add white spacing between images)
            # # from left to right: original image, activation map, overlapped image
            # grid_img = 255 * np.ones((height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8)
            # grid_img[:, :width, :] = img_orig[:, :, ::-1]
            # grid_img[:, width + GRID_SPACING:2*width + GRID_SPACING, :] = am
            # grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped

            # plt.imshow(grid_img)
            # plt.title(attribute + ': ' + str(output[i]))
            # plt.show()
        plt.title(title)
        img = np.concatenate(list_image, axis=1)
        plt.imshow(img)
        plt.show()

