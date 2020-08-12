import sys
sys.path.append('.')

import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from data import DataManger_Epoch, build_datasource
from utils import read_config

def imread(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def main(config):
    cfg_data = config['data']
    datasource = build_datasource(
        name=cfg_data['name'],
        root_dir=cfg_data['data_dir'],
        download=cfg_data['download'],
        extract=cfg_data['extract'],
        use_tqdm=True)
    
    attribute_name = datasource.get_attribute()
    width, height = datasource.get_image_size()
    data_train = datasource.get_data('test')
    random.shuffle(data_train)

    for path, label in data_train:
        img_orig = imread(path)

        # RGB image
        # img_orig = cv2.resize(img_orig, (height, width))

        # add label below image
        # plt.figtext(0.5, 0.05, str([x + ": " + str(True if y == 1.0 else False) for x, y in zip(attribute_name, label)]), ha="center", fontsize=16, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        print(str([x + ": " + str(True if y == 1.0 else False) for x, y in zip(attribute_name, label) if y == 1]))
        print()
        
        # full windows
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.imshow(img_orig)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: ./config.json)')
    args = parser.parse_args()
    
    config = read_config(args.config)
    main(config)