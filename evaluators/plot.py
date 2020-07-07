import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from utils import aggregate

def plot_loss(dpath, list_dname, output_path, low = .05, high = .95, com=10):
    """ Plot loss and accuracy from tensorboard file
    Args:
        dpath (str): path to folder contain (eg: saved/logs)
        list_dname (list(str)): list of run_id to plot.
        output_path (str): path to save csv file after concat logs from different run time
    Return:
    """
    ax = plt.gca()
    dict_data_frame = aggregate(dpath, list_dname, output_path, False)
    color = ['red', 'green']
    index = 0
    for key in dict_data_frame.keys():
        df = dict_data_frame[key]
        quant_df = df.quantile([low, high])
        df = df[(df['Value'] > quant_df.loc[low, 'Value']) & (df['Value'] < quant_df.loc[high, 'Value'])]
        df['Value'] = df['Value'].ewm(com=com).mean()
        df.plot.line(x='Step', y ='Value', label=key, color=color[index], ax=ax)
        index += 1
    plt.show()


def show_image(distances, queryset, testset, k=5, num_image=5, size_img=(2.5, 5)):
    plt.figure(figsize=((k+1)*size_img[0], num_image*size_img[1]))
    to_show = []
    for i in range(num_image):
        index = random.randint(0, len(distances)-1)
        query = queryset.get_img(index)
        topk = distances[index].topk(k, largest=False)
        to_display = [np.array(query)]
        for distance, index in zip(topk.values, topk.indices):
            img = np.array(testset.get_img(int(index.data)))
            to_display.append(img)
        to_display = np.concatenate(to_display, axis=1)
        to_show.append(to_display)
    to_show = np.concatenate(to_show, axis=0)
    plt.imshow(to_show)
    plt.axis('off')
    plt.show()
