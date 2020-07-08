import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('.')


from utils import aggregate

def plot_loss(dpath, list_dname, list_part, path_figure ,title = None, low = .05, high = .95, com=0):
    """ Plot loss and accuracy from tensorboard file
    Args:
        dpath (str): path to folder contain (eg: saved/logs)
        list_dname (list(str)): list of run_id to plot.
        list_part (list(str)): list of phase (eg: ['Accuracy_Train', 'Accuracy_Val', 'Loss_Train', 'Loss_Val'])
        output_path (str): path to save csv file after concat logs from different run time
        title (str): title for figure
        low, high (float): 
    Return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    dict_data_frame = aggregate(dpath, list_dname, list_part)
    colors = ['red', 'green', 'blue', 'orange']
    for key, color in zip(dict_data_frame.keys(), colors):
        df = dict_data_frame[key]
        quant_df = df.quantile([low, high])
        df = df[(df['Value'] > quant_df.loc[low, 'Value']) & (df['Value'] < quant_df.loc[high, 'Value'])]
        df['Value'] = df['Value'].ewm(com=com).mean()
        df.plot.line(x='Step', y ='Value', label=key.split('_')[1], color=color, ax=ax1 if key.split('_')[0]=='Accuracy' else ax2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    # show grid
    ax1.grid()
    
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    # Hide the right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    # show grid
    ax2.grid()

    if title != None:
        fig.suptitle(title)
    # plt.show()
    fig.savefig(path_figure, dpi=300)


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

if __name__ == "__main__":
    path = os.path.join('saved', 'logs')
    part = ['Accuracy_Train', 'Accuracy_Val', 'Loss_Train', 'Loss_Val']
    run_id = '0708_140143'
    plot_loss(
        dpath=path,
        list_dname=[run_id],
        list_part=part,
        path_figure=os.path.join(path, run_id, '{}.png'.format(run_id)),
        title='OSNet + BCEWithLogitsLoss, PPE dataset')