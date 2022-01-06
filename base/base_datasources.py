import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import zipfile
import tarfile

from tqdm.auto import tqdm
from shutil import copy2


class BaseDataSource(object):
    def __init__(self, root_dir, dataset_dir, phase=["train", "val", "test"], **kwargs):

        self.phase = phase
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir

    def _exists(self, extract_dir):
        raise NotImplementedError

    def get_data(self, phase="train"):
        r"""get data, must return list of (image_path, label)"""
        raise NotImplementedError

    def get_phase(self):
        r"""get list of phase."""
        return self.phase

    def _check_file_exits(self):
        r"""check all image in datasource exists"""
        for phase in self.phase:
            for path, label in self.get_data(phase):
                if not os.path.exists(path):
                    raise FileExistsError

    def show_some_image(self, num_image, num_per_row=10):
        import cv2
        import math
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        from utils import imread

        all_rand_path = np.random.choice(range(len(self.get_data("train"))), num_image)
        all_rand_image = [
            cv2.resize(imread(self.get_data("train")[x][0]), (128, 256))
            for x in all_rand_path
        ]
        fig, ax = plt.subplots(math.ceil(num_image / num_per_row), num_per_row)
        if math.ceil(num_image / num_per_row) == 1:
            for j in range(num_per_row):
                ax[j].axis("off")
                ax[j].imshow(all_rand_image[j])
        else:
            # fig.tight_layout()
            for i in range(math.ceil(num_image / num_per_row)):
                for j in range(num_per_row):
                    ax[i][j].axis("off")
                    if i * num_per_row + j < num_image:
                        ax[i][j].imshow(all_rand_image[i * num_per_row + j])
        # plt.show()
        # matplotlib.use("pgf")
        # matplotlib.rcParams.update({
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        # })
        plt.savefig("{}_show.pdf".format(self.dataset_dir))
