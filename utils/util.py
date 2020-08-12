import os
import json

import sys
sys.path.append('.')

import shutil

import numpy as np
from utils.read_config import read_config

from PIL import Image

from pathlib import Path
from collections import OrderedDict

__all__ = ['imread', 'read_json', 'write_json', 'rmdir', 'config_to_str', 'array_interweave', 'neq']

def imread(path):
    image = Image.open(path)
    return image

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def rmdir(path, remove_parent=True):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if remove_parent:
        if os.path.exists(path):
            os.rmdir(path)

def config_to_str(config):
    return "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in config.items()) + "}"

def array_interweave(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

def neq(x, y, z):
    return (x != y or z) or y != z

