import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import os
import cv2
import json
import yaml
import shutil
import subprocess
import collections
import pkg_resources

import numpy as np
from utils.read_config import read_config

from pathlib import Path
from collections import OrderedDict

__all__ = ['imread', 'read_json', 'write_json', 'rmdir', 'config_to_str', 'array_interweave', 'array_interweave3', 'neq', 'pip_install']

def imread(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

def array_interweave3(a, b, c):
    d = np.empty((a.size + b.size + c.size,), dtype=a.dtype)
    d[0::3] = a
    d[1::3] = b
    d[2::3] = c
    return d

def neq(x, y, z):
    return (x != y or z) or y != z

def pip_install(package, version=None):
    if version != None:
        if pkg_resources.get_distribution(package).version ==  version:
            return
        subprocess.check_call([sys.executable, "-m", "pip", "install", package+"=="+version])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
