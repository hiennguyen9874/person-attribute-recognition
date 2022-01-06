from collections import OrderedDict
from pathlib import Path
import numpy as np
import pkg_resources
import collections
import subprocess
import shutil
import copy
import json
import yaml
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

__all__ = [
    "read_config",
    "copyTree",
    "imread",
    "read_json",
    "write_json",
    "rmdir",
    "config_to_str",
    "array_interweave",
    "array_interweave3",
    "neq",
    "pip_install",
    "COLOR",
]


def read_config(path_config: str, base=True):
    r"""read config yml file, return dict"""

    def update(d, u):
        r"""deep update dict.
        copied from here: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        """
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def deep_merge(config, name):
        new_config = copy.deepcopy(config)
        for key, value in config[name]["default"][config[name]["name"]].items():
            if key not in config[name]:
                new_config[name][key] = value
        return new_config

    new_config = yaml.safe_load(open(path_config))
    if not base:
        return new_config

    base_config = yaml.safe_load(open(new_config["base"]))
    all_config = update(base_config, new_config)
    all_config = deep_merge(all_config, "loss")
    all_config = deep_merge(all_config, "optimizer")
    if all_config["lr_scheduler"]["enable"]:
        all_config = deep_merge(all_config, "lr_scheduler")
    return all_config


def copyTree(src, dst):
    r"""Move and overwrite files and folders

    Args:
        src (str): [description]
        dst (str): [description]
    """

    def forceMergeFlatDir(srcDir, dstDir):
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)
        for item in os.listdir(srcDir):
            srcFile = os.path.join(srcDir, item)
            dstFile = os.path.join(dstDir, item)
            forceCopyFile(srcFile, dstFile)

    def forceCopyFile(sfile, dfile):
        if os.path.isfile(sfile):
            shutil.copy2(sfile, dfile)

    def isAFlatDir(sDir):
        for item in os.listdir(sDir):
            sItem = os.path.join(sDir, item)
            if os.path.isdir(sItem):
                return False
        return True

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isfile(s):
            if not os.path.exists(dst):
                os.makedirs(dst)
            forceCopyFile(s, d)
        if os.path.isdir(s):
            isRecursive = not isAFlatDir(s)
            if isRecursive:
                copyTree(s, d)
            else:
                forceMergeFlatDir(s, d)


def imread(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
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
    r"""install package from pip

    Args:
        package (str): name of package
        version (str, optional): version of package. Defaults to None.
    """
    if version != None:
        if pkg_resources.get_distribution(package).version == version:
            return
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package + "==" + version]
        )
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class COLOR:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
