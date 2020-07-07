import os
import json
import sys
import shutil
sys.path.append('.')

from pathlib import Path
from collections import OrderedDict


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
        os.rmdir(path)

def config_to_str(config):
    return "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in config.items()) + "}"

if __name__ == "__main__":
    shutil.rmtree('/home/hien/Documents/Models_Attribute_Recognition/OSNet_Person_Attribute_Refactor/test')