import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from data.image.pa_100k import PA_100K
from data.image.peta import Peta

__datasets = {
    "pa_100k": PA_100K,
    "peta": Peta,
}


def build_datasource(name, root_dir):
    if name not in list(__datasets.keys()):
        raise KeyError
    return __datasets[name](root_dir)
