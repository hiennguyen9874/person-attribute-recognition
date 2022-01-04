import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from data.image.pa_100k import PA_100K
from data.image.peta import Peta
from data.image.ppe import PPE
from data.image.ppe_two_attribute import PPE_Two
from data.image.wider import Wider

__datasets = {'pa_100k': PA_100K, 'peta': Peta, 'ppe': PPE, 'ppe_two': PPE_Two, 'wider': Wider}

def build_datasource(name, root_dir, download=True, extract=True, use_tqdm=True):
    if name not in list(__datasets.keys()):
        raise KeyError
    return __datasets[name](root_dir, download, extract, use_tqdm)
