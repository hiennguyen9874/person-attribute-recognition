from .pa_100k import PA_100K
from .peta import Peta
from .ppe import PPE
from .ppe_two_attribute import PPE_Two

__datasets = {'pa_100k': PA_100K, 'peta': Peta, 'ppe': PPE, 'ppe_two': PPE_Two}

def build_datasource(name, root_dir, download, extract, use_tqdm):
    if name not in list(__datasets.keys()):
        raise KeyError
    return __datasets[name](root_dir, download, extract, use_tqdm)