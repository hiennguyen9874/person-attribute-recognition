from .pa_100k import PA_100K
from .peta import Peta
from .ppe import PPE

__datasets = {'pa_100k': PA_100K, 'peta': Peta, 'ppe': PPE}

def build_datasource(name, root_dir, download, extract):
    if name not in list(__datasets.keys()):
        raise KeyError
    return __datasets[name](root_dir, download, extract)