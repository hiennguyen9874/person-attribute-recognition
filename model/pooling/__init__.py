import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

from model.pooling.gem_pooling import GeneralizedMeanPoolingP
from model.pooling.avg_pooling import AvgPooling2d

__poolings = {'gem_pooling': GeneralizedMeanPoolingP, 'avg_pooling': AvgPooling2d}

def build_pooling(name, pooling_size=1):
    if name not in list(__poolings.keys()):
        raise KeyError('name error, name must in %s' % (str(list(__poolings.keys()))))
    return __poolings[name](output_size=pooling_size)

if __name__ == "__main__":
    build_pooling('gem_pooling')