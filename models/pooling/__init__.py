from .gem_pooling import GeneralizedMeanPoolingP
from .avg_pooling import FastGlobalAvgPool2d

__poolings = {'gem_pooling': GeneralizedMeanPoolingP, 'avg_pooling': FastGlobalAvgPool2d}

def build_pooling(name):
    if name not in list(__poolings.keys()):
        raise KeyError('name error, name must in %s' % (str(list(__poolings.keys()))))
    return __poolings[name]()

if __name__ == "__main__":
    build_pooling('fda')