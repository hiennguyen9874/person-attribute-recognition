import sys
sys.path.append('.')

import yaml
import json
import collections.abc

__all__ = ['read_config']

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def read_config(path_config: str, path_config_base='config/base.yml'):
    base_config = yaml.safe_load(open(path_config_base))
    new_config = yaml.safe_load(open(path_config))
    all_config = update(base_config, new_config)
    if all_config['lr_scheduler']['enable']:
        for key, value in all_config['lr_scheduler']['default'][all_config['lr_scheduler']['name']].items():
            if key not in all_config['lr_scheduler']:
                all_config['lr_scheduler'][key] = value
    return all_config

if __name__ == "__main__":
    config = read_config('config/test.yml')
    print(json.dumps(config, indent = 4))