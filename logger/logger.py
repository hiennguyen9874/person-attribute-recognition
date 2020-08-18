
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import logging
import logging.config

from pathlib import Path

from utils import read_json

def setup_logging(save_dir, log_config='logger_config.json', default_level=logging.INFO):
    r""" Setup logging configuration
    """
    log_config = Path(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))), log_config))
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = os.path.join(save_dir, handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
