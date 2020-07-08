# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

from tensorflow.core.util.event_pb2 import Event
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import ast
import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

FOLDER_NAME = 'aggregates'


def extract(dpath, dname, dpart):
    scalar_accumulators = [EventAccumulator(os.path.join(dpath, dname, dpart)).Reload().scalars]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events] for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    wall_times_per_key = [np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events], axis=0) for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events] for all_scalar_events in all_scalar_events_per_key]

    all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))

    return all_per_key


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def aggregate(dpath, list_dname, list_part, output_path=None):
    # extracts_per_subpath = [extract(dpath, dname, dpart) for dname in list_dname for dpart in list_part]
    extracts_per_subpath = []
    for i in range(len(list_dname)):
        extracts_per_subpath.append(dict())
        for dpart in list_part:
            extracts_per_subpath[i][dpart] = extract(dpath, list_dname[i], dpart)
    pass

    list_key = []
    # list_data_frame = []
    # list_key = dict()
    list_data_frame = dict()
    for i in range(len(list_dname)):
        # list_key[list_dname[i]] = dict()
        list_data_frame[list_dname[i]] = dict()
        for part in list_part:
            for key, (steps, wall_times, values) in extracts_per_subpath[i][part].items():
                df = pd.DataFrame(list(zip(wall_times, steps, np.array(values).reshape(-1))), columns=['Wall time', 'Step', 'Value'])
                list_data_frame[list_dname[i]][part] = df
                list_key.append(part)
    
    ret = dict()
    for key in list_key:
        data_frame = pd.concat([list_data_frame[x][key] for x in list_dname])
        if output_path != None:
            file_name = os.path.join(output_path, get_valid_filename(key) + '.csv')
            data_frame.to_csv(file_name)
        ret[key] = data_frame
    return ret


if __name__ == '__main__':
    path = os.path.join('saved', 'logs')
    part = ['Accuracy_Train', 'Accuracy_Val', 'Loss_Train', 'Loss_Val']
    aggregate(path, ['0707_131336'], part, 'output')
