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

def extract(path_to_folder):
    scalar_accumulators = [EventAccumulator(path_to_folder).Reload().scalars]

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


def aggregate(dpath, list_dname, output_path=None):
    extracts_per_subpath = dict()
    for dname in list_dname:
        extracts_per_subpath[dname] = dict()
        list_part = [f.name for f in os.scandir(os.path.join(dpath, dname)) if f.is_dir()]
        list_part.sort()
        extracts_per_subpath[dname]['list_part'] = list_part
        set_part1 = set()
        set_part2 = set()
        for dpart in list_part:
            x, y = dpart.split('_')
            set_part1.add(x)
            set_part2.add(y)
        for part1 in set_part1:
            extracts_per_subpath[dname][part1] = dict()
            for part2 in set_part2:
                extracts_per_subpath[dname][part1][part2] = extract(os.path.join(dpath, dname, part1 + '_' + part2))
    
    if len(list_dname) > 1:
        for i in range(0, len(list_dname)-1):
            for j in range(i+1, len(list_dname)):
                if len(set(extracts_per_subpath[list_dname[i]]['list_part']).difference(set(extracts_per_subpath[list_dname[j]]['list_part']))) != 0:
                    raise KeyError


    list_part = [set(), set()]
    list_data_frame = dict()
    for key1, value1 in extracts_per_subpath.items():
        list_data_frame[key1] = dict()
        for key2, value2 in extracts_per_subpath[key1].items():
            if key2 == 'list_part':
                continue
            list_data_frame[key1][key2] = dict()
            for key3, value3 in extracts_per_subpath[key1][key2].items():
                for key, (steps, wall_times, values) in extracts_per_subpath[key1][key2][key3].items():
                    df = pd.DataFrame(list(zip(wall_times, steps, np.array(values).reshape(-1))), columns=['Wall time', 'Step', 'Value'])
                    list_data_frame[key1][key2][key3] = df
                    list_part[0].add(key2)
                    list_part[1].add(key3)
    list_part[0] = list(list_part[0])
    list_part[1] = list(list_part[1])

    ret = dict()
    for part1 in list_part[0]:
        ret[part1] = dict()
        for part2 in list_part[1]:
            data_frame = pd.concat([list_data_frame[x][part1][part2] for x in list_dname])
            if output_path != None:
                file_name = os.path.join(output_path, get_valid_filename(part1 + '_' + part2) + '.csv')
                data_frame.to_csv(file_name)
            ret[part1][part2] = data_frame

    return ret, list_part

def aggregate1(dpath, list_dname, output_path=None):
    extracts_per_subpath = {dname: extract(os.path.join(dpath, dname)) for dname in list_dname}

    list_data_frame = dict()
    list_key = set()
    for dname, value in extracts_per_subpath.items():
        list_data_frame[dname] = dict()
        for key, (steps, wall_times, values) in extracts_per_subpath[dname].items():
            df = pd.DataFrame(list(zip(wall_times, steps, np.array(values).reshape(-1))), columns=['Wall time', 'Step', 'Value'])
            list_data_frame[dname][key] = df
            list_key.add(key)

    ret = dict()
    for key in iter(list_key):
        data_frame = pd.concat([list_data_frame[dname][key] for i in list_dname])
        if output_path != None:
            file_name = os.path.join(output_path, get_valid_filename(key) + '.csv')
            data_frame.to_csv(file_name)
        ret[key] = data_frame
    return ret

if __name__ == '__main__':
    path = os.path.join('saved', 'logs')
    aggregate1(path, ['0710_120633'])
