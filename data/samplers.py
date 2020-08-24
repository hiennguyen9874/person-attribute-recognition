import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import copy
import torch
import numpy as np

from tqdm.auto import tqdm
from itertools import repeat
from collections import defaultdict


r""" Person attribute recognition sampler
"""
class RandomBalanceBatchSamplerAttribute(torch.utils.data.Sampler):
    r""" Episode sampler, random uniform 'n' attribute, each attribute, random 'k' positive sampler and 'l' negative sampler
    Args:
        datasource (list of tuple): data from data.image.get_data()
        attribute_name: list of attribute in dataset
        num_attribute: num of attribute in one episode
        num_positive: num of positive sampler in each attribute
        num_negative: num of negative sampler in each attribute
        num_iterator: num of iterator in each epoch.
        shuffle: shuffle data before sampler
    """
    def __init__(
        self,
        datasource,
        attribute_name,
        num_attribute,
        num_positive,
        num_negative,
        num_iterator,
        shuffle=True,
        **kwargs):

        assert num_attribute <= len(attribute_name), 'num of attribute in one batch must less than num of attribute in dataset'
        
        self.datasource = datasource
        self.attribute_name = list(enumerate(attribute_name))
        self.num_attribute = num_attribute
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.num_iterator = num_iterator
        self.shuffle = shuffle

        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)

        for index, (_, label) in enumerate(self.datasource):
            for i, attribute in enumerate(attribute_name):
                if label[i] == True:
                    self.pos_dict[attribute].append(index)
                else:
                    self.neg_dict[attribute].append(index)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.attribute_name)
            for _, attribute in self.attribute_name:
                np.random.shuffle(self.pos_dict[attribute])
                np.random.shuffle(self.neg_dict[attribute])
        for _ in range(self.num_iterator):
            idx_selected_attribute = np.random.choice(len(self.attribute_name), size=self.num_attribute, replace=True)
            batch = []
            for idx in idx_selected_attribute:
                index, attribute = self.attribute_name[idx]
                pos_idxs = np.random.choice(self.pos_dict[attribute], size=self.num_positive, replace=True)
                neg_idxs = np.random.choice(self.neg_dict[attribute], size=self.num_negative, replace=True)
                batch.extend(list(zip(pos_idxs, repeat(index))))
                batch.extend(list(zip(neg_idxs, repeat(index))))
            yield batch

    def __len__(self):
        return self.num_iterator

class RandomBatchSamplerAttribute(torch.utils.data.Sampler):
    r""" Episode sampler, random attribute from multinomial distribution, 
        each attribute, random 'k' positive sampler and 'l' negative sampler.
    Args:
        datasource (list of tuple): data from data.image.get_data()
        weight (np.array): weight of training set.
        attribute_name: list of attribute in dataset
        num_attribute: num of attribute in one episode
        num_positive: num of positive sampler in each attribute
        num_negative: num of negative sampler in each attribute
        num_iterator: num of iterator in each epoch.
        shuffle: shuffle data before sampler
    """
    def __init__(
        self,
        datasource,
        weight,
        attribute_name,
        num_attribute,
        num_positive,
        num_negative,
        num_iterator,
        shuffle=True,
        **kwargs):

        assert num_attribute <= len(attribute_name), 'num of attribute in one batch must less than num of attribute in dataset'
        
        self.datasource = datasource
        
        self.weight = torch.exp(1-torch.tensor(weight, dtype=torch.float))
        self.weight /= torch.sum(self.weight)

        self.attribute_name = list(enumerate(attribute_name))
        self.num_attribute = num_attribute
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.num_iterator = num_iterator
        self.shuffle = shuffle

        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)

        for index, (_, label) in enumerate(self.datasource):
            for i, attribute in enumerate(attribute_name):
                if label[i] == True:
                    self.pos_dict[attribute].append(index)
                else:
                    self.neg_dict[attribute].append(index)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.attribute_name)
            for _, attribute in self.attribute_name:
                np.random.shuffle(self.pos_dict[attribute])
                np.random.shuffle(self.neg_dict[attribute])
        for _ in range(self.num_iterator):
            idx_selected_attribute = torch.multinomial(self.weight, self.num_attribute, replacement=True)
            batch = []
            for idx in idx_selected_attribute:
                index, attribute = self.attribute_name[idx]
                pos_idxs = np.random.choice(self.pos_dict[attribute], size=self.num_positive, replace=True)
                neg_idxs = np.random.choice(self.neg_dict[attribute], size=self.num_negative, replace=True)
                batch.extend(list(zip(pos_idxs, repeat(index))))
                batch.extend(list(zip(neg_idxs, repeat(index))))
            yield batch

    def __len__(self):
        return self.num_iterator


class RandomBatchSamplerAttribute1(torch.utils.data.Sampler):
    r""" Episode sampler, random attribute from multinomial distribution, 
        each attribute, random 'k' positive sampler and 'l' negative sampler.
    Args:
        datasource (list of tuple): data from data.image.get_data()
        weight (np.array): weight of training set.
        attribute_name: list of attribute in dataset
        num_attribute: num of attribute in one episode
        num_positive: num of positive sampler in each attribute
        num_negative: num of negative sampler in each attribute
        num_iterator: num of iterator in each epoch.
        shuffle: shuffle data before sampler
    """
    def __init__(
        self,
        datasource,
        weight,
        attribute_name,
        num_attribute,
        num_positive,
        num_negative,
        num_iterator,
        shuffle=True,
        **kwargs):

        assert num_attribute <= len(attribute_name), 'num of attribute in one batch must less than num of attribute in dataset'
        
        self.datasource = datasource
        
        self.weight = torch.exp(torch.tensor(weight, dtype=torch.float))
        self.weight /= torch.sum(self.weight)

        self.attribute_name = list(enumerate(attribute_name))
        self.num_attribute = num_attribute
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.num_iterator = num_iterator
        self.shuffle = shuffle

        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)

        for index, (_, label) in enumerate(self.datasource):
            for i, attribute in enumerate(attribute_name):
                if label[i] == True:
                    self.pos_dict[attribute].append(index)
                else:
                    self.neg_dict[attribute].append(index)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.attribute_name)
            for _, attribute in self.attribute_name:
                np.random.shuffle(self.pos_dict[attribute])
                np.random.shuffle(self.neg_dict[attribute])
        for _ in range(self.num_iterator):
            idx_selected_attribute = torch.multinomial(self.weight, self.num_attribute, replacement=True)
            batch = []
            for idx in idx_selected_attribute:
                index, attribute = self.attribute_name[idx]
                pos_idxs = np.random.choice(self.pos_dict[attribute], size=self.num_positive, replace=True)
                neg_idxs = np.random.choice(self.neg_dict[attribute], size=self.num_negative, replace=True)
                batch.extend(list(zip(pos_idxs, repeat(index))))
                batch.extend(list(zip(neg_idxs, repeat(index))))
            yield batch

    def __len__(self):
        return self.num_iterator


class RandomBatchSamplerAttributeWeight(torch.utils.data.Sampler):
    r""" Episode sampler, random attribute based weight, 
        each attribute, random 'k' positive sampler and 'l' negative sampler.
    Args:
        datasource (list of tuple): data from data.image.get_data()
        weight (np.array): weight of training set.
        attribute_name: list of attribute in dataset
        num_attribute: num of attribute in one episode
        num_positive: num of positive sampler in each attribute
        num_negative: num of negative sampler in each attribute
        num_iterator: num of iterator in each epoch.
        shuffle: shuffle data before sampler
    """
    def __init__(
        self,
        datasource,
        weight,
        attribute_name,
        num_attribute,
        num_sampler,
        num_iterator,
        shuffle=True,
        **kwargs):

        assert num_attribute <= len(attribute_name), 'num of attribute in one batch must less than num of attribute in dataset'
        
        self.datasource = datasource
        self.attribute_name = list(enumerate(attribute_name))
        self.num_attribute = num_attribute
        self.num_iterator = num_iterator
        self.shuffle = shuffle

        weight1 = np.exp(1-weight)
        weight2 = np.exp(weight)
        self.num_positive = np.rint(num_sampler*weight1/(weight1+weight2)).astype(int)
        self.num_negative = num_sampler - self.num_positive

        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)

        for index, (_, label) in enumerate(self.datasource):
            for i, attribute in enumerate(attribute_name):
                if label[i] == True:
                    self.pos_dict[attribute].append(index)
                else:
                    self.neg_dict[attribute].append(index)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.attribute_name)
            for _, attribute in self.attribute_name:
                np.random.shuffle(self.pos_dict[attribute])
                np.random.shuffle(self.neg_dict[attribute])
        for _ in range(self.num_iterator):
            idx_selected_attribute = np.random.choice(len(self.attribute_name), size=self.num_attribute, replace=True)
            batch = []
            for idx in idx_selected_attribute:
                index, attribute = self.attribute_name[idx]
                pos_idxs = np.random.choice(self.pos_dict[attribute], size=self.num_positive[idx], replace=True)
                neg_idxs = np.random.choice(self.neg_dict[attribute], size=self.num_negative[idx], replace=True)
                batch.extend(list(zip(pos_idxs, repeat(index))))
                batch.extend(list(zip(neg_idxs, repeat(index))))
            yield batch

    def __len__(self):
        return self.num_iterator

class RandomBatchSamplerAttributeWeight1(torch.utils.data.Sampler):
    r""" Episode sampler, random attribute based uniform distribution, 
        each attribute, random 'k' positive sampler and 'l' negative sampler based weight of attribute.

    Args:

        datasource (list of tuple): data from data.image.get_data()

        weight (np.array): weight of training set.

        attribute_name: list of attribute in dataset

        num_attribute: num of attribute in one episode

        num_positive: num of positive sampler in each attribute

        num_negative: num of negative sampler in each attribute

        num_iterator: num of iterator in each epoch.

        shuffle: shuffle data before sampler
    """
    def __init__(
        self,
        datasource,
        weight,
        attribute_name,
        num_attribute,
        num_sampler,
        num_iterator,
        shuffle=True,
        **kwargs):

        assert num_attribute <= len(attribute_name), 'num of attribute in one batch must less than num of attribute in dataset'
        
        self.datasource = datasource
        self.attribute_name = list(enumerate(attribute_name))
        self.num_attribute = num_attribute
        self.num_iterator = num_iterator
        self.shuffle = shuffle

        weight1 = np.exp(weight)
        weight2 = np.exp(1-weight)
        self.num_positive = np.rint(num_sampler*weight1/(weight1+weight2)).astype(int)
        self.num_negative = num_sampler - self.num_positive

        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)

        for index, (_, label) in enumerate(self.datasource):
            for i, attribute in enumerate(attribute_name):
                if label[i] == True:
                    self.pos_dict[attribute].append(index)
                else:
                    self.neg_dict[attribute].append(index)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.attribute_name)
            for _, attribute in self.attribute_name:
                np.random.shuffle(self.pos_dict[attribute])
                np.random.shuffle(self.neg_dict[attribute])
        for _ in range(self.num_iterator):
            idx_selected_attribute = np.random.choice(len(self.attribute_name), size=self.num_attribute, replace=True)
            batch = []
            for idx in idx_selected_attribute:
                index, attribute = self.attribute_name[idx]
                pos_idxs = np.random.choice(self.pos_dict[attribute], size=self.num_positive[idx], replace=True)
                neg_idxs = np.random.choice(self.neg_dict[attribute], size=self.num_negative[idx], replace=True)
                batch.extend(list(zip(pos_idxs, repeat(index))))
                batch.extend(list(zip(neg_idxs, repeat(index))))
            yield batch

    def __len__(self):
        return self.num_iterator


r""" Person re-identification sampler
"""
class SubsetIdentitySampler(torch.utils.data.Sampler):
    def __init__(self, datasource, batch_size, shuffle = True, index_dict=None, **kwargs):
        self.datasource = datasource
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = 0
        self.list_index = []
        if index_dict == None:
            self.index_dict = defaultdict(list)
            print("Processing before sampler")
            with tqdm(total=len(self.datasource)) as pbar:
                for index, (_, person_id, _, _) in enumerate(self.datasource):
                    self.index_dict[person_id].append(index)
                    self.list_index.append(index)
                    pbar.update(1)
                    self.length += 1
        else:
            self.index_dict = index_dict
            self._to_list()
            self.length = len(self.list_index)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.list_index)
        return iter(self.list_index)

    def __len__(self):
        return self.length

    def _to_list(self):
        for person_id in self.index_dict.keys():
            self.list_index.extend(self.index_dict[person_id])
    
    def get_num_classes(self):
        return len(self.index_dict)
    
    def split(self, validation_count):
        left_index_dict = defaultdict(list)
        right_index_dict = defaultdict(list)
        for person_id in self.index_dict.keys():
            idx_full = self.index_dict[person_id]
            np.random.shuffle(idx_full)
            left_index = idx_full[validation_count:]
            right_index = idx_full[0:validation_count]
            left_index_dict[person_id].extend(left_index)
            right_index_dict[person_id].extend(right_index)
        return SubsetIdentitySampler(self.datasource, self.batch_size, self.shuffle, left_index_dict), SubsetIdentitySampler(self.datasource, self.batch_size,  self.shuffle, right_index_dict)

class RandomIdentitySampler(torch.utils.data.Sampler):
    r""" https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/sampler.py
    """
    def __init__(self, datasource, batch_size=1, num_instances=1, index_dict=None, **kwargs):
        self.datasource = datasource

        if batch_size <= num_instances:
            raise ValueError('batch_size <= num_instances')
        
        if batch_size % num_instances != 0:
            raise ValueError('batch_size % num_instances != 0')

        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        if index_dict == None:
            self.index_dict = defaultdict(list)
            print("Processing before sampler")
            with tqdm(total=len(self.datasource)) as pbar:
                for index, (_, person_id, _) in enumerate(self.datasource):
                    self.index_dict[person_id].append(index)
                    pbar.update(1)
        else:
            self.index_dict = index_dict

        self.person_ids = list(self.index_dict.keys())

        self.length = 0
        for person_id in self.person_ids:
            num = len(self.index_dict[person_id])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __len__(self):
        return self.length

    def __iter__(self):
        batch_idx_dict = defaultdict(list)
        for person_id in self.person_ids:
            idxs = self.index_dict[person_id]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idx_dict[person_id].append(batch_idxs)
                    batch_idxs = []
        
        avai_pids = copy.deepcopy(self.person_ids)
        batch = []
        while len(avai_pids) >= self.num_pids_per_batch:
            idx_selected_pids = np.random.choice(len(avai_pids), self.num_pids_per_batch)
            for idx in idx_selected_pids:
                batch.extend(batch_idx_dict[avai_pids[idx]].pop(0))
                if len(batch_idx_dict[avai_pids[idx]]) == 0:
                    avai_pids.remove(avai_pids[idx])
        return iter(batch)

    def split(self, rate=0.5):
        left_index_dict = defaultdict(list)
        right_index_dict = defaultdict(list)
        for person_id in self.index_dict.keys():
            left_count = int(len(self.index_dict[person_id]) * rate)
            left_index = self.index_dict[person_id][0:left_count]
            right_index = self.index_dict[person_id][left_count:]
            left_index_dict[person_id] = left_index
            right_index_dict[person_id] = right_index
        return RandomIdentitySampler(self.datasource, self.batch_size, self.num_instances, left_index_dict), RandomIdentitySampler(self.datasource, self.batch_size, self.num_instances, right_index_dict)

class RandomBalanceBatchSampler(torch.utils.data.Sampler):
    def __init__(self, datasource, batch_size=1, num_instances=1, num_iterators = 1, index_dict=None, **kwargs):
        self.datasource = datasource

        if batch_size <= num_instances:
            raise ValueError('batch_size <= num_instances')
        
        self.batch_size = batch_size
        self.num_iterators = num_iterators
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        if index_dict == None:
            self.index_dict = defaultdict(list)
            print("Processing before sampler")
            with tqdm(total=len(self.datasource)) as pbar:
                for index, (_, person_id, _, _) in enumerate(self.datasource):
                    self.index_dict[person_id].append(index)
                    pbar.update(1)
        else:
            self.index_dict = index_dict
        self.person_ids = list(self.index_dict.keys())

    def __len__(self):
        return self.num_iterators

    def __iter__(self):
        for _ in range(self.num_iterators):
            idx_selected_pids = np.random.choice(len(self.person_ids), size=self.num_pids_per_batch, replace=True)
            batch = []
            for idx in idx_selected_pids:
                idxs = np.random.choice(self.index_dict[self.person_ids[idx]], size=self.num_instances, replace=True)
                batch.extend(idxs)
            yield batch

    def split(self, rate=0.5, num_iterators_val=1):
        left_index_dict = defaultdict(list)
        right_index_dict = defaultdict(list)
        for person_id in self.index_dict.keys():
            left_count = int(len(self.index_dict[person_id]) * rate)
            left_index = self.index_dict[person_id][0:left_count]
            right_index = self.index_dict[person_id][left_count:]
            left_index_dict[person_id].extend(left_index)
            right_index_dict[person_id].extend(right_index)
        return RandomBalanceBatchSampler(self.datasource, self.batch_size, self.num_instances, self.num_iterators, left_index_dict), RandomBalanceBatchSampler(self.datasource, self.batch_size, self.num_instances, num_iterators_val, right_index_dict)


r""" build_sampler function
"""
__samplers__ = {
    'RandomBalanceBatchSamplerAttribute': RandomBalanceBatchSamplerAttribute,
    'RandomBatchSamplerAttribute': RandomBatchSamplerAttribute
}

def build_sampler(name, config, phase, datasource, weight, attribute_name, **kwargs):
    dict_params = dict()
    if name == 'RandomBalanceBatchSamplerAttribute':
        dict_params.update({
            phase+'_num_attribute': config[phase]['num_attribute'],
            phase+'_num_positive': config[phase]['num_positive'],
            phase+'_num_negative': config[phase]['num_negative'],
            phase+'_num_iterator': config[phase]['num_iterator']
        })
        return RandomBalanceBatchSamplerAttribute(
            datasource=datasource,
            attribute_name=attribute_name,
            num_attribute=config[phase]['num_attribute'],
            num_positive=config[phase]['num_positive'],
            num_negative=config[phase]['num_negative'],
            num_iterator=config[phase]['num_iterator']
        ), dict_params

    elif name == 'RandomBatchSamplerAttribute':
        dict_params.update({
            phase+'_num_attribute': config[phase]['num_attribute'],
            phase+'_num_positive': config[phase]['num_positive'],
            phase+'_num_negative': config[phase]['num_negative'],
            phase+'_num_iterator': config[phase]['num_iterator']
        })
        return RandomBatchSamplerAttribute(
            datasource=datasource,
            weight=weight,
            attribute_name=attribute_name,
            num_attribute=config[phase]['num_attribute'],
            num_positive=config[phase]['num_positive'],
            num_negative=config[phase]['num_negative'],
            num_iterator=config[phase]['num_iterator']
        ), dict_params
    
    elif name == 'RandomBatchSamplerAttribute1':
        dict_params.update({
            phase+'_num_attribute': config[phase]['num_attribute'],
            phase+'_num_positive': config[phase]['num_positive'],
            phase+'_num_negative': config[phase]['num_negative'],
            phase+'_num_iterator': config[phase]['num_iterator']
        })
        return RandomBatchSamplerAttribute1(
            datasource=datasource,
            weight=weight,
            attribute_name=attribute_name,
            num_attribute=config[phase]['num_attribute'],
            num_positive=config[phase]['num_positive'],
            num_negative=config[phase]['num_negative'],
            num_iterator=config[phase]['num_iterator']
        ), dict_params

    elif name == 'RandomBatchSamplerAttributeWeight':
        dict_params.update({
            phase+'_num_attribute': config[phase]['num_attribute'],
            phase+'_num_sampler': config[phase]['num_sampler'],
            phase+'_num_iterator': config[phase]['num_iterator']
        })
        return RandomBatchSamplerAttributeWeight(
            datasource=datasource,
            weight=weight,
            attribute_name=attribute_name,
            num_attribute=config[phase]['num_attribute'],
            num_sampler=config[phase]['num_sampler'],
            num_iterator=config[phase]['num_iterator']
        ), dict_params
    
    elif name == 'RandomBatchSamplerAttributeWeight1':
        dict_params.update({
            phase+'_num_attribute': config[phase]['num_attribute'],
            phase+'_num_sampler': config[phase]['num_sampler'],
            phase+'_num_iterator': config[phase]['num_iterator']
        })
        return RandomBatchSamplerAttributeWeight1(
            datasource=datasource,
            weight=weight,
            attribute_name=attribute_name,
            num_attribute=config[phase]['num_attribute'],
            num_sampler=config[phase]['num_sampler'],
            num_iterator=config[phase]['num_iterator']
        ), dict_params
        
    else:
        raise KeyError()


