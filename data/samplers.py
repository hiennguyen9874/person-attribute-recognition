import torch
import torchvision
import random
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class SubSetSampler(torch.utils.data.Sampler):
    def __init__(self, datasource, batch_size, shuffle = True, index_dict=None):
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
            random.shuffle(self.list_index)
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
            random.shuffle(idx_full)
            left_index = idx_full[validation_count:]
            right_index = idx_full[0:validation_count]
            left_index_dict[person_id].extend(left_index)
            right_index_dict[person_id].extend(right_index)
        return SubSetSampler(self.datasource, self.batch_size, self.shuffle, left_index_dict), SubSetSampler(self.datasource, self.batch_size,  self.shuffle, right_index_dict)

class RandomIdentitySampler(torch.utils.data.Sampler):
    def __init__(self, datasource, batch_size=1, num_instances=1, index_dict=None):
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
            
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idx_dict[person_id].append(batch_idxs)
                    batch_idxs = []
        
        avai_pids = copy.deepcopy(self.person_ids)
        batch = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for person_id in selected_pids:
                batch.extend(batch_idx_dict[person_id].pop(0))
                if len(batch_idx_dict[person_id]) == 0:
                    avai_pids.remove(person_id)
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

class RandomBalanceBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, datasource, batch_size=1, num_instances=1, num_iterators = 1, index_dict=None):
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
            selected_pids = random.sample(self.person_ids, self.num_pids_per_batch)
            batch = []
            for person_id in selected_pids:
                idxs = np.random.choice(self.index_dict[person_id], size=self.num_instances, replace=True)
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
        