# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Blendable dataset."""

import time

import numpy as np
import torch

from megatron import print_rank_0

class BlendableDataset(torch.utils.data.Dataset):


    def __init__(self, datasets, weights):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        from megatron.data import helpers
        helpers.build_blending_indices(self.dataset_index,
                                       self.dataset_sample_index,
                                       weights, num_datasets, self.size,
                                       torch.distributed.get_rank() == 0)
        print_rank_0('> elapsed time for building blendable dataset indices: '
                     '{:.2f} (sec)'.format(time.time() - start_time))


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx]
