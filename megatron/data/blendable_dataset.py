# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Blendable dataset."""

import hashlib
import os
import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron.core import mpu

class BlendableDataset(torch.utils.data.Dataset):


    def __init__(self, datasets, weights, size, *,
                 data_cache_path=None):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indicies.
        def _build_indices():
            start_time = time.time()
            assert num_datasets < 32767
            # Dataset index is a 16-bit integer to alow at least 2^15 datasets.
            # PyTorch isn't happy casting numpy uint16 to a Torch Tensor,
            # so we use int16 although a dataset_index can never be negative.
            dataset_index = np.zeros(self.size, dtype=np.int16)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)

            from megatron.data import helpers
            helpers.build_blending_indices(dataset_index, dataset_sample_index,
                                           weights, num_datasets, self.size,
                                           torch.distributed.get_rank() == 0)
            print_rank_0('> elapsed time for building blendable dataset indices: '
                         '{:.2f} (sec)'.format(time.time() - start_time))
            return dataset_index, dataset_sample_index

        desc = "Blendable dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {weights}\n"
        desc += f"Size: {size}\n"
        self.desc = desc

        if data_cache_path:
            desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
            desc_path = os.path.join(data_cache_path, desc_hash + ".dsc")
            index_path = os.path.join(data_cache_path, desc_hash + "_index.npy")
            sample_index_path = os.path.join(data_cache_path, desc_hash + "_sample_index.npy")
            cache_hit = os.path.isfile(index_path) and os.path.isfile(sample_index_path)
            cache_success = True
            if torch.distributed.get_rank() == 0 and not cache_hit:
                print(' > WARNING: could not find index map files for blendable'
                      ' dataset, building indices on rank 0 ...', flush=True)
                dataset_index, dataset_sample_index = _build_indices()
                try:
                    os.makedirs(os.path.dirname(index_path), exist_ok=True)
                    with open(desc_path, 'wt') as fd:
                        fd.write(desc)
                        np.save(index_path, dataset_index, allow_pickle=True)
                        np.save(sample_index_path, dataset_sample_index,
                                allow_pickle=True)
                except OSError:
                    print(f'There was an error trying to create the data cache directory ({data_cache_path})')
                    print('or a file in it. This is set with the --data-cache-path argument. Please')
                    print('ensure you have write access to this directory or specify one that you do have')
                    print('write access to.')
                    cache_success = False


            counts = torch.cuda.LongTensor([cache_success])
            torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
            torch.distributed.all_reduce(counts, group=mpu.get_context_parallel_group())
            torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
            if counts[0].item() != (
                torch.distributed.get_world_size() //
                torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())):
                print_rank_0("Data index creation unsuccessful, exiting.")
                exit()

            # Load on all ranks.
            print_rank_0(f'> loading blendable dataset index: {index_path}')
            self.dataset_index = np.load(index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_index.size == self.size

            print_rank_0(f'> loading blendable dataset sample index: {sample_index_path}')
            self.dataset_sample_index = np.load(sample_index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_sample_index.size == self.size
        else:
            self.dataset_index, self.dataset_sample_index = _build_indices()


        # Check size
        _ = self.__getitem__(self.size - 1)
        try:
            _ = self.__getitem__(self.size)
            raise RuntimeError('BlendedDataset size is improperly bounded')
        except IndexError:
            pass
        print_rank_0('> size of blendable dataset: '
                     '{} samples'.format(self.size))


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return {
            "dataset_idx" : dataset_idx,
            **self.datasets[dataset_idx][sample_idx],
        }
