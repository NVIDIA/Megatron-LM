# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataloaders."""


import torch

from megatron import get_args
from megatron import mpu


def build_pretraining_data_loader(dataset, consumed_samples, random_sample=False):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size(),
        random_sample=random_sample)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)


class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, random_sample=False):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.random_sample = random_sample

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        #assert self.consumed_samples < self.total_samples, \
        #    'no samples left to consume: {}, {}'.format(self.consumed_samples,
        #                                                self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        self.epoch = self.consumed_samples // self.total_samples
        current_epoch_samples = self.consumed_samples % self.total_samples
        if self.random_sample:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(self.total_samples, generator=g).tolist()
            idx_range = idx_range_total[current_epoch_samples:]
        else:
            idx_range = range(current_epoch_samples, self.total_samples)

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                self.consumed_samples += len(batch)
                start_idx = self.data_parallel_rank * self.micro_batch_size
                end_idx = start_idx + self.micro_batch_size
                yield batch[start_idx:end_idx]
                batch = []
        self.consumed_samples += len(batch)
