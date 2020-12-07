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


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    global_batch_size = args.micro_batch_size * world_size

    # Megatron sampler
    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        global_batch_size=global_batch_size,
        rank=mpu.get_data_parallel_rank(),
        world_size=world_size)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)


class MegatronPretrainingSampler:


    def __init__(self, total_samples, consumed_samples,
                 global_batch_size, rank, world_size):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.global_batch_size = global_batch_size
        self.rank = rank

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.global_batch_size > 0, \
            'Unexpected global batch size: {}'.format(self.global_batch_size)
        assert world_size > 0,\
            'non zero world size is expected: {}'.format(world_size)
        assert self.rank < world_size,\
            'rank should be smaller than world size: {}, {}'.format(
                self.rank, world_size)

        # Batch size per rank.
        assert self.global_batch_size % world_size == 0,\
            'global batch size must be divisible by world size: {}, {}'.format(
                self.global_batch_size, world_size)
        self.batch_size_per_rank = self.global_batch_size // world_size


    def __len__(self):
        return self.total_samples


    def __iter__(self):
        batch = []
        # Last batch if not complete will be dropped.
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.global_batch_size:
                start_idx = self.rank * self.batch_size_per_rank
                end_idx = start_idx + self.batch_size_per_rank
                yield batch[start_idx:end_idx]
                batch = []
