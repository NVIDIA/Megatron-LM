# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import argparse
import os
import random
from megatron.core.device_utils import get_distributed_backend, get_local_device_count
from megatron.core.device_utils import get_distributed_init_method
import numpy
import torch

import mpu


class IdentityLayer(torch.nn.Module):
    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    mpu.model_parallel_cuda_manual_seed(seed)


def initialize_distributed(backend=None):
    """Initialize torch.distributed."""
    # Get local rank in case it is provided.
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=None,
                        help='local rank passed from distributed launcher')
    args = parser.parse_args()
    local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", get_local_device_count()))

    print('> initializing torch.distributed with local rank: {}, '
          'rank: {}, world size: {}'.format(local_rank, rank, world_size))

    # Call the init process.
    init_method = get_distributed_init_method()
    backend = backend if backend is not None else get_distributed_backend()
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method)


def print_separator(message):
    torch.distributed.barrier()
    filler_len = (78 - len(message)) // 2
    filler = '-' * filler_len
    string = '\n' + filler + ' {} '.format(message) + filler
    if torch.distributed.get_rank() == 0:
        print(string, flush=True)
    torch.distributed.barrier()
