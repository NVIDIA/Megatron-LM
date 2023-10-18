# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions for Megatron optimizer."""


from megatron.core import mpu


def shard_buffer(buffer):
    """
    Shard buffer into dp_size chunks of equal size.
    """
    context_parallel = mpu.get_context_parallel_world_size() > 1
    data_parallel_world_size = mpu.get_data_parallel_world_size(
        with_context_parallel=context_parallel
    )
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [buffer[(r*shard_size):((r+1)*shard_size)]
                      for r in range(data_parallel_world_size)]
    return sharded_buffer

