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


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility


# Intra-layer model parallel group that the current rank belongs to.
_INTRA_LAYER_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_INTER_LAYER_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and inter-layer) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_MPU_INTRA_LAYER_WORLD_SIZE = None
_MPU_INTER_LAYER_WORLD_SIZE = None
_MPU_INTRA_LAYER_RANK = None
_MPU_INTER_LAYER_RANK = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(intra_layer_model_parallel_size_=1,
                              inter_layer_model_parallel_size_=1):
    """
    Initialize model data parallel groups.

    Arguments:
        intra_layer_model_parallel_size: number of GPUs used to parallelize model intra-layer.
        inter_layer_model_parallel_size: number of GPUs used to parallelize model inter-layer.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model intra-layer, and 4 GPUs to parallelize
    the model inter-layer. The present function will
    create 8 intra-layer model-parallel groups, 4 inter-layer model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 intra-layer model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 inter-layer model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing intra-layer model parallel with size {}'.format(
            intra_layer_model_parallel_size_))
        print('> initializing inter-layer model parallel with size {}'.format(
            inter_layer_model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    intra_layer_model_parallel_size = min(intra_layer_model_parallel_size_, world_size)
    inter_layer_model_parallel_size = min(inter_layer_model_parallel_size_, world_size)
    ensure_divisibility(world_size,
                        intra_layer_model_parallel_size * inter_layer_model_parallel_size)
    data_parallel_size = world_size // (intra_layer_model_parallel_size *
                                        inter_layer_model_parallel_size)

    num_intra_layer_model_parallel_groups = world_size // intra_layer_model_parallel_size
    num_inter_layer_model_parallel_groups = world_size // inter_layer_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(inter_layer_model_parallel_size):
        start_rank = i * num_inter_layer_model_parallel_groups
        end_rank = (i + 1) * num_inter_layer_model_parallel_groups
        for j in range(intra_layer_model_parallel_size):
            ranks = range(start_rank + j, end_rank,
                          intra_layer_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the intra-layer model-parallel groups.
    global _INTRA_LAYER_MODEL_PARALLEL_GROUP
    assert _INTRA_LAYER_MODEL_PARALLEL_GROUP is None, \
        'intra-layer model parallel group is already initialized'
    for i in range(num_intra_layer_model_parallel_groups):
        ranks = range(i * intra_layer_model_parallel_size,
                      (i + 1) * intra_layer_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _INTRA_LAYER_MODEL_PARALLEL_GROUP = group

    # Build the inter-layer model-parallel groups and embedding groups
    # (first and last rank in each inter-layer model-parallel group).
    global _INTER_LAYER_MODEL_PARALLEL_GROUP
    assert _INTER_LAYER_MODEL_PARALLEL_GROUP is None, \
        'inter-layer model parallel group is already initialized'
    global _EMBEDDING_GROUP
    assert _EMBEDDING_GROUP is None, \
        'embedding group is already initialized'
    for i in range(num_inter_layer_model_parallel_groups):
        ranks = range(i, world_size,
                      num_inter_layer_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _INTER_LAYER_MODEL_PARALLEL_GROUP = group
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
        else:
            embedding_ranks = ranks
        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _INTRA_LAYER_MODEL_PARALLEL_GROUP is None or \
        _INTER_LAYER_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_intra_layer_model_parallel_group():
    """Get the intra-layer model parallel group the caller rank belongs to."""
    assert _INTRA_LAYER_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _INTRA_LAYER_MODEL_PARALLEL_GROUP


def get_inter_layer_model_parallel_group():
    """Get the inter-layer model parallel group the caller rank belongs to."""
    assert _INTER_LAYER_MODEL_PARALLEL_GROUP is not None, \
        'inter_layer_model parallel group is not initialized'
    return _INTER_LAYER_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP


def set_intra_layer_model_parallel_world_size(world_size):
    """Set the intra-layer model parallel size"""
    global _MPU_INTRA_LAYER_WORLD_SIZE
    _MPU_INTRA_LAYER_WORLD_SIZE = world_size


def set_inter_layer_model_parallel_world_size(world_size):
    """Set the inter-layer model parallel size"""
    global _MPU_INTER_LAYER_WORLD_SIZE
    _MPU_INTER_LAYER_WORLD_SIZE = world_size


def get_intra_layer_model_parallel_world_size():
    """Return world size for the intra-layer model parallel group."""
    global _MPU_INTRA_LAYER_WORLD_SIZE
    if _MPU_INTRA_LAYER_WORLD_SIZE is not None:
        return _MPU_INTRA_LAYER_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_intra_layer_model_parallel_group())


def get_inter_layer_model_parallel_world_size():
    """Return world size for the inter-layer model parallel group."""
    global _MPU_INTER_LAYER_WORLD_SIZE
    if _MPU_INTER_LAYER_WORLD_SIZE is not None:
        return _MPU_INTER_LAYER_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_inter_layer_model_parallel_group())


def set_intra_layer_model_parallel_rank(rank):
    """Set intra-layer model parallel rank."""
    global _MPU_INTRA_LAYER_RANK
    _MPU_INTRA_LAYER_RANK = rank


def set_inter_layer_model_parallel_rank(rank):
    """Set inter-layer model parallel rank."""
    global _MPU_INTER_LAYER_RANK
    _MPU_INTER_LAYER_RANK = rank


def get_intra_layer_model_parallel_rank():
    """Return my rank for the intra-layer model parallel group."""
    global _MPU_INTRA_LAYER_RANK
    if _MPU_INTRA_LAYER_RANK is not None:
        return _MPU_INTRA_LAYER_RANK
    return torch.distributed.get_rank(group=get_intra_layer_model_parallel_group())


def get_inter_layer_model_parallel_rank():
    """Return my rank for the inter-layer model parallel group."""
    global _MPU_INTER_LAYER_RANK
    if _MPU_INTER_LAYER_RANK is not None:
        return _MPU_INTER_LAYER_RANK
    return torch.distributed.get_rank(group=get_inter_layer_model_parallel_group())


def is_inter_layer_first_stage():
    """Return True if in the first inter-layer model-parallel stage, False otherwise."""
    return get_inter_layer_model_parallel_rank() == 0


def is_inter_layer_last_stage():
    """Return True if in the last inter-layer model-parallel stage, False otherwise."""
    return get_inter_layer_model_parallel_rank() == (
        get_inter_layer_model_parallel_world_size() - 1)


def get_intra_layer_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank
    in the intra-layer model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_intra_layer_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_inter_layer_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank
    in the inter-layer model parallel group."""
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()
    local_world_size = get_inter_layer_model_parallel_world_size()
    return global_rank % (global_world_size // local_world_size)


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _INTRA_LAYER_MODEL_PARALLEL_GROUP
    _INTRA_LAYER_MODEL_PARALLEL_GROUP = None
    global _INTER_LAYER_MODEL_PARALLEL_GROUP
    _INTER_LAYER_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
