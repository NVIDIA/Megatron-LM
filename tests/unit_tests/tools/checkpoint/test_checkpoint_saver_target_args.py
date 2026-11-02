# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))

from saver_base import MegatronCheckpointSaverBase


def test_target_derived_weight_shard_args_are_not_loaded_from_source_checkpoint():
    source_args = SimpleNamespace(
        tensor_parallel_num_weight_shards=1,
        gtp_weight_remat_size=1,
        expert_tensor_parallel_num_weight_shards=1,
        expert_gtp_weight_remat_size=1,
        num_layers=24,
    )
    target_args = SimpleNamespace(
        tensor_parallel_num_weight_shards=4,
        gtp_weight_remat_size=2,
        expert_tensor_parallel_num_weight_shards=8,
        expert_gtp_weight_remat_size=2,
        num_layers=12,
    )
    saver = MegatronCheckpointSaverBase(args=None, queue=None)
    saver.md = SimpleNamespace(checkpoint_args=source_args)

    result = saver._load_checkpoint_args(target_args)

    assert result.tensor_parallel_num_weight_shards == 4
    assert result.gtp_weight_remat_size == 2
    assert result.expert_tensor_parallel_num_weight_shards == 8
    assert result.expert_gtp_weight_remat_size == 2
    assert result.num_layers == 24
