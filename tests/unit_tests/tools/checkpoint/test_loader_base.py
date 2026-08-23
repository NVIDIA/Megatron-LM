# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for MegatronCheckpointLoaderBase.initialize_megatron_env.

Regression test for a bug where the checkpoint conversion loader never
initialized a default torch.distributed process group, causing
dist_checkpointing.load (via torch.distributed.get_world_size()) to fail
with "Default process group has not been initialized" when loading
torch_dist checkpoints through tools/checkpoint/convert.py.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tools', 'checkpoint')
)

from loader_base import MegatronCheckpointLoaderBase


# These scenarios assume a single-rank (or not-yet-initialized) default
# torch.distributed process group, matching how convert.py actually runs the
# loader (plain `python`, never torchrun). When pytest is launched under this
# repo's multi-rank CI harness (torch.distributed.run --nproc-per-node 8), the
# default PG is already multi-rank before collection, so initialize_megatron_env's
# `if not is_initialized()` guard correctly leaves it untouched -- skip here
# rather than asserting a single-rank world size against it. See the identical
# guard/rationale in test_gpt_hybrid_conversion_parallelism.py.
@pytest.fixture(autouse=True)
def _skip_when_multi_rank_pg():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        pytest.skip(
            "Single-rank process-group init test skipped under a multi-rank "
            "default process group."
        )


class TestInitializeMegatronEnv:
    def _make_loader(self):
        loader = MegatronCheckpointLoaderBase.__new__(MegatronCheckpointLoaderBase)
        loader.build_tokenizer = False
        loader.margs = SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            expert_model_parallel_size=1,
        )
        return loader

    def test_default_process_group_available_after_init(self):
        was_initialized = torch.distributed.is_initialized()
        loader = self._make_loader()

        try:
            with mock.patch('megatron.training.global_vars.set_global_variables'):
                loader.initialize_megatron_env()

            assert torch.distributed.is_initialized()
            # This is the exact call (megatron/core/dist_checkpointing/validation.py,
            # determine_global_metadata) that raised ValueError before the fix.
            assert torch.distributed.get_world_size() == 1
        finally:
            if not was_initialized and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    def test_does_not_reinitialize_existing_process_group(self):
        if not torch.distributed.is_initialized():
            os.environ.setdefault('MASTER_ADDR', 'localhost')
            os.environ.setdefault('MASTER_PORT', '12356')
            torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
            initialized_here = True
        else:
            initialized_here = False

        try:
            loader = self._make_loader()
            with mock.patch('megatron.training.global_vars.set_global_variables'):
                loader.initialize_megatron_env()  # must not raise re-init errors
            assert torch.distributed.get_world_size() == 1
        finally:
            if initialized_here:
                torch.distributed.destroy_process_group()
