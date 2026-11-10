# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``wrap_model_chunks_with_ddp`` pg_collection threading.

These cover the training-loop helper directly (not the MIMO models), verifying
that an explicit ``pg_collection`` built from a ``HyperCommGrid`` (no
``parallel_state`` globals) is forwarded to standard ``DistributedDataParallel``,
and that the default ``pg_collection=None`` path still falls back to MPU groups.
"""

import pytest
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires 8 GPUs")


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc1(x)


def _make_chunk():
    return _TinyModel().bfloat16().cuda()


def _config():
    return TransformerConfig(num_attention_heads=1, num_layers=1)


class TestExplicitPgCollection:
    """Drive the helper with a HyperCommGrid-derived pgc, no parallel_state."""

    def test_explicit_pg_collection_forwarded_to_ddp(self):
        from megatron.training.training import wrap_model_chunks_with_ddp

        Utils.initialize_distributed()  # torch.distributed only, no model-parallel state
        # One pure-DP grid over the 8 ranks; tp/pp/ep are size-1 (no model parallelism).
        grid = HyperCommGrid([1, 1, 1, 8], ["tp", "pp", "ep", "dp"], backend="nccl")
        dp = grid.create_pg("dp")
        pg_collection = ProcessGroupCollection(
            tp=grid.create_pg("tp"),
            pp=grid.create_pg("pp"),
            ep=grid.create_pg("ep"),
            dp=dp,
            dp_cp=dp,
            expt_dp=dp,
        )

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        wrapped = wrap_model_chunks_with_ddp(
            [_make_chunk()], _config(), ddp_config, pg_collection=pg_collection
        )

        chunk = wrapped[0]
        assert isinstance(chunk, DistributedDataParallel)
        # DDP was constructed against the grid-derived dp_cp group, not parallel_state.
        assert chunk.dp_cp_group is dp
        grid.destroy()


class TestDefaultPath:
    """Isolated back-compat check: pg_collection=None falls back to MPU globals."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_path_uses_mpu(self):
        from megatron.training.training import wrap_model_chunks_with_ddp

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        wrapped = wrap_model_chunks_with_ddp([_make_chunk()], _config(), ddp_config)

        chunk = wrapped[0]
        assert isinstance(chunk, DistributedDataParallel)
        assert chunk.dp_cp_group is parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
