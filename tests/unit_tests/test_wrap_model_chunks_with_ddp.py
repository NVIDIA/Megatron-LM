# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``wrap_model_chunks_with_ddp`` pg_collection threading.

These cover the training-loop helper directly (not the MIMO models), verifying
that an explicit ``pg_collection`` is forwarded to standard ``DistributedDataParallel``
so a caller without ``parallel_state`` globals can reuse the helper.
"""

import pytest
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
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


class TestWrapModelChunksWithDDP:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_explicit_pg_collection_forwarded_to_ddp(self):
        """An explicit pg_collection is forwarded to standard DDP."""
        from megatron.training.training import wrap_model_chunks_with_ddp

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        wrapped = wrap_model_chunks_with_ddp(
            [_make_chunk()], _config(), ddp_config, pg_collection=pg_collection
        )

        chunk = wrapped[0]
        assert isinstance(chunk, DistributedDataParallel)
        # The DDP was constructed against the explicit pgc's dp_cp group.
        assert chunk.dp_cp_group is pg_collection.dp_cp

    def test_default_path_uses_mpu(self):
        """With pg_collection=None the helper still wraps via mpu globals."""
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
