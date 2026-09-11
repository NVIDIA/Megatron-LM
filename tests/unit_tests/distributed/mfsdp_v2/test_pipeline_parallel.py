# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline-parallel context-sharing test for MFSDP v2.

``virtual_pipeline_model_parallel_size > 1`` manifests as multiple model chunks
on a rank, and ``wrap_model_chunks_with_ddp`` wraps each chunk in its own
``FullyShardedDataParallelV2``. For MFSDP v2 the helper opens one ambient
``fully_shard_context`` around the whole per-chunk loop; each chunk's adapter
joins it through ``current_fully_shard_context()`` instead of opening a second
context, and the helper finalizes the shared context exactly once after the
loop. This test drives the helper directly and asserts that all wrapped chunks
share the same ``FsdpContext``.
"""

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.fully_shard import (
    current_fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training.training import wrap_model_chunks_with_ddp
from tests.unit_tests.test_utilities import Utils


class _VirtualPipelineChunk(nn.Module):
    """One virtual-pipeline stage: a tiny dense block with its own parameters."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _mfsdp_v2_ddp_config() -> DistributedDataParallelConfig:
    """A DistributedDataParallelConfig accepted by MFSDP v2 validation."""
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        data_parallel_sharding_strategy="optim_grads_params",
    )


def _compute_config() -> TransformerConfig:
    """A minimal TransformerConfig accepted by MFSDP v2 validation."""
    return TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1)


class TestMfsdpV2PipelineSharedContext:
    """MFSDP v2 virtual-pipeline chunks share a single FSDP context."""

    def setup_method(self):
        """Initialize the single-stage process groups the wrapping call needs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self):
        """Destroy the process groups initialized in ``setup_method``."""
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices."
    )
    def test_shared_context_across_virtual_pipeline_chunks(self, distributed_setup):
        """Multiple virtual-pipeline chunks wrapped in one call share one FsdpContext."""
        device = distributed_setup.device

        # virtual_pipeline_model_parallel_size > 1 yields multiple chunks on this
        # rank; model two virtual-pipeline sub-stages here. The ambient
        # fully_shard_context is opened for every MFSDP v2 wrap, regardless of the
        # chunk count.
        chunks = [
            _VirtualPipelineChunk().to(device=device, dtype=torch.bfloat16) for _ in range(2)
        ]

        wrapped_chunks = wrap_model_chunks_with_ddp(
            chunks,
            _compute_config(),
            _mfsdp_v2_ddp_config(),
            DP=FullyShardedDataParallel,
            pg_collection=self.pg_collection,
            disable_bucketing_per_chunk=[False] * len(chunks),
        )

        # Every chunk is an MFSDP v2 wrapper whose root is an FsdpModule.
        assert len(wrapped_chunks) == len(chunks)
        for wrapped_chunk in wrapped_chunks:
            assert isinstance(wrapped_chunk, FullyShardedDataParallelV2)
            assert isinstance(wrapped_chunk.module, FsdpModule)

        # The whole point of the feature: all chunks share ONE context, so each chunk
        # must report the same context object as the first one.
        first_context = wrapped_chunks[0].module.context
        for wrapped_chunk in wrapped_chunks:
            assert wrapped_chunk.module.context is first_context

        # The shared context was finalized exactly once after the loop, so it is
        # ready to drive forward/backward without an extra finalize.
        first_context.ensure_finalized()

        # The helper left no ambient context behind: the chunks joined its scope and
        # that scope, which owned the only finalize call, has since exited.
        assert current_fully_shard_context() is None
