# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline-parallel context-sharing and prefetch tests for MFSDP v2.

``virtual_pipeline_model_parallel_size > 1`` manifests as multiple model chunks
on a rank, and ``wrap_model_chunks_with_ddp`` wraps each chunk in its own
``FullyShardedDataParallelV2``. For MFSDP v2 the helper opens one ambient
``fully_shard_context`` around the whole per-chunk loop; each chunk's adapter
joins it through ``current_fully_shard_context()`` instead of opening a second
context, and the helper finalizes the shared context exactly once after the
loop. Tests cover shared context construction and per-root prefetch cursors across
grouped microbatches, backward work, and forward-only validation.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.fully_shard import (
    current_fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.schedule import SchedulePolicy
from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training.training import wrap_model_chunks_with_ddp
from tests.unit_tests.test_utilities import Utils


def test_prefetch_follows_grouped_virtual_pipeline_chunks(monkeypatch):
    """Grouped VPP forwards/backwards and validation use independent root cursors."""
    monkeypatch.setattr(torch.cuda, 'Stream', lambda *_args: Mock())
    monkeypatch.setattr(torch.cuda.nvtx, 'range_push', Mock())
    monkeypatch.setattr(torch.cuda.nvtx, 'range_pop', Mock())
    context = FsdpContext(
        torch.device('cuda'), caller_managed_grad_sync=True, prefetch_per_root=True
    )
    context.current_stream = Mock()

    class HookModule(nn.Module, FsdpModule):
        """Run production FSDP hooks with CPU autograd and mocked communication."""

        def __init__(self, *layers):
            nn.Module.__init__(self)
            self.layers = nn.ModuleList(layers)
            self._context = context
            self._phase = FsdpModule.Phase.RESTING
            self._parameter_groups = ()
            self._unshard_event = None
            self._trainable_parameter_countdown = SimpleNamespace(initial_value=0)
            self._schedule_policy = SchedulePolicy()
            self._unshard_parameter_groups = Mock(
                side_effect=lambda: setattr(self, '_unshard_event', object())
            )
            self.reshard = Mock(side_effect=lambda: setattr(self, '_unshard_event', None))
            self._reduce_gradient_groups = Mock()
            self._register_hooks()
            context.register_module(self)

        def forward(self, inputs):
            for layer in self.layers:
                inputs = layer(inputs)
            return inputs * 2

    chunks = [HookModule(HookModule(), HookModule()) for _ in range(2)]
    context.finalize()
    train_schedule = get_schedule_table(5, 2, 3)
    eval_schedule = get_schedule_table(4, 2, 2)
    assert [chunk_id for _, chunk_id in train_schedule[:6]] == [0, 0, 0, 1, 1, 1]

    def forward_pass(schedule):
        outputs = [[], []]
        for _, chunk_id in schedule:
            chunk = chunks[chunk_id]
            outputs[chunk_id].append(chunk(torch.ones(1, requires_grad=True)))
            assert list(context.forward_order) == [chunk, *chunk.layers]
            assert context.forward_order.next_item() is None
        return outputs

    for _ in range(2):
        for chunk in chunks:
            chunk.train()
        outputs = forward_pass(train_schedule)
        for _, forward_chunk_id in train_schedule:
            # The interleaved scheduler reverses chunk IDs for backward.
            chunk_id = len(chunks) - forward_chunk_id - 1
            outputs[chunk_id].pop(0).sum().backward()
            chunk = chunks[chunk_id]
            assert list(context.backward_order) == [chunk, *reversed(chunk.layers)]
            assert context.backward_order.next_item() is None
        for chunk in chunks:
            chunk.eval()
        with torch.no_grad():
            forward_pass(eval_schedule)


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

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices.")
    def test_shared_context_across_virtual_pipeline_chunks(self, distributed_setup):
        """Multiple virtual-pipeline chunks wrapped in one call share one FsdpContext."""
        device = distributed_setup.device

        # virtual_pipeline_model_parallel_size > 1 yields multiple chunks on this
        # rank; model two virtual-pipeline sub-stages here. The ambient
        # fully_shard_context is opened for every MFSDP v2 wrap, regardless of the
        # chunk count.
        chunks = [_VirtualPipelineChunk().to(device=device, dtype=torch.bfloat16) for _ in range(2)]

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
        for wrapped_chunk in wrapped_chunks:
            first_context.activate_root_prefetch(wrapped_chunk.module)
            assert list(first_context.forward_order) == [
                module
                for module in wrapped_chunk.module.modules()
                if isinstance(module, FsdpModule)
            ]

        # The helper left no ambient context behind: the chunks joined its scope and
        # that scope, which owned the only finalize call, has since exited.
        assert current_fully_shard_context() is None
