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

from collections import deque
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import PrefetchOrder
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionEvent,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.fully_shard import (
    current_fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
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


class TestMfsdpV2PipelineSharedContext:
    """MFSDP v2 virtual-pipeline chunks share a single FSDP context."""

    def setup_method(self):
        """Initialize two pipeline stages for the wrapping call."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self):
        """Destroy the process groups initialized in ``setup_method``."""
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices.")
    @pytest.mark.parametrize("num_chunks", [1, 2], ids=["pp", "vpp"])
    @pytest.mark.parametrize("schedule", ["grouped", "interleaved"])
    def test_pipeline_chunks_record_and_replay_prefetch(
        self, distributed_setup, num_chunks, schedule
    ):
        """PP and VPP preserve prefetch through repeated forward and backward calls."""
        device = distributed_setup.device

        # The ambient context is shared for both ordinary PP and VPP wrapping.
        chunks = [
            _VirtualPipelineChunk().to(device=device, dtype=torch.bfloat16)
            for _ in range(num_chunks)
        ]
        unsharded_weights = {chunk: chunk.fc.weight for chunk in chunks}
        config = TransformerConfig(
            num_layers=4,
            hidden_size=16,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2 if num_chunks == 2 else None,
        )

        wrapped_chunks = wrap_model_chunks_with_ddp(
            chunks,
            config,
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

        assert first_context.prefetch_order is PrefetchOrder.RECORDED
        assert first_context.is_recording_execution_order
        assert list(first_context.execution_order) == []
        for chunk in wrapped_chunks:
            assert chunk.module._schedule_policy.forward_prefetch_size is None
            assert chunk.module._schedule_policy.backward_prefetch_size is None
        # Warmup/cooldown spans multiple autograd backward calls in the same context.
        inputs = torch.ones(2, 16, device=device, dtype=torch.bfloat16)
        forward_chunks = [chunk for chunk in wrapped_chunks for _ in range(3)]
        if schedule == "grouped":
            calls = [(chunk, Phase.FORWARD) for chunk in forward_chunks] + [
                (chunk, Phase.BACKWARD) for chunk in reversed(forward_chunks)
            ]
        else:
            pending_chunks = deque(forward_chunks[:2])
            calls = [(chunk, Phase.FORWARD) for chunk in pending_chunks]
            for chunk in forward_chunks[2:]:
                calls.extend([(pending_chunks.popleft(), Phase.BACKWARD), (chunk, Phase.FORWARD)])
                pending_chunks.append(chunk)
            calls.extend((chunk, Phase.BACKWARD) for chunk in pending_chunks)
        expected = [ExecutionEvent(chunk.module, phase) for chunk, phase in calls]
        cross_phase_prefetch_checks = 0
        for iteration in range(3):
            outputs = {chunk: deque() for chunk in wrapped_chunks}
            for index, (chunk, phase) in enumerate(calls):
                if iteration and index:
                    previous_chunk, previous_phase = calls[index - 1]
                    if previous_phase is not phase and previous_chunk is not chunk:
                        # Check before demand hooks can gather this chunk's weights.
                        weight = unsharded_weights[chunk.module]
                        storage_bytes = weight.untyped_storage().nbytes()
                        assert storage_bytes > 0
                        cross_phase_prefetch_checks += 1
                if phase is Phase.FORWARD:
                    outputs[chunk].append(chunk(inputs))
                else:
                    outputs[chunk].popleft().float().sum().backward()
                    assert first_context.is_recording_execution_order is (iteration == 0)
                    if iteration and index < len(calls) - 1:
                        assert first_context.execution_order.position != 0
            first_context.finish_grad_sync()
            assert not first_context.is_recording_execution_order
            assert first_context.execution_order.position == 0
            assert list(first_context.execution_order) == expected
        if num_chunks == 2 and schedule == "interleaved":
            assert cross_phase_prefetch_checks > 0


@pytest.mark.parametrize(
    "pp, vpp, ep_overlap, expected",
    [
        (1, None, False, PrefetchOrder.REGISTRATION),
        (2, None, False, PrefetchOrder.RECORDED),
        (2, 2, False, PrefetchOrder.RECORDED),
        (1, None, True, PrefetchOrder.RECORDED),
    ],
)
def test_scheduled_execution_requires_recorded_prefetch(pp, vpp, ep_overlap, expected):
    """PP/VPP and EP overlap select recording; ordinary eager execution retains registration."""
    config = SimpleNamespace(
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        overlap_moe_expert_parallel_comm=ep_overlap,
    )
    assert FullyShardedDataParallelV2.prefetch_order_for(config) is expected
