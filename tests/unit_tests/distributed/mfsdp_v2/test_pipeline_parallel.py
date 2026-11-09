# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline schedule and context-sharing tests for MFSDP v2.

``virtual_pipeline_model_parallel_size > 1`` manifests as multiple model chunks
on a rank, and ``wrap_model_chunks_with_ddp`` wraps each chunk in its own
``FullyShardedDataParallelV2``. For MFSDP v2 the helper opens one ambient
``fully_shard_context`` around the whole per-chunk loop; each chunk's adapter
joins it through ``current_fully_shard_context()`` instead of opening a second
context, and the helper finalizes the shared context exactly once after the
loop. This test drives the helper directly and asserts that all wrapped chunks
share the same ``FsdpContext``.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.fully_shard import (
    current_fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
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

    @pytest.mark.parametrize(
        "delay_wgrad_compute,num_microbatches",
        [(False, 3), (True, 1)],
        ids=["accumulation", "delayed_wgrad"],
    )
    def test_schedule_waits_after_all_backward_work(
        self, distributed_setup, monkeypatch, delay_wgrad_compute, num_microbatches
    ):
        """A schedule waits once, after delayed gradients, and produces the correct sum."""
        device = distributed_setup.device
        config = _compute_config()
        torch.manual_seed(1234)
        layer = te.Linear(
            16, 16, bias=False, device=device, delay_wgrad_compute=delay_wgrad_compute
        )
        layer.model_type = ModelType.encoder_or_decoder
        layer.set_input_tensor = lambda _: None
        reference = nn.Linear(16, 16, bias=False, device=device)
        reference.load_state_dict({"weight": layer.weight.detach().clone()})
        model = FullyShardedDataParallelV2(
            config=config,
            ddp_config=_mfsdp_v2_ddp_config(),
            module=layer,
            pg_collection=self.pg_collection,
        )
        context = model.module.context
        finish_grad_sync = Mock(wraps=context.finish_grad_sync)
        monkeypatch.setattr(context, "finish_grad_sync", finish_grad_sync)
        config.no_sync_func = model.no_sync

        def finalize_model_grads(models, *args, **kwargs):
            finish_grad_sync.assert_not_called()
            if delay_wgrad_compute:
                layer.backward_dw()
            models[0].finish_grad_sync()

        config.finalize_model_grads_func = finalize_model_grads
        torch.manual_seed(4321 + distributed_setup.rank)
        inputs = [
            torch.randn(4, 16, device=device, requires_grad=True) for _ in range(num_microbatches)
        ]

        def forward_step(data_iterator, model):
            def loss_func(output):
                loss = output.square().mean()
                return loss, {"loss": loss.detach()}

            return model(next(data_iterator)), loss_func

        forward_backward_no_pipelining(
            forward_step_func=forward_step,
            data_iterator=iter(inputs),
            model=[model],
            num_microbatches=len(inputs),
            seq_length=4,
            micro_batch_size=1,
            pg_collection=self.pg_collection,
        )
        finish_grad_sync.assert_called_once_with()
        for x in inputs:
            (reference(x.detach()).square().mean() / len(inputs)).backward()
        dist.all_reduce(reference.weight.grad, op=dist.ReduceOp.AVG)
        torch.testing.assert_close(layer.weight.grad.full_tensor(), reference.weight.grad)

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

        # The helper left no ambient context behind: the chunks joined its scope and
        # that scope, which owned the only finalize call, has since exited.
        assert current_fully_shard_context() is None
