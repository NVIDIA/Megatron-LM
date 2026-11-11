# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Changing GA must not publish an incomplete GTP output-head gradient to DDP."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.layers import (
    linear_with_grad_accumulation_and_async_allreduce as tp_linear,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _torchrun_dist_init,
    reset_gtp_globals,
)


class _Head(torch.nn.Module):
    """A GTP output head with optionally repeated uses."""

    def __init__(self):
        super().__init__()
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            params_dtype=torch.bfloat16,
            tensor_parallel_num_weight_shards=2,
        )
        self.model_type = ModelType.encoder_or_decoder
        self.heads = 1
        self.output_layer = torch.nn.Linear(64, 64, bias=False, device='cuda', dtype=torch.bfloat16)
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        gtp.wrap_module_params_gtp(
            self.output_layer, ['weight'], parallel_state.get_gtp_weight_remat_group()
        )

    def set_input_tensor(self, input_tensor):
        """The non-pipeline schedule has no incoming activation."""

    def forward(self, x):
        """Give each head a distinct gradient."""
        return sum(
            tp_linear(
                x * (head + 1),
                self.output_layer.weight,
                None,
                False,
                False,
                False,
                tp_group=self.tp_group,
                gtp_remat_size=2,
            )
            for head in range(self.heads)
        )


@pytest.fixture
def model(monkeypatch):
    """Use GTP2 and at least two DDP ranks."""
    world_size = torch.distributed.get_world_size()
    if world_size < 4 or world_size % 2:
        pytest.skip('Requires an even number of ranks >= 4')
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(gtp_remat_size=2)
    monkeypatch.setattr(gtp.GTP_CONFIG, 'async_reduction', False)
    monkeypatch.setattr(gtp.GTP_CONFIG, 'weight_prefetch', False)
    head = _Head()
    try:
        yield head
    finally:
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
        parallel_state.destroy_model_parallel()


def _finalize(models, *args, **kwargs):
    """Flush GTP before DDP, as in production finalization."""
    gtp.wait_for_gtp_grad_reduction_on_current_stream()
    for model in models:
        model.finish_grad_sync()


def _forward_step(data_iterator, model):
    """Return the head loss to the standard schedule."""
    return model(next(data_iterator)), lambda value: (value.sum(), {})


@pytest.mark.parametrize('mode', ['schedule', 'explicit', 'fallback'])
@pytest.mark.parametrize('heads, no_sync', [(1, False), (3, True)])
def test_dynamic_ga_ddp_gradient(model, heads, mode, no_sync, monkeypatch):
    """Check exact gradients and one RS across rising/falling GA with overlapping DDP."""
    model.heads = heads
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(overlap_grad_reduce=True, grad_reduce_in_fp32=True),
        model,
    )
    gtp.classify_gtp_remat_chains(ddp)
    weight = model.output_layer.weight
    rs = Mock(wraps=weight._reduce_scatter)
    monkeypatch.setattr(weight, '_reduce_scatter', rs)
    model.config.no_sync_func = ddp.no_sync if no_sync else None
    model.config.finalize_model_grads_func = _finalize
    for microbatches in (2, 4, 1, 3):
        ddp.zero_grad_buffer()
        rs.reset_mock()
        batches = [
            torch.full(
                (2, 1, 64),
                (torch.distributed.get_rank() + 1) * (i + 1),
                device='cuda',
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for i in range(microbatches)
        ]
        if mode == 'schedule':
            forward_backward_no_pipelining(
                forward_step_func=_forward_step,
                data_iterator=iter(batches),
                model=ddp,
                num_microbatches=microbatches,
                seq_length=2,
                micro_batch_size=1,
            )
        else:
            for i, x in enumerate(batches):
                with ddp.no_sync() if no_sync and i < microbatches - 1 else nullcontext():
                    loss = ddp(x).sum() / microbatches
                    if mode == 'explicit' and i == microbatches - 1:
                        gtp.finalize_gtp_forward_counts(ddp)
                    loss.backward()
            before_fence = rs.call_count
            _finalize([ddp])
        # Two tokens times mean rank factor; loss scaling is rounded to BF16.
        expected = (torch.distributed.get_world_size() + 1) * sum(range(1, microbatches + 1))
        expected *= sum(range(1, heads + 1)) * float(
            torch.tensor(1 / microbatches, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            weight.main_grad, torch.full_like(weight.main_grad, expected), rtol=0, atol=1e-5
        )
        assert rs.call_count == 1
        if mode != 'schedule':
            assert before_fence == (1 if mode == 'explicit' else 0)


def test_checkpointed_head_uses_fence(model):
    """An uncounted checkpoint forward must defer reduction until the final fence."""
    gtp.classify_gtp_remat_chains(model)
    weight = model.output_layer.weight
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    x = torch.ones(2, 1, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    loss = model(x).sum() + checkpoint(model, x, use_reentrant=True).sum()
    gtp.finalize_gtp_forward_counts(model)
    loss.backward()
    assert torch.count_nonzero(weight.main_grad) == 0
    gtp.wait_for_gtp_grad_reduction_on_current_stream()
    # Two uses, two tokens, unit input on every rank.
    torch.testing.assert_close(
        weight.main_grad, torch.full_like(weight.main_grad, 4), rtol=0, atol=0
    )
