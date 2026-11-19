# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical and wrapper-contract tests for FP32 bias gradient accumulation."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.ops.linear.bias_grad import accumulate_bias_grad
from megatron.core.tensor_parallel.layers import LinearWithGradAccumulationAndAsyncCommunication
from megatron.core.transformer import TransformerConfig


def _inputs():
    x = torch.randn(4, 33, 8, device="cuda", dtype=torch.bfloat16)
    dy = torch.ones_like(x)
    dy[:, 0] = 1 / 256
    weight = torch.nn.Parameter(torch.randn(8, 8, device="cuda", dtype=torch.bfloat16))
    bias = torch.nn.Parameter(torch.zeros(8, device="cuda", dtype=torch.bfloat16))
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    return x, dy, weight, bias


@pytest.mark.parametrize("layout", ["dense", "strided", "empty"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_fused_bias_reduction_layouts(layout, overwrite):
    dy = torch.ones(4, 33, 8, device="cuda", dtype=torch.bfloat16)
    dy[:, 0] = 1 / 256
    if layout == "strided":
        dy = dy.transpose(0, 1)[::2, :, ::2]
    elif layout == "empty":
        dy = dy[:0]
    main_grad = torch.full((dy.shape[-1],), 0.25, device="cuda", dtype=torch.float32)
    for _ in range(2):
        accumulate_bias_grad(dy, main_grad, overwrite)
    expected = dy.double().sum((0, 1))
    if not overwrite:
        expected = 0.25 + 2 * expected
    torch.testing.assert_close(main_grad.double(), expected, rtol=0, atol=0)


def _linear(x, weight, bias, fusion=True, deferred=None):
    return LinearWithGradAccumulationAndAsyncCommunication.apply(
        x, weight, bias, fusion, False, False, deferred, 0, None, 1, None
    )


class _BiasLinear(torch.nn.Module):
    def __init__(self, bias_use, *, defer_wgrad):
        super().__init__()
        _, _, self.weight, self.bias = _inputs()
        self.bias_use = bias_use
        self.defer_wgrad = defer_wgrad
        if bias_use == "fused":
            self.second_weight = torch.nn.Parameter(self.weight.detach().clone())

    def forward(self, x):
        output = _linear(x, self.weight, self.bias, deferred=[] if self.defer_wgrad else None)
        if self.bias_use == "ordinary":
            return output + 2 * self.bias
        if self.bias_use == "fused":
            second_output = _linear(
                x, self.second_weight, self.bias, deferred=[] if self.defer_wgrad else None
            )
            return output + 2 * second_output
        return output


def _expected_bias_grad(dy, bias_use):
    expected = dy.float().sum((0, 1))
    if bias_use == "ordinary":
        expected += (2 * dy).sum((0, 1)).float()
    elif bias_use == "fused":
        expected *= 3
    return expected


@pytest.mark.parametrize("split", [None, 0, 1])
def test_fp32_bias_accumulation(split):
    x, dy, weight, bias = _inputs()
    bias.main_grad = torch.full_like(bias, 0.25, dtype=torch.float32)
    bias.grad_added_to_main_grad = False
    pairs = [(x, dy)] if split is None else zip(x.chunk(2, split), dy.chunk(2, split))
    for xx, dd in pairs:
        _linear(xx, weight, bias, deferred=[]).backward(dd)
        assert not bias.grad_added_to_main_grad
        assert torch.count_nonzero(bias.grad) == 0
        bias.grad = None
    expected = dy.double().sum((0, 1)) + 0.25
    torch.testing.assert_close(bias.main_grad.double(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["no_buffer", "no_protocol", "bf16_buffer", "fusion_off"])
def test_bias_ordinary_autograd_fallback(reason):
    x, dy, weight, bias = _inputs()
    if reason != "no_buffer":
        bias.main_grad = torch.zeros_like(
            bias, dtype=torch.bfloat16 if reason == "bf16_buffer" else torch.float32
        )
    if reason != "no_protocol":
        bias.grad_added_to_main_grad = False
    fusion = reason != "fusion_off"
    _linear(x, weight, bias, fusion=fusion, deferred=[] if fusion else None).backward(dy)
    torch.testing.assert_close(bias.grad, dy.sum((0, 1)), rtol=0, atol=0)
    if hasattr(bias, "main_grad"):
        assert torch.count_nonzero(bias.main_grad) == 0


def test_bias_accumulation_with_deferred_weight_gradient():
    x, dy, weight, bias = _inputs()
    bias.main_grad = torch.zeros_like(bias, dtype=torch.float32)
    bias.grad_added_to_main_grad = False
    deferred = []
    _linear(x, weight, bias, deferred=deferred).backward(dy)
    assert len(deferred) == 1
    assert torch.count_nonzero(weight.main_grad) == 0
    torch.testing.assert_close(bias.main_grad.double(), dy.double().sum((0, 1)), rtol=0, atol=0)


def test_frozen_bias_does_not_acquire_fsdp_gradient_buffer():
    x, dy, weight, bias = _inputs()
    bias.requires_grad_(False)
    bias.__fsdp_param__ = True
    bias.grad_added_to_main_grad = False

    def forbidden():
        raise AssertionError("Frozen bias must not allocate a gradient bucket")

    bias.get_main_grad = forbidden
    _linear(x, weight, bias, deferred=[]).backward(dy)
    assert bias.grad is None


def test_fsdp_bias_buffer_is_acquired_in_backward_and_reacquired():
    x, dy, weight, bias = _inputs()
    bias.__fsdp_param__ = True
    bias.grad_added_to_main_grad = False
    buffers = []

    def get_main_grad():
        buffer = torch.zeros_like(bias, dtype=torch.float32)
        buffers.append(buffer)
        return buffer

    bias.get_main_grad = get_main_grad
    for _ in range(2):
        count = len(buffers)
        out = _linear(x, weight, bias, deferred=[])
        assert len(buffers) == count  # No forward-time gradient bucket allocation.
        out.backward(dy)
        assert len(buffers) == count + 1
        assert bias.bias_grad_added_to_main_grad
        assert not bias.grad_added_to_main_grad
        torch.testing.assert_close(buffers[-1].double(), dy.double().sum((0, 1)), rtol=0, atol=0)
        bias.grad = None
        bias.bias_grad_added_to_main_grad = False  # FSDP consumes this in its hook.


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("zero_out_wgrad", [False, True])
@pytest.mark.parametrize("bias_use", ["once", "ordinary", "fused"])
def test_ddp_bias_accumulation(overlap, zero_out_wgrad, bias_use):
    if not dist.is_initialized():
        pytest.skip("Run with torchrun and the distributed test fixture")
    parallel_state.initialize_model_parallel()
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            bf16=True,
            calculate_per_token_loss=True,
        )
        model = _BiasLinear(bias_use, defer_wgrad=True)
        wrapped = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=overlap, average_in_collective=False
            ),
            model,
        )
        model.bias.zero_out_wgrad = zero_out_wgrad
        x, dy, _, _ = _inputs()
        dy *= dist.get_rank() + 1
        expected = _expected_bias_grad(dy, bias_use) * 2
        dist.all_reduce(expected)
        # Two optimization cycles check buffer/flag reset; two microbatches
        # exercise no_sync and final hook registration with real collectives.
        for _ in range(2):
            wrapped.zero_grad_buffer()
            for microbatch in range(2):
                with wrapped.no_sync() if microbatch == 0 else nullcontext():
                    wrapped(x).backward(dy)
            wrapped.finish_grad_sync()
            torch.testing.assert_close(model.bias.main_grad, expected, rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("strategy", ["no_shard", "optim_grads_params"])
@pytest.mark.parametrize("bias_use", ["once", "ordinary", "fused"])
def test_megatron_fsdp_bias_accumulation(strategy, bias_use):
    from megatron.core.distributed.fsdp.mcore_adapter_v1 import FullyShardedDataParallelV1
    from megatron.core.utils import is_te_min_version

    if not is_te_min_version("2.10"):
        pytest.skip("Megatron FSDP v1 fused accumulation requires Transformer Engine >= 2.10")
    if not dist.is_initialized():
        pytest.skip("Run under the distributed test runner")
    parallel_state.initialize_model_parallel()
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            bf16=True,
            calculate_per_token_loss=True,
            gradient_accumulation_fusion=True,
        )
        model = _BiasLinear(bias_use, defer_wgrad=False)
        wrapped = FullyShardedDataParallelV1(
            config,
            DistributedDataParallelConfig(
                data_parallel_sharding_strategy=strategy,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=False,
                megatron_fsdp_main_grads_dtype=torch.float32,
                megatron_fsdp_grad_comm_dtype=torch.float32,
            ),
            model,
            fsdp_unit_modules=[_BiasLinear],
        )
        x, dy, _, _ = _inputs()
        dy *= dist.get_rank() + 1
        expected = _expected_bias_grad(dy, bias_use)
        dist.all_reduce(expected)
        for _ in range(2):
            wrapped.zero_grad_buffer()
            for microbatch in range(2):
                with wrapped.no_sync() if microbatch == 0 else nullcontext():
                    wrapped(x).backward(dy)
            wrapped.finish_grad_sync()
            grad = model.bias.grad
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            # FSDP optimizer shards follow flattened bucket boundaries, not
            # necessarily DTensor's evenly chunked per-parameter layout. Check
            # each owned entry directly; this fixture has identical channels.
            # A rank that owns no entries of this parameter has grad=None.
            # Aggregate checks before asserting so all ranks take the same
            # path into the next iteration (or fail together).
            count = 0 if grad is None else grad.numel()
            mismatches = 0 if grad is None else int((grad != 2 * expected[0]).sum())
            coverage = torch.tensor([count, mismatches], device=x.device, dtype=torch.int64)
            dist.all_reduce(coverage)
            expected_count = expected.numel()
            if strategy == "no_shard":
                expected_count *= dist.get_world_size()
            assert coverage.tolist() == [expected_count, 0], (
                coverage.tolist(),
                dist.get_rank(),
                grad,
                2 * expected,
            )
    finally:
        parallel_state.destroy_model_parallel()
