# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measure BF16 bias-gradient rounding at microbatch and token-shard boundaries."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu, bias_gelu_back, bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer import TransformerConfig


def _fused_bias_dropout_add(x, bias):
    # Keep the upstream gradient identical across partitions.
    bias_dropout_add = get_bias_dropout_add(training=True, fused=True)
    return bias_dropout_add((x, bias), torch.zeros_like(x), 0.0)


def _accumulate_bias_grad(fused_op, grad_output_chunks, expected_input_grad_scale):
    first_chunk = grad_output_chunks[0]
    bias = torch.zeros(
        first_chunk.shape[-1],
        dtype=first_chunk.dtype,
        device=first_chunk.device,
        requires_grad=True,
    )
    main_grad = torch.zeros_like(bias, dtype=torch.float32)

    for grad_output in grad_output_chunks:
        x = torch.zeros_like(grad_output, requires_grad=True)
        fused_op(x, bias).backward(grad_output)
        assert bias.grad.dtype == torch.bfloat16
        torch.testing.assert_close(x.grad, grad_output * expected_input_grad_scale, rtol=0, atol=0)
        # Model the wrapper boundary: consume each BF16 leaf gradient in FP32.
        main_grad.add_(bias.grad)
        bias.grad = None

    return main_grad.double()


@pytest.mark.parametrize("fusion", ["bias_dropout_fusion", "bias_activation_fusion"])
@pytest.mark.parametrize("pattern", ["split_loses", "whole_loses"])
def test_bf16_fused_bias_grad_partition_precision(fusion, pattern):
    # [sequence, batch, hidden]. Entries make rounding ties exactly representable.
    dy = torch.ones((4, 4, 8), device="cuda", dtype=torch.bfloat16)
    if pattern == "split_loses":
        dy[2:, 2:] = -1
        dy[0, 0] += 1 / 32
        dy[2, 2] += 1 / 32
        expected_errors = {"whole": 0.0, "batch": 1 / 32, "token": 1 / 32}
    else:
        dy[0, 0] += 1 / 16
        expected_errors = {"whole": 1 / 16, "batch": 0.0, "token": 0.0}

    if fusion == "bias_dropout_fusion":
        fused_op = _fused_bias_dropout_add
        input_grad_scale = 1.0
    else:
        fused_op = bias_gelu_impl
        # At x + bias = 0, the GELU derivative is exactly 1/2.
        input_grad_scale = 0.5

    reference = dy.double().sum(dim=(0, 1)) * input_grad_scale
    partitions = {"whole": (dy,), "batch": dy.chunk(2, dim=1), "token": dy.chunk(2, dim=0)}
    for partition, chunks in partitions.items():
        actual = _accumulate_bias_grad(fused_op, chunks, input_grad_scale)
        expected_error = expected_errors[partition] * input_grad_scale
        torch.testing.assert_close(actual, reference - expected_error, rtol=0, atol=0)
        max_error = (actual - reference).abs().max().item()
        print(
            f"{fusion} {pattern} {partition}: bias[0]={actual[0].item():.8f} "
            f"fp64_ref[0]={reference[0].item():.8f} max_abs_error={max_error:.8f}"
        )


def _random_bias_inputs():
    rng = torch.Generator(device="cuda").manual_seed(20261119)
    shape = (128, 16, 128)  # [token, batch, hidden]
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)
    bias = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16, generator=rng)
    return x, dy, bias


@pytest.mark.parametrize(
    ("name", "fused_op"),
    [
        pytest.param("BDA", _fused_bias_dropout_add, id="bda"),
        pytest.param("GeLU", bias_gelu_impl, id="gelu"),
    ],
)
def test_random_bf16_fused_bias_grad_partition_precision(name, fused_op):
    x, dy, bias = _random_bias_inputs()
    bias.requires_grad_()
    layouts = {
        "whole": [(x, dy)],
        "batch": zip(x.chunk(4, dim=1), dy.chunk(4, dim=1)),
        "token": zip(x.chunk(4, dim=0), dy.chunk(4, dim=0)),
    }

    for layout, chunks in layouts.items():
        actual = torch.zeros_like(bias, dtype=torch.float32)
        rounded = torch.zeros_like(actual)
        fp32 = torch.zeros_like(actual)
        reference = torch.zeros_like(bias, dtype=torch.float64)

        for x_chunk, dy_chunk in chunks:
            x_chunk = x_chunk.detach().requires_grad_()
            fused_op(x_chunk, bias).backward(dy_chunk)
            assert bias.grad.dtype == torch.bfloat16
            actual.add_(bias.grad)
            bias.grad = None

            # The input gradient is also the per-position bias contribution.
            # Compare reductions of that same tensor to isolate bias summation.
            partial = x_chunk.grad.sum(dim=(0, 1), dtype=torch.float32)
            rounded.add_(partial.to(torch.bfloat16))
            fp32.add_(partial)
            reference.add_(x_chunk.grad.sum(dim=(0, 1), dtype=torch.float64))

        torch.testing.assert_close(actual, rounded, rtol=0, atol=0)
        error = (actual - reference).abs()
        fp32_error = (fp32 - reference).abs()
        assert fp32_error.max() < error.max()
        relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)
        print(
            f"{name} {layout}: max_abs={error.max().item():.8g} "
            f"relative_l2={relative_l2.item():.8g} "
            f"fp32_max_abs={fp32_error.max().item():.8g}"
        )


@pytest.mark.parametrize("explicit_sum", [False, True], ids=["autograd_sum", "explicit_sum"])
def test_fp32_gelu_bias_backward_output_is_cast_to_bf16(explicit_sum):
    class _GeLUWithFp32BiasGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, bias):
            ctx.save_for_backward(x, bias)
            return bias_gelu(bias, x)

        @staticmethod
        def backward(ctx, grad_output):
            x, bias = ctx.saved_tensors
            tmp = bias_gelu_back(grad_output, bias, x)
            bias_grad = tmp.float()
            if explicit_sum:
                bias_grad = bias_grad.sum(dim=(0, 1))
            return tmp, bias_grad

    x, dy, bias = _random_bias_inputs()
    x.requires_grad_()
    bias.requires_grad_()
    _GeLUWithFp32BiasGrad.apply(x, bias).backward(dy)
    assert bias.grad.dtype == torch.bfloat16
    fp32 = x.grad.sum(dim=(0, 1), dtype=torch.float32)
    torch.testing.assert_close(bias.grad, fp32.to(torch.bfloat16), rtol=0, atol=0)
    assert torch.count_nonzero(fp32 - bias.grad.float()) > 0


def _run_fused_bias(op, x, bias, enabled):
    if op in {"bda", "bda_fp32_residual"}:
        bda = get_bias_dropout_add(training=True, fused=True, gradient_accumulation_fusion=enabled)
        residual_dtype = torch.float32 if op == "bda_fp32_residual" else x.dtype
        residual = torch.zeros_like(x, dtype=residual_dtype)
        return bda((x, bias), residual, 0.0)
    if op == "weighted_quick_geglu":
        return weighted_bias_quick_geglu_impl(
            x,
            bias,
            torch.ones_like(x[..., :1]).reshape(-1, 1),
            gradient_accumulation_fusion=enabled,
        )
    activation = {"gelu": bias_gelu_impl, "swiglu": bias_swiglu_impl, "geglu": bias_geglu_impl}[op]
    return activation(x, bias, gradient_accumulation_fusion=enabled)


@pytest.mark.parametrize(
    "op", ["bda", "bda_fp32_residual", "gelu", "swiglu", "geglu", "weighted_quick_geglu"]
)
@pytest.mark.parametrize("wrapper", ["ddp", "no_shard", "optim_grads_params"])
def test_fused_bias_wrapper_accumulation(op, wrapper):
    if not dist.is_initialized():
        pytest.skip("Requires the distributed test runner")
    if wrapper != "ddp":
        from megatron.core.distributed.fsdp.mcore_adapter_v1 import FullyShardedDataParallelV1
        from megatron.core.utils import is_te_min_version

        if not is_te_min_version("2.10"):
            pytest.skip("Megatron FSDP v1 fused accumulation requires Transformer Engine >= 2.10")

    gated = op in {"swiglu", "geglu", "weighted_quick_geglu"}
    width = 16 if gated else 8

    class BiasOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(width, device="cuda", dtype=torch.bfloat16))

        def forward(self, x):
            first = _run_fused_bias(op, x, self.bias, True)
            second = _run_fused_bias(op, x, self.bias, True)
            return first + second + self.bias[: first.shape[-1]]

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
        model = BiasOnly()
        if wrapper == "ddp":
            wrapped = DistributedDataParallel(
                config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), model
            )
        else:
            wrapped = FullyShardedDataParallelV1(
                config,
                DistributedDataParallelConfig(
                    data_parallel_sharding_strategy=wrapper,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    megatron_fsdp_main_grads_dtype=torch.float32,
                    megatron_fsdp_grad_comm_dtype=torch.float32,
                ),
                model,
                fsdp_unit_modules=[BiasOnly],
            )

        x = torch.zeros((4, 33, width), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if gated:
            with torch.no_grad():
                x[..., 8:] = 1
        bias = x.new_zeros(width)
        output = _run_fused_bias(op, x, bias, False)
        dy = torch.ones_like(output) * (dist.get_rank() + 1)
        dy[:, 0] *= 1 / 256
        (pointwise,) = torch.autograd.grad(output, x, dy)
        expected = 2 * pointwise.sum((0, 1), dtype=torch.float32)
        # The ordinary use of the same bias still takes the normal BF16 path.
        expected[:8] += dy.to(torch.bfloat16).sum((0, 1)).float()
        dist.all_reduce(expected)
        expected *= 2

        for _ in range(2):
            wrapped.zero_grad_buffer()
            for microbatch in range(2):
                with wrapped.no_sync() if microbatch == 0 else nullcontext():
                    wrapped(x.detach()).backward(dy)
            wrapped.finish_grad_sync()
            grad = model.bias.main_grad if wrapper == "ddp" else model.bias.grad
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            if wrapper == "optim_grads_params":
                # Bias is the only parameter, so its flattened shards concatenate in rank order.
                local_grad = [] if grad is None else grad.tolist()
                by_rank = [None] * dist.get_world_size()
                dist.all_gather_object(by_rank, local_grad)
                grad = expected.new_tensor([value for shard in by_rank for value in shard])
            torch.testing.assert_close(grad, expected, rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("enabled", [False, True])
def test_bda_bias_grad_uses_dropout_mask(enabled):
    x = torch.ones((32, 4, 8), device="cuda", dtype=torch.bfloat16)
    dy = torch.linspace(0.25, 1.0, x.numel(), device="cuda").view_as(x)
    bias = torch.nn.Parameter(torch.zeros_like(x[0, 0]))
    bias.main_grad = torch.zeros_like(bias, dtype=torch.float32)
    bias.grad_added_to_main_grad = False
    bda = get_bias_dropout_add(training=True, fused=True, gradient_accumulation_fusion=enabled)
    output = bda((x, bias), torch.zeros_like(x, dtype=torch.float32), 0.5)
    output.backward(dy)
    expected = (dy * output.detach()).sum((0, 1))
    if enabled:
        torch.testing.assert_close(bias.main_grad, expected, rtol=1e-6, atol=0)
        torch.testing.assert_close(bias.grad, torch.zeros_like(bias), rtol=0, atol=0)
    else:
        torch.testing.assert_close(bias.grad, expected.to(bias.dtype), rtol=0, atol=0)
        torch.testing.assert_close(bias.main_grad, torch.zeros_like(expected), rtol=0, atol=0)
