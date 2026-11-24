# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay TE batched projection with its canonical matrix gradient owner."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.te_batched_linear import TEBatchedLinear
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridAttention,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


@pytest.mark.skipif(
    TEBatchedLinear is None, reason="Transformer Engine BatchedLinear is unavailable"
)
@pytest.mark.parametrize("mode", ["bf16", "mxfp8", "primary_mxfp8"])
@pytest.mark.parametrize("fused", [False, True])
def test_batched_linear_gradient_owner_replays(mode, fused):
    """Replay outputs/dgrad/wgrad, including fresh buffers and two accumulated microbatches."""
    import transformer_engine.common.recipe as recipes
    import transformer_engine.pytorch as te

    recipe = None
    if mode != "bf16":
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)
        recipe = recipes.MXFP8BlockScaling(fp8_format=recipes.Format.E4M3)
    torch.manual_seed(7971)
    init = (
        te.quantized_model_init(enabled=True, recipe=recipe)
        if mode == "primary_mxfp8"
        else nullcontext()
    )
    with init:
        module = TEBatchedLinear(
            2,
            64,
            64,
            input_batch_dim=-2,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
            accumulate_into_main_grad=fused,
            save_original_input=True,
        )
    weight = module.weight
    attention = SimpleNamespace(_uses_te_batched_linear=True, linear_o_group_proj=module)
    inputs = torch.randn(32, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(value):
        weight.grad = None
        if fused:
            # Fresh allocation catches a cached main_grad view surviving a buffer reset.
            weight.main_grad = torch.full(
                weight.shape, float("nan"), device="cuda", dtype=torch.float32
            )
            weight.grad_added_to_main_grad = False
            weight.zero_out_wgrad = True
        outputs = []
        for microbatch, offset in enumerate((0.0, 0.125)):
            if fused:
                weight.overwrite_main_grad = microbatch == 0
            with te.autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext():
                output = DSv4HybridAttention._apply_linear_o_group_proj(attention, value + offset)
            output.float().square().mean().backward()
            outputs.append(output.detach())
        assert module.weight is weight and weight.shape == (128, 64)
        if fused:
            assert weight.grad_added_to_main_grad
        gradient = weight.main_grad if fused else weight.grad
        assert torch.isfinite(gradient).all()
        return outputs, value.grad, gradient

    assert_replays_bit_exact(run, (inputs,), backward=False, contention=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fallback_grouped_projection_replays():
    """Replay the batched-GEMM fallback used without TE BatchedLinear: outputs, dgrad and wgrad."""
    torch.manual_seed(9058)
    groups, in_features, rank = 2, 64, 64
    weight = torch.nn.Parameter(
        torch.randn(groups * rank, in_features, device="cuda", dtype=torch.bfloat16)
        * in_features**-0.5
    )
    attention = SimpleNamespace(
        _uses_te_batched_linear=False,
        _linear_o_group_proj_weight=weight,
        output_projection_local_groups=groups,
    )
    inputs = torch.randn(16, 2, groups, in_features, device="cuda", dtype=torch.bfloat16)
    inputs.requires_grad_(True)

    def run(value):
        weight.grad = None
        outputs = []
        for offset in (0.0, 0.125):
            output = DSv4HybridAttention._apply_linear_o_group_proj(attention, value + offset)
            output.float().square().mean().backward()
            outputs.append(output.detach())
        assert outputs[0].shape == (16, 2, groups, rank)
        assert torch.isfinite(weight.grad).all()
        return outputs, value.grad, weight.grad

    assert_replays_bit_exact(run, (inputs,), backward=False, contention=True)
