# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch

from megatron.core.transformer.hyper_connection import HyperConnectionModule, _proj_rms_impl
from megatron.core.transformer.transformer_config import TransformerConfig


def _plain_proj_rms(x, weight, eps, eps_inside_sqrt):
    x = x.reshape(x.shape[0] * x.shape[1], -1).to(torch.float32)
    weight = weight.to(torch.float32)
    if eps_inside_sqrt:
        # GLM normalizes the FP32 input before its projection.
        r = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        proj = torch.matmul(x * r, weight.t())
    else:
        proj = torch.matmul(x, weight.t())
        r = (x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5) + eps).reciprocal()
    return proj, r


@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
@pytest.mark.parametrize("used_outputs", ["both", "projection", "norm"])
def test_mhc_projection_recompute_preserves_math_and_saves_activation_dtype(
    eps_inside_sqrt, used_outputs
):
    torch.manual_seed(123)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_hyper_connections=True,
        mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
    )
    module = HyperConnectionModule(config, layer_number=1)
    x = torch.randn(4, 2, 32).bfloat16()
    weight = torch.randn(24, 32) * 0.02
    module.mapping_proj.weight.data.copy_(weight)
    grad_proj = torch.randn(4, 2, 24)
    grad_norm = torch.randn(4, 2, 1)

    def run_plain():
        x_run = x.detach().clone().requires_grad_(True)
        weight_run = weight.detach().clone().requires_grad_(True)
        proj, norm = _plain_proj_rms(x_run, weight_run, module.norm_eps, eps_inside_sqrt)
        proj = proj.view(4, 2, 24)
        norm = norm.view(4, 2, 1)
        loss = 0
        if used_outputs in ("both", "projection"):
            loss = loss + (proj * grad_proj).sum()
        if used_outputs in ("both", "norm"):
            loss = loss + (norm * grad_norm).sum()
        loss.backward()
        return (proj.detach(), norm.detach()), (x_run.grad, weight_run.grad)

    def run_module():
        x_run = x.detach().clone().requires_grad_(True)
        module.mapping_proj.weight.grad = None
        saved = []

        def pack(tensor):
            saved.append((tensor.dtype, tuple(tensor.shape)))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            proj, norm = module._projection_and_get_norm(x_run)
            loss = 0
            if used_outputs in ("both", "projection"):
                loss = loss + (proj * grad_proj).sum()
            if used_outputs in ("both", "norm"):
                loss = loss + (norm * grad_norm).sum()
            loss.backward()
        return (proj.detach(), norm.detach()), (x_run.grad, module.mapping_proj.weight.grad), saved

    (proj_ref, norm_ref), (x_grad_ref, weight_grad_ref) = run_plain()
    (proj, norm), (x_grad, weight_grad), saved = run_module()

    assert torch.equal(proj, proj_ref)
    assert torch.equal(norm, norm_ref)
    assert torch.equal(x_grad, x_grad_ref)
    assert (weight_grad is None) == (weight_grad_ref is None)
    if weight_grad_ref is not None:
        assert torch.equal(weight_grad, weight_grad_ref)

    fp32_input = (torch.float32, (8, 32))
    bf16_input = (torch.bfloat16, (8, 32))
    assert fp32_input not in saved
    assert bf16_input in saved


@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
def test_mhc_projection_stays_fp32_under_autocast(eps_inside_sqrt):
    """The FP32 projection/RMS recipe must not be downcast by an outer autocast scope."""
    torch.manual_seed(456)
    x = torch.randn(8, 32)
    weight = torch.randn(24, 32)
    eps = 1e-6

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual_proj, actual_norm = _proj_rms_impl(x, weight, eps, eps_inside_sqrt)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with torch.autocast(device_type="cpu", enabled=False):
            if eps_inside_sqrt:
                expected_norm = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
                expected_proj = torch.matmul(x.float() * expected_norm, weight.float().t())
            else:
                expected_proj = torch.matmul(x.float(), weight.float().t())
                expected_norm = 1.0 / (x.float().norm(dim=-1, keepdim=True) / math.sqrt(32) + eps)

    assert actual_proj.dtype == torch.float32
    assert actual_norm.dtype == torch.float32
    torch.testing.assert_close(actual_proj, expected_proj)
    torch.testing.assert_close(actual_norm, expected_norm)
