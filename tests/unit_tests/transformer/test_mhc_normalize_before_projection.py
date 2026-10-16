# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""mhc_normalize_before_projection, the opt-in GLM mHC projection order.

HF GLM5Next (huggingface/transformers commit 93c8b7b485963a10800c91f55304db6be211c2bd,
Glm5NextTextHyperConnection) feeds the RMS-normalized residual to its FP32 mapping
projection. PR 7521, the default, projects first and scales the projection by the norm
in _compute_h. The flag is independent of mhc_norm_eps_inside_sqrt, which only selects
the norm formula, so every combination of the two is checked against plain PyTorch.
The compiled functions run eagerly here; this checks arithmetic, not compilation.
"""

import math
from argparse import ArgumentParser
from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.hyper_connection import HyperConnectionModule, _proj_rms_impl
from megatron.core.transformer.transformer_config import TransformerConfig

_DEVICES = pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
_COMBINATIONS = pytest.mark.parametrize(
    "eps_inside_sqrt, normalize_before_projection",
    [(False, False), (True, False), (False, True), (True, True)],
)


def _config(**overrides):
    values = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_keep_mappings_in_fp32=True,
        layernorm_epsilon=1e-5,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _module(device, eps_inside_sqrt, normalize_before_projection):
    torch.manual_seed(31)
    config = _config(
        mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
        mhc_normalize_before_projection=normalize_before_projection,
    )
    module = HyperConnectionModule(config, layer_number=1).to(device)
    with torch.no_grad():
        module.mapping_proj.weight.normal_(std=0.05)
        module.bias.normal_(std=0.1)
        module.alpha_pre.fill_(0.9)
        module.alpha_post.fill_(1.1)
        module.alpha_res.fill_(1.3)
    return module


@pytest.fixture
def ieee_fp32_matmul():
    """Run FP32 GEMMs in IEEE FP32 for one test, then restore the float32 matmul precision.

    NGC PyTorch containers set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, so CUDA FP32 GEMMs
    default to TF32. TF32 rounds the normalized projection input to a 10-bit mantissa,
    which separates the two orders by ~5e-4 instead of FP32 rounding. The module and both
    plain references run under this one setting.
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        assert not torch.backends.cuda.matmul.allow_tf32
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _plain_norm(x, eps, eps_inside_sqrt):
    if eps_inside_sqrt:
        return torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return (x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1]) + eps).reciprocal()


def _plain_proj_rms(x, weight, eps, eps_inside_sqrt, normalize_before_projection):
    x = x.reshape(-1, x.shape[-1]).float()
    weight = weight.float()
    norm = _plain_norm(x, eps, eps_inside_sqrt)
    projection_input = x * norm if normalize_before_projection else x
    return torch.matmul(projection_input, weight.t()), norm


def _plain_compute_h(module, x, eps_inside_sqrt, normalize_before_projection):
    """Mapping logits, H_pre and H_post of one order in plain PyTorch."""
    n = module.n
    proj, norm = _plain_proj_rms(
        x, module.mapping_proj.weight, module.norm_eps, eps_inside_sqrt, normalize_before_projection
    )
    scales = torch.cat(
        [module.alpha_pre.expand(n), module.alpha_post.expand(n), module.alpha_res.expand(n * n)]
    )
    if normalize_before_projection:
        logits = proj * scales + module.bias
    else:
        logits = norm * proj * scales + module.bias
    logits = logits.view(*x.shape[:2], -1)
    return logits, logits[..., :n].sigmoid() + 1e-6, logits[..., n : 2 * n].sigmoid() * 2


def test_flag_defaults_off_and_parses_through_real_cli():
    from megatron.training.arguments import add_megatron_arguments

    assert _config().mhc_normalize_before_projection is False
    parser = add_megatron_arguments(ArgumentParser(allow_abbrev=False))
    assert parser.parse_args([]).mhc_normalize_before_projection is False
    args = parser.parse_args(["--mhc-normalize-before-projection"])
    assert args.mhc_normalize_before_projection is True
    # Orthogonal to the norm formula: one flag does not imply the other.
    assert args.mhc_norm_eps_inside_sqrt is False


@_DEVICES
@_COMBINATIONS
@pytest.mark.parametrize("autocast", [False, True])
def test_projection_helper_matches_plain_reference(
    device, eps_inside_sqrt, normalize_before_projection, autocast
):
    """Both orders stay FP32 under autocast and follow their own plain recipe."""
    torch.manual_seed(7)
    x = torch.randn(16, 128, device=device).bfloat16()
    weight = torch.randn(24, 128, device=device) * 0.05
    eps = 1e-5
    context = (
        torch.autocast(device_type=device, dtype=torch.bfloat16) if autocast else nullcontext()
    )
    with context:
        proj, norm = _proj_rms_impl(
            x,
            weight,
            eps,
            eps_inside_sqrt=eps_inside_sqrt,
            normalize_before_projection=normalize_before_projection,
        )
    expected_proj, expected_norm = _plain_proj_rms(
        x, weight, eps, eps_inside_sqrt, normalize_before_projection
    )
    assert proj.dtype == norm.dtype == torch.float32
    assert torch.equal(norm, expected_norm)
    assert torch.equal(proj, expected_proj)


@_DEVICES
@_COMBINATIONS
@pytest.mark.usefixtures("ieee_fp32_matmul")
def test_compute_h_matches_plain_reference(device, eps_inside_sqrt, normalize_before_projection):
    """_compute_h scales by the norm only when the projection did not consume it.

    Forward values must match bit for bit; gradients reach the input and every parameter.
    """
    module = _module(device, eps_inside_sqrt, normalize_before_projection)
    n = module.n
    x = torch.randn(8, 2, 128, device=device).bfloat16()
    actual_input = x.clone().requires_grad_()
    reference_input = x.clone().requires_grad_()
    logits, pre, post = _plain_compute_h(
        module, reference_input, eps_inside_sqrt, normalize_before_projection
    )
    with torch.no_grad():
        other_logits = _plain_compute_h(module, x, eps_inside_sqrt, not normalize_before_projection)
    # The orders are algebraically equal, so only rounding tells them apart.
    torch.testing.assert_close(other_logits[0], logits, rtol=1e-4, atol=1e-5)
    assert not torch.equal(other_logits[0], logits), "Fixture must distinguish the orders."

    with torch.compiler.set_stance("force_eager"):
        h_pre, h_post, h_res = module._compute_h(*module._projection_and_get_norm(actual_input))
        assert torch.equal(h_pre, pre)
        assert torch.equal(h_post, post)
        assert torch.equal(h_res, logits[..., 2 * n :])
        cotangents = [torch.randn_like(t) for t in (h_pre, h_post, h_res)]
        sum((t * g).sum() for t, g in zip((h_pre, h_post, h_res), cotangents)).backward()
    actual_grads = [actual_input.grad] + [p.grad.clone() for p in module.parameters()]
    module.zero_grad(set_to_none=True)
    expected = (pre, post, logits[..., 2 * n :])
    sum((t * g).sum() for t, g in zip(expected, cotangents)).backward()
    expected_grads = [reference_input.grad] + [p.grad for p in module.parameters()]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


@_DEVICES
@_COMBINATIONS
def test_projection_recompute_is_exact_and_saves_activation_dtype(
    device, eps_inside_sqrt, normalize_before_projection
):
    """The checkpointed projection replays bit for bit and saves only the BF16 input."""
    module = _module(device, eps_inside_sqrt, normalize_before_projection)
    x = torch.randn(8, 2, 128, device=device).bfloat16().requires_grad_()
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, tuple(tensor.shape)))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        proj, norm = module._projection_and_get_norm(x)
    with torch.no_grad():
        replay_proj, replay_norm = module._projection_and_get_norm(x)
    assert torch.equal(replay_proj, proj)
    assert torch.equal(replay_norm, norm)
    assert (torch.float32, (16, 128)) not in saved
    assert (torch.bfloat16, (16, 128)) in saved

    grad_proj, grad_norm = torch.randn_like(proj), torch.randn_like(norm)
    ((proj * grad_proj).sum() + (norm * grad_norm).sum()).backward()
    reference_input = x.detach().clone().requires_grad_()
    reference_weight = module.mapping_proj.weight.detach().clone().requires_grad_()
    expected_proj, expected_norm = _plain_proj_rms(
        reference_input,
        reference_weight,
        module.norm_eps,
        eps_inside_sqrt,
        normalize_before_projection,
    )
    assert torch.equal(proj.view_as(expected_proj), expected_proj)
    assert torch.equal(norm.view_as(expected_norm), expected_norm)
    (
        (expected_proj * grad_proj.view_as(expected_proj)).sum()
        + (expected_norm * grad_norm.view_as(expected_norm)).sum()
    ).backward()
    assert torch.equal(x.grad, reference_input.grad)
    assert torch.equal(module.mapping_proj.weight.grad, reference_weight.grad)


@_DEVICES
@_COMBINATIONS
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mappings_full_recompute_replay_is_exact(
    device, eps_inside_sqrt, normalize_before_projection, dtype
):
    """A reentrant checkpoint of compute_mappings replays every order bit for bit.

    PR 7521's test_mhc_projection_matches_with_and_without_grad, for each combination with
    mhc_normalize_before_projection: the no-grad projection matches the grad-mode one, and
    the checkpointed mappings match the plain call's outputs and all gradients.
    """
    module = _module(device, eps_inside_sqrt, normalize_before_projection)
    x = torch.randn(8, 2, 128, device=device, dtype=dtype, requires_grad=True)
    proj, norm = module._projection_and_get_norm(x)
    with torch.no_grad():
        replay_proj, replay_norm = module._projection_and_get_norm(x)
    assert torch.equal(replay_proj, proj)
    assert torch.equal(replay_norm, norm)

    expected = module.compute_mappings(x)
    sum(t.float().square().sum() for t in expected).backward()
    expected_grads = [x.grad.clone()] + [p.grad.clone() for p in module.parameters()]
    module.zero_grad(set_to_none=True)
    replay_x = x.detach().clone().requires_grad_(True)
    actual = checkpoint(module.compute_mappings, replay_x, use_reentrant=True)
    for a, e in zip(actual, expected):
        assert torch.equal(a, e)
    sum(t.float().square().sum() for t in actual).backward()
    for a, e in zip([replay_x.grad] + [p.grad for p in module.parameters()], expected_grads):
        assert torch.equal(a, e)


def test_fused_mhc_module_rejects_normalize_before_projection():
    """The module guard also covers configs modified after constructor validation."""
    config = _config(use_fused_mhc=True, mhc_keep_mappings_in_fp32=False)
    config.mhc_normalize_before_projection = True
    with pytest.raises(ValueError, match="Fused mHC does not support"):
        HyperConnectionModule(config, layer_number=1)
