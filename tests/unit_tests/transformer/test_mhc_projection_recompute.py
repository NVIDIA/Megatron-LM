# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.hyper_connection import HyperConnectionModule, _proj_rms_impl
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
def test_mhc_projection_matches_with_and_without_grad(device, dtype, eps_inside_sqrt):
    """Full-recompute forward and backward replay must use identical FP32 arithmetic."""
    torch.manual_seed(10)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
        mhc_keep_mappings_in_fp32=True,
    )
    module = HyperConnectionModule(config, layer_number=1).to(device)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor and eager differ on CUDA")
@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
def test_mhc_mappings_replay_matches_when_dynamo_cache_is_full(eps_inside_sqrt):
    """A full-recompute forward must not mix compiled and eager mapping arithmetic.

    Dynamo's per-function recompile limit is shared by every layer and test in the
    process. Pin the state in which the no-grad forward of a reentrant checkpoint
    finds the cache full after the grad-mode forward has been compiled.
    """
    config_names = ("recompile_limit", "cache_size_limit")
    limit = next(name for name in config_names if hasattr(torch._dynamo.config, name))
    torch._dynamo.reset()
    try:
        torch.manual_seed(10)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=4,
            mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
            mhc_keep_mappings_in_fp32=True,
        )
        module = HyperConnectionModule(config, layer_number=1).cuda()
        # Another sequence length first: with dynamic shapes, the entry it leads to
        # would not accept the detached inputs of the no-grad forward below.
        module.compute_mappings(torch.randn(16, 2, 128, device="cuda", requires_grad=True))
        x = torch.randn(8, 2, 128, device="cuda", requires_grad=True)
        expected = module.compute_mappings(x)
        sum(t.float().square().sum() for t in expected).backward()
        expected_grads = [x.grad.clone()] + [p.grad.clone() for p in module.parameters()]
        module.zero_grad(set_to_none=True)
        replay_x = x.detach().clone().requires_grad_(True)
        with torch._dynamo.config.patch(**{limit: 1}):
            actual = checkpoint(module.compute_mappings, replay_x, use_reentrant=True)
            for a, e in zip(actual, expected):
                assert torch.equal(a, e)
            sum(t.float().square().sum() for t in actual).backward()
        grads = [replay_x.grad] + [p.grad for p in module.parameters()]
        for a, e in zip(grads, expected_grads):
            assert torch.equal(a, e)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("frozen", [False, True])
def test_mhc_compute_h_runs_one_specialization(device, input_requires_grad, frozen):
    """Both halves of a full-recompute pair run the same compiled ``_compute_h`` entry.

    The no-grad forward and the grad-mode replay must not need a second Dynamo
    entry, whatever the inputs' requires_grad flags and the module's frozen
    parameters, also after another sequence length has been compiled.
    """
    config_names = ("recompile_limit", "cache_size_limit")
    limit = next(name for name in config_names if hasattr(torch._dynamo.config, name))
    fail_names = ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit")
    fail = next((name for name in fail_names if hasattr(torch._dynamo.config, name)), None)
    torch._dynamo.reset()
    try:
        torch.manual_seed(10)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=4,
            mhc_keep_mappings_in_fp32=True,
        )
        module = HyperConnectionModule(config, layer_number=1).to(device)
        module.requires_grad_(not frozen)

        def inputs(s):
            base = torch.randn(s * 2, 24, device=device, requires_grad=input_requires_grad)
            # Views of 2-D tensors, as _projection_and_get_norm returns them.
            return (base * 1).view(s, 2, 24), (base[:, :1].abs() + 1).view(s, 2, 1)

        module._compute_h_one_specialization(*inputs(16))
        proj, r = inputs(8)
        replay = module._compute_h_one_specialization(proj, r)
        # A second entry must fail hard instead of running eagerly; the unit-test
        # package turns Dynamo errors into eager fallbacks (suppress_errors).
        patch = {limit: 2, "suppress_errors": False}
        if fail is not None:
            patch[fail] = True
        with torch._dynamo.config.patch(**patch):
            with torch.no_grad():
                forward = module._compute_h_one_specialization(proj, r)
            again = module._compute_h_one_specialization(proj, r)
        needs_grad = input_requires_grad or not frozen
        for f, g, a in zip(forward, replay, again):
            assert not f.requires_grad
            assert g.requires_grad == needs_grad
            assert torch.equal(f, g) and torch.equal(a, g)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mhc_mappings_under_inference_mode(device):
    """Inference mode records no autograd; its mappings match the no-grad forward's."""
    torch.manual_seed(10)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_keep_mappings_in_fp32=True,
    )
    module = HyperConnectionModule(config, layer_number=1).to(device)
    x = torch.randn(8, 2, 128, device=device)
    with torch.no_grad():
        expected = module.compute_mappings(x)
    with torch.inference_mode():
        actual = module.compute_mappings(x.clone())
    for a, e in zip(actual, expected):
        assert not a.requires_grad
        torch.testing.assert_close(a, e)


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
        enable_mhc_connections=True,
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
