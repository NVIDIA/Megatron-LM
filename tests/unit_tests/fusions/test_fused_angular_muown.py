# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The fused AngularMuOwn row kernels against the same update written with torch ops."""

import pytest
import torch

from megatron.core.fusions.fused_angular_muown import (
    HAVE_TRITON,
    fused_angular_muown_apply_update,
    fused_angular_muown_momentum_inputs,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)

NORM_EPS = 1e-12


def _signed_floor(g):
    return torch.where(g < 0, -1.0, 1.0) * g.abs().clamp_min(NORM_EPS)


def _rows(n_rows, n_cols, grad_dtype):
    """Weights with signed row scales, a gradient, and nonzero momentum and moments."""
    torch.manual_seed(1234)
    weight = torch.randn(n_rows, n_cols, device="cuda")
    row_scale = weight.norm(dim=1, keepdim=True) * torch.where(
        torch.rand(n_rows, 1, device="cuda") < 0.5, -1.0, 1.0
    )
    return dict(
        weight=weight,
        grad=torch.randn(n_rows, n_cols, device="cuda").to(grad_dtype),
        momentum=torch.randn(n_rows, n_cols, device="cuda"),
        row_scale=row_scale,
        row_scale_exp_avg=torch.randn(n_rows, 1, device="cuda"),
        row_scale_exp_avg_sq=torch.rand(n_rows, 1, device="cuda"),
    )


# One block of columns, and several with a partial last one.
@pytest.mark.parametrize("n_cols", [96, 5000])
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("nesterov", [False, True], ids=["momentum", "nesterov"])
def test_momentum_inputs_match_torch(n_cols, grad_dtype, nesterov):
    """Gradient split, moments, momentum, and Newton-Schulz input."""
    rows = _rows(64, n_cols, grad_dtype)
    beta, (beta1, beta2) = 0.9, (0.9, 0.95)

    expected = {key: value.clone() for key, value in rows.items()}
    direction = expected["weight"] / _signed_floor(expected["row_scale"])
    grad = expected["grad"].float()
    radial_grad = (grad * direction).sum(dim=1, keepdim=True)
    direction_grad = (grad - radial_grad * direction) * expected["row_scale"]
    expected["row_scale_exp_avg"].lerp_(radial_grad, 1 - beta1)
    expected["row_scale_exp_avg_sq"].mul_(beta2).addcmul_(radial_grad, radial_grad, value=1 - beta2)
    expected["momentum"].mul_(beta).add_(direction_grad)
    expected_ns_input = (
        direction_grad + beta * expected["momentum"] if nesterov else expected["momentum"]
    )

    ns_input = torch.empty_like(rows["weight"])
    fused_angular_muown_momentum_inputs(
        rows["weight"],
        rows["grad"],
        rows["momentum"],
        ns_input,
        rows["row_scale"],
        rows["row_scale_exp_avg"],
        rows["row_scale_exp_avg_sq"],
        beta=beta,
        betas=(beta1, beta2),
        norm_eps=NORM_EPS,
        nesterov=nesterov,
    )
    for key in ("momentum", "row_scale_exp_avg", "row_scale_exp_avg_sq"):
        torch.testing.assert_close(rows[key], expected[key], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ns_input, expected_ns_input, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_cols", [96, 5000])
@pytest.mark.parametrize("update_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_apply_update_matches_torch(n_cols, update_dtype):
    """Rotation, retraction, the Adam step on the row scales, and the rewritten weights."""
    rows = _rows(64, n_cols, torch.float32)
    update = (0.01 * torch.randn(64, n_cols, device="cuda")).to(update_dtype)
    lr, kappa, weight_decay, step = 0.05, 0.7, 0.1, 3
    (beta1, beta2), adam_eps = (0.9, 0.95), 1e-8

    expected = {key: value.clone() for key, value in rows.items()}
    direction = expected["weight"] / _signed_floor(expected["row_scale"])
    direction = direction - lr * kappa * update.float()
    row_norm = direction.norm(dim=1, keepdim=True)
    denominator = row_norm.clamp_min(NORM_EPS)
    row_scale = (1 - lr * weight_decay) * expected["row_scale"] - lr * (
        expected["row_scale_exp_avg"] / (1 - beta1**step)
    ) / ((expected["row_scale_exp_avg_sq"] / (1 - beta2**step)).sqrt() + adam_eps)
    expected_weight = direction * (row_scale / denominator)
    expected_row_scale = row_scale * (row_norm / denominator)

    fused_angular_muown_apply_update(
        rows["weight"],
        update,
        rows["row_scale"],
        rows["row_scale_exp_avg"],
        rows["row_scale_exp_avg_sq"],
        direction_lr=lr * kappa,
        lr=lr,
        decay=1 - lr * weight_decay,
        bias_corrections=(1 - beta1**step, 1 - beta2**step),
        adam_eps=adam_eps,
        norm_eps=NORM_EPS,
    )
    torch.testing.assert_close(rows["weight"], expected_weight, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rows["row_scale"], expected_row_scale, rtol=1e-5, atol=1e-6)


def test_empty_rows_are_a_no_op():
    """A rank without rows of a tensor launches nothing."""
    rows = _rows(0, 32, torch.float32)
    fused_angular_muown_apply_update(
        rows["weight"],
        torch.empty_like(rows["weight"]),
        rows["row_scale"],
        rows["row_scale_exp_avg"],
        rows["row_scale_exp_avg_sq"],
        direction_lr=0.1,
        lr=0.1,
        decay=1.0,
        bias_corrections=(0.1, 0.05),
        adam_eps=1e-8,
        norm_eps=NORM_EPS,
    )
