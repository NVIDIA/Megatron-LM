# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused row kernels of the AngularMuOwn optimizer step.

AngularMuOwn writes each row of a Muon matrix as ``W_i = g_i U_i`` with a unit direction ``U_i``
and a signed scale ``g_i`` (see ``FsdpAngularMuOwn``). Its work before the owner gather and after
the scatter is per row, so each is one Triton kernel with one program per local row: a first pass
over the columns reduces the row, a second pass writes it.

The ``(rows, cols)`` tensors are this rank's local rows, contiguous; the row states are
contiguous ``(rows, 1)`` FP32 tensors. Computation is in FP32.
"""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _momentum_inputs_kernel(
    weight_ptr,
    grad_ptr,
    momentum_ptr,
    ns_input_ptr,
    row_scale_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    n_cols,
    beta,
    beta1,
    beta2,
    norm_eps,
    NESTEROV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols
    g = tl.load(row_scale_ptr + row)
    # copysign(max(|g|, norm_eps), g), with g = -0 taken as +0.
    scale = tl.where(g < 0, -1.0, 1.0) * tl.maximum(tl.abs(g), norm_eps)

    # U = W / g.  d = <G, U>: the radial gradient, which is dL/dg.
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        w = tl.load(weight_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        acc += grad * (w / scale)
    radial_grad = tl.sum(acc, axis=0)

    # Adam moments of g:  m_g <- beta1 * m_g + (1 - beta1) * d,
    #                     v_g <- beta2 * v_g + (1 - beta2) * d^2.
    exp_avg = tl.load(exp_avg_ptr + row)
    tl.store(exp_avg_ptr + row, exp_avg + (1 - beta1) * (radial_grad - exp_avg))
    exp_avg_sq = tl.load(exp_avg_sq_ptr + row)
    tl.store(exp_avg_sq_ptr + row, exp_avg_sq * beta2 + (1 - beta2) * radial_grad * radial_grad)

    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        w = tl.load(weight_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        # dL/dU = g * (G - d * U): the direction gradient, tangent to the unit sphere.
        direction_grad = (grad - radial_grad * (w / scale)) * g
        # Raw-sum momentum:  M <- beta * M + dL/dU.
        momentum = tl.load(momentum_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        momentum = beta * momentum + direction_grad
        tl.store(momentum_ptr + base + cols, momentum.to(momentum_ptr.dtype.element_ty), mask=mask)
        # Newton-Schulz input:  dL/dU + beta * M with Nesterov, otherwise M.
        if NESTEROV:
            ns_input = direction_grad + beta * momentum
        else:
            ns_input = momentum
        tl.store(ns_input_ptr + base + cols, ns_input.to(ns_input_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _apply_update_kernel(
    weight_ptr,
    update_ptr,
    row_scale_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    n_cols,
    direction_lr,
    lr,
    decay,
    bias_correction1,
    bias_correction2,
    adam_eps,
    norm_eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols
    g = tl.load(row_scale_ptr + row)
    # copysign(max(|g|, norm_eps), g), with g = -0 taken as +0.
    scale = tl.where(g < 0, -1.0, 1.0) * tl.maximum(tl.abs(g), norm_eps)

    # U+ = W / g - lr * kappa_t * O;  rho = ||U+||,  rho_den = max(rho, norm_eps).
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        w = tl.load(weight_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        direction = w / scale - direction_lr * update
        acc += direction * direction
    row_norm = tl.sqrt(tl.sum(acc, axis=0))
    denominator = tl.maximum(row_norm, norm_eps)

    # Adam on g with decoupled decay:
    #   g <- (1 - lr * lambda') * g - lr * m_hat / (sqrt(v_hat) + eps),
    #   m_hat = m_g / (1 - beta1^t),  v_hat = v_g / (1 - beta2^t).
    exp_avg = tl.load(exp_avg_ptr + row)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + row)
    new_g = decay * g - lr * (exp_avg / bias_correction1) / (
        tl.sqrt(exp_avg_sq / bias_correction2) + adam_eps
    )

    # W = g * U+ / rho_den, with the U+ that the old g recovers.
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        w = tl.load(weight_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        direction = w / scale - direction_lr * update
        w = direction * (new_g / denominator)
        tl.store(weight_ptr + base + cols, w.to(weight_ptr.dtype.element_ty), mask=mask)
    # g <- g * rho / rho_den: a no-op unless the clamp fired, where it keeps |g| = ||W||.
    tl.store(row_scale_ptr + row, new_g * (row_norm / denominator))


def fused_angular_muown_momentum_inputs(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    ns_input: torch.Tensor,
    row_scale: torch.Tensor,
    row_scale_exp_avg: torch.Tensor,
    row_scale_exp_avg_sq: torch.Tensor,
    *,
    beta: float,
    betas: tuple[float, float],
    norm_eps: float,
    nesterov: bool,
) -> None:
    """Split the gradient by rows, update the moments, and write the Newton-Schulz input.

    Updates ``momentum``, ``row_scale_exp_avg``, ``row_scale_exp_avg_sq``, and ``ns_input`` in
    place; ``weight``, ``grad``, and ``row_scale`` are read only.
    """
    n_rows, n_cols = weight.shape
    block = min(triton.next_power_of_2(n_cols), 4096)
    _momentum_inputs_kernel[(n_rows,)](
        weight,
        grad,
        momentum,
        ns_input,
        row_scale,
        row_scale_exp_avg,
        row_scale_exp_avg_sq,
        n_cols,
        beta,
        betas[0],
        betas[1],
        norm_eps,
        NESTEROV=nesterov,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )


def fused_angular_muown_apply_update(
    weight: torch.Tensor,
    update: torch.Tensor,
    row_scale: torch.Tensor,
    row_scale_exp_avg: torch.Tensor,
    row_scale_exp_avg_sq: torch.Tensor,
    *,
    direction_lr: float,
    lr: float,
    decay: float,
    bias_corrections: tuple[float, float],
    adam_eps: float,
    norm_eps: float,
) -> None:
    """Rotate and retract the rows, take the Adam step on their scales, and rewrite the weights.

    ``direction_lr`` is ``lr * kappa_t`` and ``decay`` is ``1 - lr * lambda'``. Updates
    ``weight`` and ``row_scale`` in place.
    """
    n_rows, n_cols = weight.shape
    block = min(triton.next_power_of_2(n_cols), 4096)
    _apply_update_kernel[(n_rows,)](
        weight,
        update,
        row_scale,
        row_scale_exp_avg,
        row_scale_exp_avg_sq,
        n_cols,
        direction_lr,
        lr,
        decay,
        bias_corrections[0],
        bias_corrections[1],
        adam_eps,
        norm_eps,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
