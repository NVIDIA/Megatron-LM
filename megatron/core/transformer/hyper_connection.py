# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from megatron.core._rank_utils import log_single_rank, safe_get_rank
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer.mhc_fused_ops import (
    MHC_FUSED_OPS,
    MHC_PRECISION_FLAGS,
    mhc_fused_ops_errors,
    mhc_fused_ops_field_errors,
    triton_only_ops,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionOperation,
    ResidualConnectionState,
)
from megatron.core.transformer.residual_recompute import (
    ResidualStreamRecomputeContext,
    build_recompute_block_end_plan,
    build_recompute_layer_managers,
    finalize_recompute_block,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import ensure_params_ready, nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import MHCCheckpointManager
    from megatron.core.transformer.mhc_recompute import MHCRecomputeArenaSlot

logger = logging.getLogger(__name__)

_MHC_SINKHORN_EPS = 1e-6
_MHC_COMPUTE_H_EPS = 1e-6

# What this process has already logged: each distinct fused-ops line, and each per-op fallback
# warning (by operations and reason).
_FUSED_OPS_LINES_LOGGED: set[str] = set()
_PER_OP_FALLBACK_WARNED: set[Tuple[Tuple[str, ...], str]] = set()


def _resolve_mhc_checkpoint(
    manager: CheckpointWithoutOutputManager | None,
    checkpoint_fn: Callable[..., Tensor] | None,
    *,
    output_slot: Optional['MHCRecomputeArenaSlot'] = None,
) -> Callable[..., Tensor]:
    """Resolve a ``checkpoint(function, *args)`` callable.

    Use checkpoint_fn if supplied; otherwise create a fresh CheckpointWithoutOutput
    per call and register it with manager, if provided. The managed checkpoint writes
    its single output into output_slot, if provided. Reject supplying checkpoint_fn
    together with manager or output_slot.
    """
    if checkpoint_fn is not None:
        if manager is not None:
            raise ValueError("Specify either checkpoint_fn or an mHC recompute manager, not both.")
        if output_slot is not None:
            raise ValueError(
                "An mHC output_slot requires an mHC recompute manager, not checkpoint_fn."
            )
        return checkpoint_fn

    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    def checkpoint_with_manager(function, *args):
        # Preserve the legacy fp8=False default.
        ckpt = CheckpointWithoutOutput(ckpt_manager=manager, output_slot=output_slot)
        return ckpt.checkpoint(function, *args)

    return checkpoint_with_manager


def build_mhc_recompute_layer_plan(
    num_layers: int, mhc_recompute_layer_num: Optional[int], use_mhc_recompute: bool
) -> Tuple[list[Optional[CheckpointWithoutOutputManager]], list[bool]]:
    """Deprecated adapter returning per-layer managers and block-end markers.

    Use ``build_residual_stream_recompute_plan`` when recompute is enabled. It returns
    one context per layer instead of two lists and does not take an enabled flag.
    """
    warnings.warn(
        "build_mhc_recompute_layer_plan is deprecated and will be removed in a future release. "
        "Use build_residual_stream_recompute_plan when recompute is enabled instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not use_mhc_recompute or num_layers == 0:
        return [None] * num_layers, [False] * num_layers

    block_ends = build_recompute_block_end_plan(num_layers, mhc_recompute_layer_num)
    return build_recompute_layer_managers(block_ends), list(block_ends)


def finalize_mhc_recompute_layer(
    mhc_manager: Optional[CheckpointWithoutOutputManager],
    hidden_states: Tensor,
    is_last_in_recompute_block: bool,
) -> None:
    """Deprecated adapter; use ``ResidualStreamRecomputeContext.finalize`` instead."""
    warnings.warn(
        "finalize_mhc_recompute_layer is deprecated and will be removed in a future release. "
        "Use ResidualStreamRecomputeContext.finalize(hidden_states) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    finalize_recompute_block(mhc_manager, hidden_states, is_last_in_recompute_block)


# dynamic=True handles the hybrid mHC variable-shape path (was blanket-disabled)
@torch.compile
def _sinkhorn_iterations(input_logits: Tensor, num_iterations: int, eps: float) -> Tensor:
    M = input_logits.softmax(dim=-1) + eps
    M = M / (M.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iterations - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class SinkhornKnopp(torch.autograd.Function):
    """Sinkhorn-Knopp projection to doubly stochastic matrix.

    This is an autograd.Function because the iterative forward is re-executed
    during backward (under torch.enable_grad) so that PyTorch's autograd can
    differentiate through it without storing all intermediate iteration states.
    """

    @staticmethod
    def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
        """Run Sinkhorn iterations and save inputs for backward recomputation."""
        # This forward is always no-grad. Canonicalize the input's requires_grad flag
        # too, so checkpointing reuses the same compiled specialization even when
        # Dynamo's cache is full instead of mixing compiled and eager arithmetic.
        M = _sinkhorn_iterations(input_logits.detach(), num_iterations, eps)
        ctx.save_for_backward(input_logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return M

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Recompute forward under enable_grad and back-propagate."""
        (input_logits,) = ctx.saved_tensors
        with torch.enable_grad():
            logits = input_logits.detach().requires_grad_(True)
            M = _sinkhorn_iterations(logits, ctx.num_iterations, ctx.eps)
            M.backward(grad_output)
        return logits.grad, None, None


def native_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    """Native Sinkhorn-Knopp (autograd.Function wrapper)."""
    return SinkhornKnopp.apply(input_logits, num_iterations, eps)


# dynamic=True handles the hybrid mHC variable-shape path (was blanket-disabled)
@torch.compile
def native_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Native n-stream weighted aggregation: out = sum_j(h_pre_j * x_j)."""
    return (x * h_pre.unsqueeze(-1)).sum(dim=2).to(x.dtype)


@torch.compile
def _native_h_aggregate_into_forward(x: Tensor, h_pre: Tensor, out: Tensor) -> Tensor:
    """Fuse the same multiply/reduction as native_h_aggregate into caller-owned storage."""
    # Reduce in the promoted dtype and round once into ``out``, as native_h_aggregate does.
    # sum(..., out=out) would round FP32 mapping products to a BF16 buffer before reducing.
    return out.copy_((x * h_pre.unsqueeze(-1)).sum(dim=2))


class NativeHAggregateInto(torch.autograd.Function):
    """Native H-aggregate whose result is produced in caller-owned storage.

    The explicit output is needed when the consumer is a CUDA Graph: its input
    address is part of the captured launch parameters and therefore cannot be
    repaired by rebinding a logical tensor after recomputation.
    """

    @staticmethod
    def forward(ctx, x: Tensor, h_pre: Tensor, out: Tensor) -> Tensor:
        """Aggregate into caller-owned ``out`` after validating its contract."""
        if out.shape != x.shape[:2] + x.shape[3:]:
            raise ValueError(
                f"H-aggregate output shape {tuple(out.shape)} does not match "
                f"{tuple(x.shape[:2] + x.shape[3:])}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError("H-aggregate output dtype/device must match x")
        # Mirror FusedHAggregateInto.forward's contract exactly: the consumer graph
        # dereferences the address seen at capture time, so a strided slot would
        # hand it a layout it was not captured against. Unreachable while the arena
        # only vends contiguous views -- the point is that both entry points fail
        # on the same inputs if that ever changes.
        if not out.is_contiguous():
            raise ValueError("H-aggregate caller-owned output must be contiguous")
        if out.requires_grad:
            raise ValueError("H-aggregate caller-owned output must be a detached tensor")

        ctx.mark_dirty(out)
        # An eager multiply materializes a rounded intermediate;
        # the allocating path fuses it with the reduction. Keep both paths fused
        # without allocating an aggregate and copying it into the graph's slot.
        # The forward is always no-grad. Canonicalize the inputs' requires_grad flags, as
        # SinkhornKnopp.forward does, so a checkpoint's no-grad forward and its replay reuse
        # one compiled specialization even when Dynamo's cache is full.
        _native_h_aggregate_into_forward(x.detach(), h_pre.detach(), out)
        ctx.save_for_backward(x, h_pre)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Back-propagate the aggregation, accumulating grad_h in fp32."""
        x, h_pre = ctx.saved_tensors
        grad_output_expanded = grad_output.unsqueeze(2)
        grad_x = grad_output_expanded * h_pre.unsqueeze(-1)
        # grad_h reduces over the hidden dimension, thousands of elements wide,
        # while the forward only reduces over the handful of residual streams.
        # Upcast both operands: torch.sum already accumulates bf16 in fp32, so
        # asking for dtype=float32 would change nothing; the cost is rounding each
        # product to bf16 before that reduction. h_pre carries the residual mixing
        # coefficients, so error here shifts how streams combine in every layer.
        # Matches _torch_h_aggregate_bwd, including its note on why the
        # temporary-free matmul form is not used.
        grad_h = torch.sum(grad_output_expanded.float() * x.float(), dim=-1)
        return grad_x.to(dtype=x.dtype), grad_h.to(dtype=h_pre.dtype), None


def native_h_aggregate_into(x: Tensor, h_pre: Tensor, out: Tensor) -> Tensor:
    """Native weighted aggregation that writes directly to ``out``."""
    return NativeHAggregateInto.apply(x, h_pre, out)


def _h_res_mix(h_res: Tensor, residual: Tensor) -> Tensor:
    """H_res.T @ residual as multiply-adds over the streams, rounded once to h_res.dtype.

    mixed[..., i, :] = sum_j h_res[..., j, i] * residual[..., j, :] for h_res [..., n, n] and
    residual [..., n, C], accumulated in FP32 (FP64 for FP64 inputs) in ascending j, as the
    fused forward kernel does. With n <= 4 streams a GEMM buys nothing, and its precision
    depends on the TF32 policy and on how Inductor pads and lowers each compiled
    specialization: such a split made full-recompute forwards differ from the reference.
    """
    with torch.autocast(device_type=h_res.device.type, enabled=False):
        acc_dtype = torch.promote_types(h_res.dtype, torch.float32)
        h_res_acc = h_res.to(acc_dtype)
        # Upcast one residual row per product and accumulate in place: an uncompiled call
        # (Dynamo disabled or out of recompiles) then holds two FP32 copies of the residual,
        # not four.
        mixed = h_res_acc[..., 0, :, None] * residual[..., 0:1, :].to(acc_dtype)
        for j in range(1, h_res.shape[-1]):
            mixed += h_res_acc[..., j, :, None] * residual[..., j : j + 1, :].to(acc_dtype)
    return mixed.to(h_res.dtype)


def _native_h_post_bda_impl(
    h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """Native H_res.T @ residual + H_post * (x [+ bias])."""
    C = original_residual.shape[-1]
    mixed = _h_res_mix(h_res, original_residual.to(h_res.dtype))
    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)
    if bias is not None:
        bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
        return (x_expanded + bias_expanded + mixed).to(original_residual.dtype)
    return (x_expanded + mixed).to(original_residual.dtype)


def _native_h_post_bda_bwd_impl(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Gradients of _native_h_post_bda_impl, in the dtypes autograd gives them.

    Each term receives the output gradient in its own dtype and each input gradient is
    rounded to its input's dtype, as autograd through the forward does. The mixing gradients
    are FP32 multiply-adds like the forward, not GEMMs.
    """
    C = original_residual.shape[-1]
    h_post_expanded = h_post.unsqueeze(-1)
    x_dtype = torch.promote_types(h_post.dtype, x.dtype)
    sum_dtype = torch.promote_types(x_dtype, h_res.dtype)
    if bias is not None:
        bias_dtype = torch.promote_types(h_post.dtype, bias.dtype)
        sum_dtype = torch.promote_types(sum_dtype, bias_dtype)
    with torch.autocast(device_type=h_res.device.type, enabled=False):
        grad_sum = grad_output.to(sum_dtype)
        grad_mixed = grad_sum.to(h_res.dtype)
        # grad_h_res[j, i] = sum_c residual[j, c] * grad_mixed[i, c]: FP32 products (the residual
        # promotes to the upcast gradient row) reduced over C. One gradient row at a time, so an
        # uncompiled call (Dynamo disabled or out of recompiles) never holds an n x n x C product.
        acc_dtype = torch.promote_types(h_res.dtype, torch.float32)
        residual = original_residual.to(h_res.dtype)
        grad_h_res = torch.stack(
            [
                (residual * grad_mixed[..., i : i + 1, :].to(acc_dtype)).sum(dim=-1)
                for i in range(h_res.shape[-1])
            ],
            dim=-1,
        )
        # grad_residual[j, :] = sum_i h_res[j, i] * grad_mixed[i, :]
        grad_residual = _h_res_mix(h_res.transpose(-1, -2), grad_mixed)
        grad_x_expanded = grad_sum.to(x_dtype)
        grad_h_post = (grad_x_expanded * x.unsqueeze(2)).sum(dim=-1).to(h_post.dtype)
        grad_x = (grad_x_expanded * h_post_expanded).sum(dim=2).to(x.dtype)
        grad_bias = None
        if bias is not None:
            grad_bias_expanded = grad_sum.to(bias_dtype)
            grad_h_post_bias = (grad_bias_expanded * bias.view(1, 1, 1, C)).sum(dim=-1)
            grad_h_post = grad_h_post + grad_h_post_bias.to(h_post.dtype)
            grad_bias = (grad_bias_expanded * h_post_expanded).sum(dim=(0, 1, 2)).to(bias.dtype)
    return (
        grad_h_res.to(h_res.dtype),
        grad_residual.to(original_residual.dtype),
        grad_h_post,
        grad_x,
        grad_bias,
    )


class NativeHPostBDA(torch.autograd.Function):
    """Native mHC residual update whose compiled forward serves every pass.

    Full recompute runs the forward under no_grad and replays it with gradients. As one
    compiled function, each grad mode and set of requires_grad flags was a separate Dynamo
    specialization that Inductor could lower differently, so a recomputed output could differ
    from the reference, and the specializations exhausted the recompile limit, after which
    calls mixed compiled and eager arithmetic. The compiled forward runs on detached inputs, as
    in SinkhornKnopp; unlike SinkhornKnopp, a compiled backward computes the gradients from the
    saved inputs without recomputing the output.
    """

    @staticmethod
    def forward(
        ctx,
        forward_impl: Callable,
        backward_impl: Callable,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        """Run ``forward_impl`` on detached inputs and save them for ``backward_impl``."""
        output = forward_impl(
            h_res.detach(),
            original_residual.detach(),
            h_post.detach(),
            x.detach(),
            None if bias is None else bias.detach(),
        )
        ctx.save_for_backward(h_res, original_residual, h_post, x, bias)
        ctx.backward_impl = backward_impl
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: Tensor):
        """Return every input's gradient from one compiled call on detached tensors."""
        saved = [None if t is None else t.detach() for t in ctx.saved_tensors]
        return (None, None, *ctx.backward_impl(grad_output.detach(), *saved))


@torch.compile
def _native_h_post_bda_forward(
    h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """native_h_post_bda's compiled forward, with the default compiler options."""
    return _native_h_post_bda_impl(h_res, original_residual, h_post, x, bias)


@torch.compile
def _native_h_post_bda_backward(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """native_h_post_bda's compiled backward, with the default compiler options."""
    return _native_h_post_bda_bwd_impl(grad_output, h_res, original_residual, h_post, x, bias)


def native_h_post_bda(
    h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """Compiled mHC residual update (NativeHPostBDA with the default compiler options)."""
    return NativeHPostBDA.apply(
        _native_h_post_bda_forward,
        _native_h_post_bda_backward,
        h_res,
        original_residual,
        h_post,
        x,
        bias,
    )


@lru_cache(maxsize=1)
def _get_precision_preserving_h_post_bda():
    """Lazily compile HF's activation-dtype rounding, without changing global settings.

    The backward is compiled with the same option, so the gradients round the
    activation-dtype products as eager autograd does.
    """
    message = (
        "mhc_preserve_activation_dtype_rounding requires torch.compile support for "
        "emulate_precision_casts; no eager or reduced-precision fallback is supported."
    )
    try:
        from torch._inductor import list_options
    except ImportError as error:
        raise RuntimeError(message) from error
    if "emulate_precision_casts" not in list_options():
        raise RuntimeError(message)
    options = {"emulate_precision_casts": True}
    try:
        forward = torch.compile(_native_h_post_bda_impl, options=options)
        backward = torch.compile(_native_h_post_bda_bwd_impl, options=options)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(message) from error
    return partial(NativeHPostBDA.apply, forward, backward)


def _proj_rms_impl(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    eps_inside_sqrt: bool = False,
    normalize_before_projection: bool = False,
) -> Tuple[Tensor, Tensor]:
    """FP32 projection + RMS normalization from an activation-dtype input.

    With normalize_before_projection, the projection consumes the RMS-normalized input,
    as GLM does, and the caller must not scale it by the returned norm again. Otherwise
    the caller scales the projection afterwards. Preserve this distinction rather than
    commuting the FP32 GEMM.
    """
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    # Casting the operands is not sufficient under CUDA/CPU autocast: matmul and
    # reductions are still eligible to downcast float32 inputs.  Keep the whole
    # projection/RMS recipe in FP32; the caller controls the output cast.
    with torch.autocast(device_type=x.device.type, enabled=False):
        if eps_inside_sqrt:
            norm = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        else:
            norm = 1.0 / (x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1]) + eps)
        if normalize_before_projection:
            # HF 93c8b7b Glm5NextTextHyperConnection normalizes the FP32 residual
            # before F.linear. This intentionally differs from PR 7521's ordering.
            proj = F.linear(x * norm, weight)
        else:
            proj = torch.matmul(x, weight.t())
    return proj, norm


# dynamic=True handles the hybrid mHC variable-shape path (was blanket-disabled)
@torch.compile
def native_proj_rms(
    x: Tensor, weight: Tensor, eps: float = 1e-6, eps_inside_sqrt: bool = False
) -> Tuple[Tensor, Tensor]:
    """Compiled FP32 projection + RMS normalization.

    ``x`` may retain the activation dtype so callers can checkpoint this function without
    saving an FP32 upcast of the residual stream.
    """
    return _proj_rms_impl(x, weight, eps, eps_inside_sqrt)


# dynamic=True handles the hybrid mHC variable-shape path (was blanket-disabled)
@torch.compile
def native_fused_add_3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Native 3-way elementwise add (torch.compile fuses into single kernel)."""
    return a + b + c


class BroadcastTensorFused(torch.autograd.Function):
    """Split one tensor into 3 autograd-graph children sharing the same storage.

    During backward the three incoming gradients are summed with a caller-
    supplied fused-add function (cuTile or torch.compile fallback) instead of
    PyTorch's default sequential accumulation.
    """

    @staticmethod
    def forward(ctx, x, fused_add_3_fn):
        """Return three view aliases and save the fused gradient combiner."""
        ctx.fused_add_3_fn = fused_add_3_fn
        return x.view_as(x), x.view_as(x), x.view_as(x)

    @staticmethod
    def backward(ctx, grad1, grad2, grad3):
        """Combine gradients from the three broadcast aliases."""
        grads = [g for g in (grad1, grad2, grad3) if g is not None]
        if len(grads) == 0:
            return None, None
        if len(grads) == 1:
            return grads[0], None
        if len(grads) == 2:
            return grads[0] + grads[1], None
        return ctx.fused_add_3_fn(grad1, grad2, grad3), None


@torch.compile
def learned_output_contract(
    hidden_states: Tensor, head_fn: Tensor, base: Tensor, scale: Tensor, n: int, eps: float
) -> Tensor:
    """Learned output contraction: n-stream → 1-stream via sigmoid-gated weighted sum."""
    dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    head_fn = head_fn.to(torch.float32)
    base = base.to(torch.float32)
    scale = scale.to(torch.float32)
    rsqrt = torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(hidden_states, head_fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + eps
    y = torch.sum(pre.unsqueeze(-1) * hidden_states.view(*hidden_states.shape[:-1], n, -1), dim=-2)
    return y.to(dtype)


# ============================================================================
# Fused-operation selection: labels and logging
# ============================================================================

# The entry of fused_mhc_kernels' backend status message that describes each operation.
_FMK_STATUS_KEYS = {"proj_rms": "proj_rms_compute_h"}


def _read_or_error(read: Callable[[], object]) -> str:
    """Return ``str(read())``, or the name of the exception it raises."""
    try:
        return str(read())
    except Exception as error:  # torch raises when the legacy and new TF32 APIs were mixed
        return f"<{type(error).__name__}>"


def _precision_fields() -> Tuple[Tuple[str, str], ...]:
    """Return the FP32 matmul precision state that decides whether TF32 runs."""
    # One accessor per try, so that one raising accessor does not hide the others. Never read
    # allow_tf32: it raises in the same mixed states and does not decide TF32 by itself.
    return (
        ("fp32_matmul_precision", _read_or_error(torch.get_float32_matmul_precision)),
        (
            "cuda.matmul.fp32_precision",
            _read_or_error(lambda: torch.backends.cuda.matmul.fp32_precision),
        ),
        (
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "unset"),
        ),
        ("NVIDIA_TF32_OVERRIDE", os.environ.get("NVIDIA_TF32_OVERRIDE", "unset")),
    )


def _unfused_op_labels(config: TransformerConfig) -> Dict[str, str]:
    """Return the labels of the unfused operations a module binds for ``config``."""
    # From the flag: the default module must never call _get_precision_preserving_h_post_bda.
    post = (
        "native(emulate-casts)"
        if config.mhc_preserve_activation_dtype_rounding
        else "native(compiled)"
    )
    return {"proj_rms": "native", "sinkhorn": "native", "h_aggregate": "native", "h_post_bda": post}


def _fmk_op_labels(policy: str, ops: Iterable[str]) -> Dict[str, str]:
    """Label operations with the fused kernels' own backend resolution.

    For example ``fmk(auto:fwd:triton,bwd:native)``; an entry the status message lacks prints
    as ``?``.
    """
    from megatron.core.fusions import fused_mhc_kernels as fmk  # lazy: it imports this module

    # The resolution that log_fused_mhc_backend_once prints. It validates the policy first.
    message, _ = fmk._mhc_backend_status(policy)
    resolved = dict(part.split("=", 1) for part in message.split("; ") if "=" in part)
    return {op: f"fmk({policy}:{resolved.get(_FMK_STATUS_KEYS.get(op, op), '?')})" for op in ops}


def _log_mhc_fused_ops_once(config: TransformerConfig, mode: str, labels: Dict[str, str]) -> None:
    """Log which mHC operations run fused, on every rank, once per process per distinct line.

    Logged when ``use_fused_mhc`` or an mHC precision flag is set. Every rank logs, because
    the TF32 state and the backend availability are properties of each rank's process.
    """
    if not (config.use_fused_mhc or any(getattr(config, flag) for flag in MHC_PRECISION_FLAGS)):
        return
    # As log_single_rank does: without INFO, do not look up the rank, which warns in processes
    # without torch.distributed, RANK or SLURM_PROCID.
    if not logger.isEnabledFor(logging.INFO):
        return
    fields = [
        f"mode={mode}",
        f"policy={config.mhc_fused_backend}",
        f"MHC_FORCE_BACKEND={os.environ.get('MHC_FORCE_BACKEND', 'unset')}",
    ]
    fields += [f"{op}={labels[op]}" for op in MHC_FUSED_OPS]
    fields += [f"{name}={value}" for name, value in _precision_fields()]
    line = f"[rank {safe_get_rank()}] [mHC] fused ops: " + " ".join(fields)
    if line not in _FUSED_OPS_LINES_LOGGED:
        _FUSED_OPS_LINES_LOGGED.add(line)
        logger.info(line)


def _warn_per_op_fallback_once(ops: List[str], reason: str) -> None:
    """Warn once per process that listed operations keep their unfused implementation."""
    key = (tuple(ops), reason)
    if key not in _PER_OP_FALLBACK_WARNED:
        _PER_OP_FALLBACK_WARNED.add(key)
        log_single_rank(
            logger,
            logging.WARNING,
            f"[mHC] Falling back to the unfused path for mhc_fused_ops {list(ops)}: {reason}.",
        )


# ============================================================================
# HyperConnectionModule
# ============================================================================


# TODO: keep hyper connection in fp32 computation
class HyperConnectionModule(MegatronModule):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.

    Implements the complete mHC propagation:
        x_{l+1} = H_res^T @ x_l + H_post^T @ F(H_pre @ x_l)

    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream → 1-stream (H_pre @ x)
    3. Expansion: 1-stream → n-stream (H_post^T @ output)
    4. Residual merge: H_res^T @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries

    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.sinkhorn_eps = _MHC_SINKHORN_EPS
        self.compute_h_eps = _MHC_COMPUTE_H_EPS

        # Projection weights for dynamic mappings
        # Input: [s, b, n*C] -> Output: n^2 + 2n values per token
        # - H_pre: n values
        # - H_post: n values
        # - H_res: n^2 values (before Sinkhorn projection)
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size, self.n * self.n + 2 * self.n, bias=False
        )

        init_alpha = config.mhc_init_gating_factor
        # Learnable scaling factors (Eq. 5 in paper)
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))

        # Static bias terms
        self.bias = nn.Parameter(torch.zeros(self.n * self.n + 2 * self.n))
        mark_keep_in_fp32(self.mapping_proj.weight)
        mark_keep_in_fp32(self.alpha_pre)
        mark_keep_in_fp32(self.alpha_post)
        mark_keep_in_fp32(self.alpha_res)
        mark_keep_in_fp32(self.bias)
        self.norm_eps = config.layernorm_epsilon if config.mhc_norm_eps_inside_sqrt else 1e-6

        # Choose implementation: unified fused kernels vs reference modules.
        # The fused public API selects the backend per operation internally.
        # fused_add_3 always uses torch.compile (native_fused_add_3) regardless
        # of the kernel backend. cuTile's register overhead (56 regs/thread for
        # a trivial a+b+c) is not worth it for a pure memory-bound elementwise op.
        self._fused_add_3_op = native_fused_add_3

        # The all-operations fused mode computes the projection and compute_h in one op,
        # so there _projection_and_get_norm — and therefore _proj_rms_op — is never
        # reached. In per-op mode _projection_and_get_norm runs unless proj_rms binds
        # that projection-first fused op.
        # Full recompute forwards run under no_grad and replay with gradients. Use the
        # same FP32 arithmetic in both passes; compiling only one changes norm reductions.
        self._proj_rms_op = partial(
            _proj_rms_impl,
            eps_inside_sqrt=config.mhc_norm_eps_inside_sqrt,
            normalize_before_projection=config.mhc_normalize_before_projection,
        )
        # The Triton GLM projection pair (fused_mhc_glm_projection), which per-op proj_rms binds
        # with mhc_norm_eps_inside_sqrt and mhc_normalize_before_projection; None everywhere else.
        self._glm_proj_rms_op = None

        if config.use_fused_mhc and config.mhc_fused_ops is None:
            if (
                config.mhc_norm_eps_inside_sqrt
                or config.mhc_normalize_before_projection
                or config.mhc_keep_mappings_in_fp32
                or config.mhc_keep_pre_in_fp32
                or config.mhc_preserve_activation_dtype_rounding
            ):
                raise ValueError(
                    "Fused mHC does not support FP32 mixing, epsilon inside sqrt, "
                    "normalize-before-projection or activation-dtype rounding preservation."
                )
            from megatron.core.fusions.fused_mhc_kernels import (
                fused_h_aggregate,
                fused_h_aggregate_into,
                fused_h_post_bda,
                fused_proj_rms_compute_h,
                fused_sinkhorn,
                log_fused_mhc_backend_once,
            )

            backend = config.mhc_fused_backend
            log_fused_mhc_backend_once(backend)
            self._sinkhorn_op = partial(fused_sinkhorn, backend=backend)
            self._h_aggregate_op = partial(fused_h_aggregate, backend=backend)
            self._h_aggregate_into_op = partial(fused_h_aggregate_into, backend=backend)
            self._h_post_bda_op = partial(fused_h_post_bda, backend=backend)
            self._proj_rms_compute_h_op = partial(fused_proj_rms_compute_h, backend=backend)
            fused_mode, fused_labels = "all-ops", _fmk_op_labels(backend, MHC_FUSED_OPS)
        else:
            self._sinkhorn_op = native_sinkhorn
            self._h_aggregate_op = native_h_aggregate
            self._h_aggregate_into_op = native_h_aggregate_into
            self._h_post_bda_op = (
                _get_precision_preserving_h_post_bda()
                if config.mhc_preserve_activation_dtype_rounding
                else native_h_post_bda
            )
            self._proj_rms_compute_h_op = None
            fused_mode, fused_labels = "unfused", _unfused_op_labels(config)
            if config.use_fused_mhc:
                # Per-op mode: rebind only the operations listed in mhc_fused_ops.
                fused_mode = "per-op"
                fused_labels.update(self._bind_per_op_fused_ops(config))
        self._fused_op_labels = fused_labels
        _log_mhc_fused_ops_once(config, fused_mode, fused_labels)

        self._init_weights()

    def _bind_per_op_fused_ops(self, config: TransformerConfig) -> Dict[str, str]:
        """Rebind only the operations listed in ``mhc_fused_ops``; return their log labels.

        Every other operation keeps the unfused object bound before this call. Under
        ``MHC_FORCE_BACKEND=native`` or ``mhc_fused_backend="native"`` the listed operations
        keep them too, so in per-op mode the native policy means the unfused path. The
        all-operations mode binds the fused kernels' projection-first native op instead; the
        difference is intended.

        ``sinkhorn`` binds the fixed-launch Triton Sinkhorn of fused_mhc_glm_projection when
        Triton is available; under ``auto`` without Triton it keeps the unfused object, with a
        warning. Triton counts as unavailable also under ``MHC_DISABLE_TRITON=1`` and
        ``MHC_FORCE_BACKEND=cutile``. proj_rms with ``mhc_norm_eps_inside_sqrt`` and
        ``mhc_normalize_before_projection`` binds the Triton GLM projection pair of
        fused_mhc_glm_projection when Triton is available, with the TF32 rounding of
        ``mhc_fused_tf32_rounding``; under ``auto`` without Triton it keeps the unfused object,
        with a warning. At call time the pair runs only on inputs it supports; on any other input
        the module warns once and runs the unfused projection under ``auto``, and raises under
        ``triton``.
        """
        # Configs edited after TransformerConfig validation. The table needs a well-formed
        # list, so it runs only when the field checks pass.
        errors = mhc_fused_ops_field_errors(config) or mhc_fused_ops_errors(config)
        if errors:
            raise ValueError(
                "Fused mHC does not support this mhc_fused_ops selection: "
                + "; ".join(errors)
                + "."
            )
        from megatron.core.fusions import fused_mhc_kernels as fmk  # lazy: it imports this module

        policy = config.mhc_fused_backend
        # log_fused_mhc_backend_once would describe the fused kernels' choice for operations
        # this module does not bind, so only validate: a deferred MHC_FORCE_BACKEND error, an
        # unknown policy or an explicit policy whose backend is unavailable still raises here.
        # Read the fused kernels' state through the module, so that tests can patch it.
        fmk._validate_mhc_backend(policy)
        if fmk._MHC_FORCED_BACKEND == "native":
            return {op: "native(MHC_FORCE_BACKEND=native)" for op in config.mhc_fused_ops}
        if policy == "native":
            return {op: "native(mhc_fused_backend=native)" for op in config.mhc_fused_ops}

        labels, unavailable = {}, []
        fmk_labels = _fmk_op_labels(policy, MHC_FUSED_OPS)
        for op in MHC_FUSED_OPS:
            if op not in config.mhc_fused_ops:
                continue
            if op == "sinkhorn":
                # Triton-only. The policy is auto or triton here: native returned above, the
                # table rejects cutile, and triton without Triton raised in _validate_mhc_backend.
                if fmk.is_triton_available():
                    # Lazy: the file imports fused_mhc_kernels, which imports this module.
                    from megatron.core.fusions import fused_mhc_glm_projection as glm

                    self._sinkhorn_op = glm.fixed_launch_sinkhorn
                    labels[op] = f"triton(num_warps={glm.SINKHORN_NUM_WARPS})"
                else:
                    labels[op] = "native(triton unavailable)"
                    unavailable.append(op)
                continue
            if op in triton_only_ops(config):
                # proj_rms with mhc_norm_eps_inside_sqrt and mhc_normalize_before_projection
                # (sinkhorn continued above): the Triton GLM projection pair, Triton-only like the
                # Sinkhorn. The policy is auto or triton here.
                if fmk.is_triton_available():
                    # Lazy: the file imports fused_mhc_kernels, which imports this module.
                    from megatron.core.fusions import fused_mhc_glm_projection as glm

                    self._glm_proj_rms_op = glm.glm_proj_rms_op(config.mhc_fused_tf32_rounding)
                    labels[op] = self._glm_proj_rms_op.label
                else:
                    labels[op] = "native(triton unavailable)"
                    unavailable.append(op)
                continue
            if op == "proj_rms":
                self._proj_rms_compute_h_op = partial(fmk.fused_proj_rms_compute_h, backend=policy)
            elif op == "h_aggregate":
                self._h_aggregate_op = partial(fmk.fused_h_aggregate, backend=policy)
                self._h_aggregate_into_op = partial(fmk.fused_h_aggregate_into, backend=policy)
            else:  # h_post_bda
                self._h_post_bda_op = partial(fmk.fused_h_post_bda, backend=policy)
            labels[op] = fmk_labels[op]

        if unavailable:
            _warn_per_op_fallback_once(unavailable, "Triton is unavailable")
        return labels

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # Honor the mcore convention: skip weight init when the caller will load a
        # checkpoint over these parameters anyway (e.g. meta-device construction).
        if self.config.perform_initialization:
            nn.init.xavier_uniform_(self.mapping_proj.weight)

        # Set sequence_parallel attribute on parameters for gradient synchronization
        # across TP ranks when sequence_parallel is enabled.
        # This is required because HyperConnectionModule uses non-TP-aware layers
        # (nn.Linear, nn.RMSNorm) whose gradients need to be all-reduced.
        if self.config.sequence_parallel:
            setattr(self.mapping_proj.weight, 'sequence_parallel', True)
            setattr(self.alpha_pre, 'sequence_parallel', True)
            setattr(self.alpha_post, 'sequence_parallel', True)
            setattr(self.alpha_res, 'sequence_parallel', True)
            setattr(self.bias, 'sequence_parallel', True)

    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Projection + RMS normalization.

        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        # Keep the activation-dtype input in autograd and upcast inside the recomputed
        # projection. This avoids retaining a full FP32 copy of the residual stream.
        x_2d = x.reshape(s * b, nC)
        # mapping_proj is read here, not called, so its forward pre-hook never publishes it.
        ensure_params_ready([self.mapping_proj.weight])
        weight = self.mapping_proj.weight
        glm_op = self._glm_proj_rms_op
        if glm_op is not None:
            # Before the checkpoint below, so that a full-recompute forward under no_grad and its
            # replay run the same op; asked through the attribute, so that the spy counts refusals.
            if glm_op.supported(x_2d, weight):
                proj, r = glm_op(x_2d, weight, self.norm_eps)
                return proj.view(s, b, proj.shape[-1]), r.view(s, b, 1)
            self._refuse_glm_proj_rms(glm_op.refusal(x_2d, weight))
        if torch.is_grad_enabled() and (x_2d.requires_grad or weight.requires_grad):
            proj, r = checkpoint(
                self._proj_rms_op,
                x_2d,
                weight,
                self.norm_eps,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            proj, r = self._proj_rms_op(x_2d, weight, self.norm_eps)
        return proj.view(s, b, proj.shape[-1]), r.view(s, b, 1)

    def _refuse_glm_proj_rms(self, reason: str) -> None:
        """An explicit triton policy fails; auto warns once per reason and runs the unfused
        projection."""
        message = f"the Triton GLM projection pair does not support this input: {reason}"
        if self.config.mhc_fused_backend == "triton":
            raise RuntimeError(
                f"mHC fused backend 'triton' was requested for ['proj_rms'], but {message}."
            )
        _warn_per_op_fallback_once(["proj_rms"], message)

    # Static shapes: see _compute_h_one_specialization.
    @torch.compile(dynamic=False)
    def _compute_h(self, proj: Tensor, r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute h from projected hidden states and scaling factors.

        Args:
            proj: [s, b, n^2 + 2n] - projected hidden states
            r: [s, b, 1] - scaling factors

        Returns:
            h_pre: [s, b, n] - aggregation weights
            h_post: [s, b, n] - expansion weights
            h_res: [s, b, n^2] - residual mixing logits
        """
        alpha_ = torch.cat(
            [
                self.alpha_pre.expand(self.n),
                self.alpha_post.expand(self.n),
                self.alpha_res.expand(self.n * self.n),
            ],
            dim=-1,
        )

        # A normalize-before-projection GEMM already consumed the RMS scaling.
        if self.config.mhc_normalize_before_projection:
            h = proj * alpha_ + self.bias
        else:
            h = r * proj * alpha_ + self.bias
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., : self.n].sigmoid() + self.compute_h_eps  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n : 2 * self.n].sigmoid() * 2
        h_res = h[..., 2 * self.n :]
        return h_pre, h_post, h_res

    def _compute_h_one_specialization(
        self, proj: Tensor, r: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run the compiled ``_compute_h`` in its grad-mode specialization only.

        Dynamo specializes ``_compute_h`` on grad mode and on whether its inputs
        require grad. Full activation recompute runs the forward under no_grad and
        replays it with gradients enabled; with two specializations, a full Dynamo
        cache makes the missing one run eagerly, and the replay no longer reproduces
        the forward's mappings bit for bit. Both halves therefore run the grad-mode
        specialization on inputs that require grad. A grad-mode call that needs a
        gradient runs it directly, so autograd uses its compiled backward; an input
        that does not require grad is replaced by a detached one that does. Every
        other call, under no_grad or with nothing to differentiate, runs it under
        enable_grad on detached inputs and returns detached outputs.

        ``_compute_h`` is compiled with static shapes because a dynamic-shape entry
        compiled from the view inputs of a grad-mode call guards on their view base,
        which detached inputs lack: the two halves of one recompute would then use
        different entries. With static shapes they share the entry for their shape,
        or, once Dynamo's cache is full, both run eagerly.
        """
        params = (self.alpha_pre, self.alpha_post, self.alpha_res, self.bias)
        if torch.is_grad_enabled() and (
            proj.requires_grad or r.requires_grad or any(p.requires_grad for p in params)
        ):
            if not proj.requires_grad:
                proj = proj.detach().requires_grad_(True)
            if not r.requires_grad:
                r = r.detach().requires_grad_(True)
            return self._compute_h(proj, r)
        with torch.enable_grad():
            outputs = self._compute_h(
                proj.detach().requires_grad_(True), r.detach().requires_grad_(True)
            )
        return tuple(output.detach() for output in outputs)

    @nvtx_decorator(message="HyperConnection::compute_mappings")
    def compute_mappings(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute mHC mappings from input hidden states.

        Reference: Eq. (5) and (8) in mHC paper

        Args:
            x: [s, b, n*C] - n-stream hidden states

        Returns:
            h_pre: [s, b, n] - aggregation weights (sigmoid activated)
            h_post: [s, b, n] - expansion weights (2*sigmoid activated)
            h_res: [s, b, n, n] - residual mixing matrix (doubly stochastic)
        """
        s, b, _ = x.shape

        if self._proj_rms_compute_h_op is not None:
            # Fused path: proj_rms + compute_h in one kernel launch sequence
            x_2d = x.reshape(s * b, self.n * self.hidden_size)
            # The fused kernel reads mapping_proj.weight without calling mapping_proj.
            ensure_params_ready([self.mapping_proj.weight])
            with torch.cuda.nvtx.range("HyperConnection::fused_proj_rms_compute_h"):
                h_pre, h_post, h_res, _ = self._proj_rms_compute_h_op(
                    x_2d,
                    self.mapping_proj.weight,
                    self.alpha_pre,
                    self.alpha_post,
                    self.alpha_res,
                    self.bias,
                    self.n,
                    self.norm_eps,
                    self.compute_h_eps,
                )
            h_pre = h_pre.view(s, b, self.n)
            h_post = h_post.view(s, b, self.n)
            h_res = h_res.view(s, b, self.n, self.n)
        else:
            # Native path: separate proj_rms + _compute_h
            with torch.cuda.nvtx.range("HyperConnection::projection_and_get_norm"):
                proj, r = self._projection_and_get_norm(x)
            with torch.cuda.nvtx.range("HyperConnection::compute_h"):
                h_pre, h_post, h_res = self._compute_h_one_specialization(proj, r)
            h_res = h_res.view(s, b, self.n, self.n)

        h_res = self._sinkhorn_op(
            h_res, self.sinkhorn_iterations, self.sinkhorn_eps
        )  # [s, b, n, n]

        dtype = torch.float32 if self.config.mhc_keep_mappings_in_fp32 else x.dtype
        pre_dtype = torch.float32 if getattr(self.config, "mhc_keep_pre_in_fp32", False) else dtype
        return h_pre.to(pre_dtype), h_post.to(dtype), h_res.to(dtype)

    # dynamic=True handles the hybrid mHC variable-shape path (was blanket-disabled)
    @torch.compile
    def _apply_h_post(self, x: Tensor, h_post: Tensor) -> Tensor:
        """
        Core implementation of H_post application to a single tensor.

        Computes: H_post^T @ x

        Args:
            x: Input tensor, can be either:
               - [s, b, C] - standard hidden states
               - [C] - bias tensor (will be broadcast)
            h_post: [s, b, n] - expansion weights

        Returns:
            output: [s, b, n*C] - expanded tensor
        """
        n = self.n
        s, b, _ = h_post.shape

        if x.dim() == 1:
            # x is bias with shape [C], need to broadcast to [s, b, 1, C]
            C = x.shape[0]
            x_expanded = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(s, b, 1, C)
        else:
            # x is [s, b, C]
            C = x.shape[-1]
            x_expanded = x.unsqueeze(2)  # [s, b, 1, C]

        # h_post^T @ x : [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C]
        # Using broadcast multiply instead of einsum
        result = h_post.unsqueeze(-1) * x_expanded
        return result.view(s, b, n * C)

    @nvtx_decorator(message="HyperConnection::apply_h_post")
    def apply_h_post(
        self,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        h_post: Tensor,
        manager: Optional['MHCCheckpointManager'] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post to x and optionally bias, with optional checkpointing.

        This is the unified entry point that handles both normal execution
        and checkpoint-based execution for memory efficiency.

        Args:
            x_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - hidden states
                - bias: [C] or None - optional bias tensor
            h_post: [s, b, n] - expansion weights
            manager: Optional MHCCheckpointManager for checkpoint management.
                When provided, wraps _apply_h_post with CheckpointWithoutOutput.

        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [s, b, n*C] - expanded hidden states
                - bias_out: [s, b, n*C] or None - expanded bias if input bias was not None
        """
        x, bias = x_with_bias

        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            # Checkpoint _apply_h_post to discard the output
            x_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._apply_h_post, x, h_post
            )

            # Checkpoint _apply_h_post for bias if not None
            if bias is not None:
                bias_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                    self._apply_h_post, bias, h_post
                )
            else:
                bias_out = None
        else:
            # Normal execution without checkpoint
            x_out = self._apply_h_post(x, h_post)
            bias_out = self._apply_h_post(bias, h_post) if bias is not None else None

        return x_out, bias_out

    def aggregate(self, x: Tensor, h_pre: Tensor, out: Optional[Tensor] = None) -> Tensor:
        """
        Aggregate n-stream to 1-stream.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights

        Returns:
            aggregated: [s, b, C] - single stream hidden states
        """
        s, b, _ = x.shape
        C = self.hidden_size
        x_streams = x.view(s, b, self.n, C)
        if out is None:
            return self._h_aggregate_op(x_streams, h_pre)
        return self._h_aggregate_into_op(x_streams, h_pre, out)

    # dynamic=True handles the hybrid mHC variable-shape path (was blanket-disabled)
    @torch.compile
    def apply_h_res(self, h_res: Tensor, residual: Tensor) -> Tensor:
        """
        Apply H_res to residual using H_res weights.

        Computes: H_res.T @ residual

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            residual: [s, b, n*C] - n-stream hidden states
        """
        s, b, _ = residual.shape
        n = self.n
        C = self.hidden_size

        # [s, b, n*C] -> [s, b, n, C]
        residual_streams = residual.view(s, b, n, C).to(h_res.dtype)

        # FP32 multiply-adds, not a GEMM: [s, b, n, n].T @ [s, b, n, C] -> [s, b, n, C]
        mixed = _h_res_mix(h_res.view(s, b, n, n), residual_streams)

        return mixed.view(s, b, n * C)

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager: Optional['MHCCheckpointManager'] = None,
        output_slot: Optional['MHCRecomputeArenaSlot'] = None,
        return_residual: bool = False,
        *,
        checkpoint_fn: Callable[..., Tensor] | None = None,
    ) -> Tuple[Tensor, ...]:
        """
        Full mHC forward pass.

        Uses BroadcastTensorFused to split hidden_states into 3 autograd-graph
        children so that gradient accumulation from the 3 consumers
        (compute_mappings, aggregate, fused_h_res_h_post_bda) is handled by a
        single fused add instead of PyTorch's default sequential accumulation.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            mhc_recompute_manager: Optional MHCCheckpointManager for checkpoint management.
                When provided, uses _forward_with_checkpoint for memory-efficient execution.
            output_slot: Optional arena slot used as the aggregate kernel's
                caller-owned output for both forward and recompute. A differing
                slot dtype is handled by casting the aggregate into the slot.
                Requires mhc_recompute_manager.
            return_residual: Also return the residual view selected by BroadcastTensorFused.
            checkpoint_fn: Checkpoint callable for aggregation; mutually exclusive with
                mhc_recompute_manager and output_slot.

        Returns:
            A 3-tuple by default, or a 4-tuple when return_residual=True.
            Internal mHC consumers request the residual branch explicitly so
            fused_h_res_h_post_bda consumes the BroadcastTensorFused output.
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            residual: [s, b, n*C] - residual view for fused_h_res_h_post_bda
        """

        if mhc_recompute_manager is not None or checkpoint_fn is not None:
            result = self._forward_with_checkpoint(
                hidden_states,
                mhc_recompute_manager,
                output_slot=output_slot,
                checkpoint_fn=checkpoint_fn,
            )
        else:
            if output_slot is not None:
                raise ValueError("fixed mHC outputs require an mHC recompute manager")
            result = self._forward_normal(hidden_states)
        return result if return_residual else result[:3]

    def _forward_normal(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Normal forward pass without checkpointing.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            residual: [s, b, n*C] - residual view for fused_h_res_h_post_bda
        """
        # Split into 3 views to avoid extra grad accumulations in backward
        hs_for_mappings, hs_for_aggregate, hs_for_residual = BroadcastTensorFused.apply(
            hidden_states, self._fused_add_3_op
        )

        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hs_for_mappings)

        # Aggregate for layer input
        with torch.cuda.nvtx.range("HyperConnection::aggregate"):
            aggregated = self.aggregate(hs_for_aggregate, h_pre)

        return aggregated, h_res, h_post, hs_for_residual

    def _forward_with_checkpoint(
        self,
        hidden_states: Tensor,
        manager: Optional['MHCCheckpointManager'] = None,
        output_slot: Optional['MHCRecomputeArenaSlot'] = None,
        *,
        checkpoint_fn: Callable[..., Tensor] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass with checkpointing for memory efficiency.

        compute_mappings is called directly (not checkpointed) since its outputs
        (h_pre, h_post, h_res) are needed downstream. Only aggregate is wrapped with
        CheckpointWithoutOutput and auto-registered to the manager.
        apply_h_res is deferred to fused_h_res_h_post_bda for kernel fusion.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            manager: Optional MHCCheckpointManager for unified recomputation
            output_slot: Optional attention CUDA Graph input slot. Aggregation
                preserves the input dtype before casting into a differing slot dtype.
                Requires manager.
            checkpoint_fn: Checkpoint callable, mutually exclusive with manager and
                output_slot.

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            residual: [s, b, n*C] - residual view for fused_h_res_h_post_bda
        """
        checkpoint_fn = _resolve_mhc_checkpoint(manager, checkpoint_fn, output_slot=output_slot)

        # Split into 3 views to avoid extra grad accumulations in backward
        hs_for_mappings, hs_for_aggregate, hs_for_residual = BroadcastTensorFused.apply(
            hidden_states, self._fused_add_3_op
        )

        h_pre, h_post, h_res = self.compute_mappings(hs_for_mappings)

        def aggregate_function(x, h):
            # Both forward and recompute must populate the graph's captured input address.
            # Read writer per call so each invocation gets a fresh detached view.
            out = output_slot.writer
            if out.dtype == x.dtype:
                return self.aggregate(x, h, out=out)

            # FP32 residual streams can feed a lower-precision attention graph.
            # Preserve the aggregation precision and checkpoint the cast/copy too,
            # so recompute restores the captured bytes before attention backward.
            return out.copy_(self.aggregate(x, h))

        # Checkpoint aggregate - auto-registers to manager. Without an output slot, pass the
        # bound aggregate so a caller-supplied checkpoint_fn receives the owner method.
        aggregate_fn = self.aggregate if output_slot is None else aggregate_function
        aggregated = checkpoint_fn(aggregate_fn, hs_for_aggregate, h_pre)

        return aggregated, h_res, h_post, hs_for_residual

    # ==================== Block-level utilities ====================

    @staticmethod
    def input_expand(x: Tensor, n: int) -> Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.

        Simple replication strategy: each stream initialized as a copy of input.

        Args:
            x: [s, b, C] - single stream hidden states
            n: Number of residual streams

        Returns:
            expanded: [s, b, n*C] - n-stream hidden states
        """
        s, b, C = x.shape
        # Replicate input to n streams
        expanded = x.unsqueeze(2).expand(s, b, n, C).contiguous()
        return expanded.view(s, b, n * C)

    @staticmethod
    def output_contract(x: Tensor, n: int) -> Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.

        Simple averaging strategy: average all streams.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            n: Number of residual streams

        Returns:
            contracted: [s, b, C] - single stream hidden states
        """
        s, b, nC = x.shape
        C = nC // n
        # Average all streams. Accumulate the reduction in fp32 (then downcast to
        # the input dtype) to avoid bf16 accumulation drift in the final contracted stream.
        x_streams = x.view(s, b, n, C)
        contracted = x_streams.float().mean(dim=2).to(x.dtype)
        return contracted

    # ==================== Fused kernel placeholder ====================

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda")
    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional['MHCCheckpointManager'] = None,
        *,
        checkpoint_fn: Callable[..., Tensor] | None = None,
    ) -> Tensor:
        """
        Fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.

        This is a placeholder for future kernel fusion optimization.
        Currently implements the operations sequentially using native PyTorch.

        The computation flow is:
            1. mixed = H_res.T @ original_residual (apply_h_res)
            2. expanded = H_post^T @ layer_output (apply_h_post)
            3. output = dropout(expanded + bias) + mixed (bias-dropout-add)

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states (before H_res applied)
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - layer output (attention or MLP output)
                - bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: Optional MHCCheckpointManager for checkpoint management.
                When provided, each operation is wrapped with CheckpointWithoutOutput.
            checkpoint_fn: Optional output-discard checkpoint callable for this write.
                Mutually exclusive with manager; omit both for an eager or terminal write.

        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        if manager is not None or checkpoint_fn is not None:
            return self._fused_h_res_h_post_bda_with_checkpoint(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
                manager,
                checkpoint_fn=checkpoint_fn,
            )
        else:
            return self._fused_h_res_h_post_bda_native(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
            )

    def _fused_h_res_h_post_bda_native(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
    ) -> Tensor:
        """Run the shared write arithmetic eagerly."""
        x, bias = layer_output_with_bias
        return self._compute_write(
            h_res,
            original_residual,
            h_post,
            x,
            bias,
            dropout_prob=dropout_prob,
            training=training,
            fused=fused,
        )

    def _compute_write(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Optional[Tensor],
        *,
        dropout_prob: float,
        training: bool,
        fused: bool,
    ) -> Tensor:
        """
        Compute the residual update for eager execution and checkpoint replay.

        When dropout is zero (or inference), uses a single fused/reference kernel
        for H_res.T @ residual + H_post * (x + bias). Falls back to unfused
        implementation when dropout is needed.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            x: [s, b, C] - layer output
            bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation

        Returns:
            output: [s, b, n*C] - final output
        """
        if dropout_prob == 0.0 or not training:
            s, b, _ = original_residual.shape
            n = self.n
            C = self.hidden_size
            orig_reshaped = original_residual.view(s, b, n, C)
            output = self._h_post_bda_op(h_res, orig_reshaped, h_post, x, bias)
            return output.view(s, b, n * C)

        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        with torch.cuda.nvtx.range("HyperConnection::apply_h_res"):
            mixed = self.apply_h_res(h_res, original_residual)
        with torch.cuda.nvtx.range("HyperConnection::apply_h_post"):
            x_expanded = self._apply_h_post(x, h_post)
            bias_expanded = self._apply_h_post(bias, h_post) if bias is not None else None
        bda_func = get_bias_dropout_add(training, fused)
        with torch.cuda.nvtx.range("HyperConnection::bda"):
            output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
        return output.to(original_residual.dtype)

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda_with_checkpoint")
    def _fused_h_res_h_post_bda_with_checkpoint(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional['MHCCheckpointManager'] = None,
        *,
        checkpoint_fn: Callable[..., Tensor] | None = None,
    ) -> Tensor:
        """
        Checkpointed variant of _fused_h_res_h_post_bda_native.

        Wraps compute in CheckpointWithoutOutput for activation memory savings.
        Tensor operands are separate positional arguments so checkpointing tracks
        their gradients. Optional bias may be None; execution settings are bound
        for replay while eager execution uses the same numerical helper.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: Optional MHCCheckpointManager for checkpoint management
            checkpoint_fn: Checkpoint callable, mutually exclusive with manager.

        Returns:
            output: [s, b, n*C] - final output
        """
        checkpoint_fn = _resolve_mhc_checkpoint(manager, checkpoint_fn)

        x, bias = layer_output_with_bias
        return checkpoint_fn(
            partial(self._compute_write, dropout_prob=dropout_prob, training=training, fused=fused),
            h_res,
            original_residual,
            h_post,
            x,
            bias,
        )


@dataclass(frozen=True)
class HyperConnectionExecutionContext:
    """Current mHC parameter owner and BDA fusion setting for one residual operation.

    ``owner_residual_view`` makes an input-residual read carry the owner's residual view
    for this call only, for callers whose other execution path already carries that view.
    """

    owner: HyperConnectionModule
    bias_dropout_fusion: bool = False
    owner_residual_view: bool = False


class HyperConnectionResidualConnection(ResidualConnection):
    """Parameterless mHC read or write adapter using the owner supplied at each call.

    Reads carry the residual and predicted maps to the write. Set use_input_residual
    to preserve callers that carry the input instead of the owner's residual view.
    Owner dtypes are preserved; incompatible branch dtype requests are rejected.
    Output-discard replay covers aggregation and nonterminal writes only.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        operation: ResidualConnectionOperation,
        use_input_residual: bool = False,
    ) -> None:
        super().__init__(
            config.mhc_num_residual_streams * config.hidden_size,
            config.hidden_size,
            operation=operation,
        )
        if use_input_residual and operation != "read":
            raise ValueError("use_input_residual is only supported for mHC reads.")
        self.use_input_residual = use_input_residual

    @staticmethod
    def _execution_context(execution_context: object | None) -> HyperConnectionExecutionContext:
        if not isinstance(execution_context, HyperConnectionExecutionContext):
            raise TypeError("mHC residual operations require HyperConnectionExecutionContext.")
        return execution_context

    def _eager_read(
        self,
        hidden_states: Tensor,
        *,
        fp32_residual_connection: bool,
        branch_input_dtype: torch.dtype | None,
        execution_context: object | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Read eagerly, without registering output-discard replay."""

        # The owner chooses dtypes; the base validates any requested branch dtype.
        del fp32_residual_connection, branch_input_dtype
        return self._read_from_owner(
            hidden_states, self._execution_context(execution_context), checkpoint_fn=None
        )

    def _read_from_owner(
        self,
        hidden_states: Tensor,
        execution_context: HyperConnectionExecutionContext,
        *,
        checkpoint_fn: Callable[..., Tensor] | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Share owner invocation and state packing without choosing a replay policy."""

        if self.use_input_residual and not execution_context.owner_residual_view:
            # Preserve the dedicated Transformer's residual path and three-output owner hooks.
            residual = hidden_states
            branch_input, h_res, h_post = apply_module(execution_context.owner)(
                hidden_states, checkpoint_fn=checkpoint_fn
            )
        else:
            branch_input, h_res, h_post, residual = apply_module(execution_context.owner)(
                hidden_states, return_residual=True, checkpoint_fn=checkpoint_fn
            )
        return branch_input, (residual, h_res, h_post)

    def _eager_write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
        execution_context: object | None,
    ) -> Tensor:
        """Write eagerly, without registering output-discard replay."""

        return self._write_to_owner(
            branch_output,
            state,
            self._execution_context(execution_context),
            dropout_probability=dropout_probability,
            training=training,
            checkpoint_fn=None,
        )

    @staticmethod
    def _write_to_owner(
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        execution_context: HyperConnectionExecutionContext,
        *,
        dropout_probability: float,
        training: bool,
        checkpoint_fn: Callable[..., Tensor] | None,
    ) -> Tensor:
        """Invoke the owner's write without changing its output dtype."""

        if len(state) != 3:
            raise ValueError("mHC residual write requires the selected residual and both maps.")
        residual, h_res, h_post = state
        value = branch_output if isinstance(branch_output, tuple) else (branch_output, None)
        return execution_context.owner.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            value,
            dropout_prob=dropout_probability,
            training=training,
            fused=execution_context.bias_dropout_fusion,
            checkpoint_fn=checkpoint_fn,
        )

    def _checkpoint_read(
        self,
        hidden_states: Tensor,
        context: ResidualStreamRecomputeContext,
        *,
        fp32_residual_connection: bool,
        branch_input_dtype: torch.dtype | None,
        execution_context: object | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        # Let the existing mHC owner checkpoint aggregation rather than replaying prediction
        # and rebuilding the selected residual alias through the generic whole-read path.
        del fp32_residual_connection, branch_input_dtype
        return self._read_from_owner(
            hidden_states,
            self._execution_context(execution_context),
            checkpoint_fn=context.checkpoint,
        )

    def _checkpoint_write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        context: ResidualStreamRecomputeContext,
        *,
        dropout_probability: float,
        training: bool,
        execution_context: object | None,
    ) -> Tensor:
        """Replay nonterminal writes while retaining the block's terminal output."""

        return self._write_to_owner(
            branch_output,
            state,
            self._execution_context(execution_context),
            dropout_probability=dropout_probability,
            training=training,
            checkpoint_fn=None if context.is_block_end else context.checkpoint,
        )


# ==================== Checkpoint utilities for mHC ====================


class HyperConnectionCheckpoint:
    """
    Checkpoint utility for mHC intermediate activations.

    Implements the paper's "recomputing strategy" to reduce memory footprint
    by discarding intermediate n-stream activations and recomputing on-the-fly.
    """

    @staticmethod
    def compute_optimal_block_size(num_layers: int, num_streams: int) -> int:
        """
        Compute optimal recomputation block size.

        From paper Eq. (20): L_r^* ≈ sqrt(nL/(n+2))

        Args:
            num_layers: Total number of transformer layers
            num_streams: Number of residual streams (n)

        Returns:
            block_size: Optimal block size for checkpointing
        """
        block_size = int(math.sqrt(num_streams * num_layers / (num_streams + 2)))
        return max(1, block_size)
