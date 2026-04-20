# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attention Residual modules.

This file contains the standalone Full Attention Residual operator used by the
Attention Residuals reproduction. Megatron transformer layers use hidden-state
tensors shaped as [sequence, batch, hidden].
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@dataclass
class AttnResState:
    """Mutable AttnRes history for one transformer block forward."""

    history: list[Tensor] = field(default_factory=list)
    residual_type: Literal['full', 'block'] = 'full'
    num_sublayers: int | None = None
    num_blocks: int = 8
    completed_blocks: list[Tensor] = field(default_factory=list)
    partial_block: Tensor | None = None
    partial_block_count: int = 0
    sublayer_count: int = 0

    def append(self, value: Tensor) -> None:
        if self.residual_type == 'full':
            self.history.append(value)
            return

        if self.residual_type != 'block':
            raise ValueError(f"unsupported AttnRes residual_type: {self.residual_type}")

        self.partial_block = (
            value if self.partial_block is None else self.partial_block + value
        )
        self.partial_block_count += 1
        self.sublayer_count += 1

        if self.partial_block_count >= self.block_size:
            self.completed_blocks.append(self.partial_block)
            self.partial_block = None
            self.partial_block_count = 0

    @property
    def block_size(self) -> int:
        if self.num_sublayers is None:
            raise ValueError("num_sublayers is required for Block AttnRes")
        return max(1, (self.num_sublayers + self.num_blocks - 1) // self.num_blocks)

    def values(self) -> list[Tensor]:
        if self.residual_type == 'full':
            return self.history
        if self.residual_type != 'block':
            raise ValueError(f"unsupported AttnRes residual_type: {self.residual_type}")

        values = list(self.completed_blocks)
        if self.partial_block is not None:
            values.append(self.partial_block)
        return values

    @classmethod
    def full(cls, initial_value: Tensor) -> 'AttnResState':
        return cls(history=[initial_value], residual_type='full')

    @classmethod
    def block(
        cls, initial_value: Tensor, num_sublayers: int, num_blocks: int
    ) -> 'AttnResState':
        if num_sublayers <= 0:
            raise ValueError(f"num_sublayers must be positive, got {num_sublayers}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        return cls(
            history=[],
            residual_type='block',
            num_sublayers=num_sublayers,
            num_blocks=num_blocks,
            completed_blocks=[initial_value],
        )


def _full_attn_res_compute(
    values: Sequence[Tensor],
    query: Tensor,
    weight: Tensor,
    hidden_size: int,
    eps: float,
    use_rmsnorm: bool,
) -> Tensor:
    """Compute Full AttnRes with regular PyTorch ops."""

    reference = values[0]
    query = query.to(dtype=reference.dtype)
    weight = weight.to(dtype=reference.dtype)

    logits = []
    for value in values:
        if use_rmsnorm:
            variance = value.pow(2).mean(dim=-1, keepdim=True)
            key = value * torch.rsqrt(variance + eps) * weight
        else:
            key = value
        logits.append(torch.einsum("h,sbh->sb", query, key))

    weights = torch.softmax(torch.stack(logits, dim=0), dim=0).to(dtype=reference.dtype)
    output = torch.zeros_like(reference)
    for depth, value in enumerate(values):
        output = output + weights[depth].unsqueeze(-1) * value

    return output


if _TRITON_AVAILABLE:

    @triton.jit
    def _attn_res_reduce_kernel(
        value,
        query,
        weight,
        partial_dot,
        partial_sq,
        hidden_size: tl.constexpr,
        num_hidden_blocks: tl.constexpr,
        block_h: tl.constexpr,
        use_rmsnorm: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        hidden_block_id = tl.program_id(1)
        hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
        hidden_mask = hidden_offsets < hidden_size

        value_offsets = token_id * hidden_size + hidden_offsets
        value_tile = tl.load(value + value_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        query_tile = tl.load(query + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)

        if use_rmsnorm:
            weight_tile = tl.load(weight + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
            dot = tl.sum(value_tile * weight_tile * query_tile, axis=0)
            sq = tl.sum(value_tile * value_tile, axis=0)
        else:
            dot = tl.sum(value_tile * query_tile, axis=0)
            sq = 0.0

        partial_offset = token_id * num_hidden_blocks + hidden_block_id
        tl.store(partial_dot + partial_offset, dot)
        tl.store(partial_sq + partial_offset, sq)

    @triton.jit
    def _attn_res_accum_kernel(
        value,
        depth_weights,
        output,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        hidden_block_id = tl.program_id(1)
        hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
        hidden_mask = hidden_offsets < hidden_size

        value_offsets = token_id * hidden_size + hidden_offsets
        value_tile = tl.load(value + value_offsets, mask=hidden_mask, other=0.0)
        output_tile = tl.load(output + value_offsets, mask=hidden_mask, other=0.0)
        depth_weight = tl.load(depth_weights + token_id)

        output_tile += depth_weight * value_tile
        tl.store(output + value_offsets, output_tile, mask=hidden_mask)

    @triton.jit
    def _attn_res_backward_reduce_kernel(
        value,
        grad_output,
        query,
        weight,
        partial_dot,
        partial_sq,
        partial_gv,
        hidden_size: tl.constexpr,
        num_hidden_blocks: tl.constexpr,
        block_h: tl.constexpr,
        use_rmsnorm: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        hidden_block_id = tl.program_id(1)
        hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
        hidden_mask = hidden_offsets < hidden_size

        value_offsets = token_id * hidden_size + hidden_offsets
        value_tile = tl.load(value + value_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        grad_tile = tl.load(grad_output + value_offsets, mask=hidden_mask, other=0.0).to(
            tl.float32
        )
        query_tile = tl.load(query + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)

        if use_rmsnorm:
            weight_tile = tl.load(weight + hidden_offsets, mask=hidden_mask, other=0.0).to(
                tl.float32
            )
            dot = tl.sum(value_tile * weight_tile * query_tile, axis=0)
            sq = tl.sum(value_tile * value_tile, axis=0)
        else:
            dot = tl.sum(value_tile * query_tile, axis=0)
            sq = 0.0
        gv = tl.sum(value_tile * grad_tile, axis=0)

        partial_offset = token_id * num_hidden_blocks + hidden_block_id
        tl.store(partial_dot + partial_offset, dot)
        tl.store(partial_sq + partial_offset, sq)
        tl.store(partial_gv + partial_offset, gv)

    @triton.jit
    def _attn_res_backward_apply_kernel(
        value,
        grad_output,
        query,
        weight,
        alpha,
        beta,
        raw_dot,
        inv_rms,
        grad_value,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
        use_rmsnorm: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        hidden_block_id = tl.program_id(1)
        hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
        hidden_mask = hidden_offsets < hidden_size

        value_offsets = token_id * hidden_size + hidden_offsets
        value_tile = tl.load(value + value_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        grad_tile = tl.load(grad_output + value_offsets, mask=hidden_mask, other=0.0).to(
            tl.float32
        )
        query_tile = tl.load(query + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)

        alpha_scalar = tl.load(alpha + token_id).to(tl.float32)
        beta_scalar = tl.load(beta + token_id).to(tl.float32)

        if use_rmsnorm:
            weight_tile = tl.load(weight + hidden_offsets, mask=hidden_mask, other=0.0).to(
                tl.float32
            )
            raw_dot_scalar = tl.load(raw_dot + token_id).to(tl.float32)
            inv_scalar = tl.load(inv_rms + token_id).to(tl.float32)
            inv_cubed = inv_scalar * inv_scalar * inv_scalar
            grad_value_tile = alpha_scalar * grad_tile + beta_scalar * (
                inv_scalar * query_tile * weight_tile
                - raw_dot_scalar * inv_cubed * value_tile / hidden_size
            )
        else:
            grad_value_tile = alpha_scalar * grad_tile + beta_scalar * query_tile

        tl.store(grad_value + value_offsets, grad_value_tile, mask=hidden_mask)

    @triton.jit
    def _attn_res_backward_param_kernel(
        value,
        query,
        weight,
        beta,
        inv_rms,
        grad_query,
        grad_weight,
        num_tokens: tl.constexpr,
        hidden_size: tl.constexpr,
        block_t: tl.constexpr,
        block_h: tl.constexpr,
        use_rmsnorm: tl.constexpr,
    ):
        token_block_id = tl.program_id(0)
        hidden_block_id = tl.program_id(1)
        token_offsets = token_block_id * block_t + tl.arange(0, block_t)
        hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
        token_mask = token_offsets < num_tokens
        hidden_mask = hidden_offsets < hidden_size

        value_offsets = token_offsets[:, None] * hidden_size + hidden_offsets[None, :]
        mask = token_mask[:, None] & hidden_mask[None, :]
        value_tile = tl.load(value + value_offsets, mask=mask, other=0.0).to(tl.float32)
        beta_tile = tl.load(beta + token_offsets, mask=token_mask, other=0.0).to(tl.float32)

        if use_rmsnorm:
            inv_tile = tl.load(inv_rms + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
            scale = beta_tile * inv_tile
            query_tile = tl.load(query + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
            weight_tile = tl.load(weight + hidden_offsets, mask=hidden_mask, other=0.0).to(
                tl.float32
            )
            grad_query_tile = tl.sum(scale[:, None] * value_tile * weight_tile[None, :], axis=0)
            grad_weight_tile = tl.sum(scale[:, None] * value_tile * query_tile[None, :], axis=0)
            tl.atomic_add(grad_weight + hidden_offsets, grad_weight_tile, sem="relaxed", mask=hidden_mask)
        else:
            grad_query_tile = tl.sum(beta_tile[:, None] * value_tile, axis=0)

        tl.atomic_add(grad_query + hidden_offsets, grad_query_tile, sem="relaxed", mask=hidden_mask)


def _full_attn_res_triton_forward(
    values: Sequence[Tensor],
    query: Tensor,
    weight: Tensor,
    hidden_size: int,
    eps: float,
    use_rmsnorm: bool,
) -> Tensor:
    """Compute Full AttnRes forward with Triton kernels when possible."""

    if not _TRITON_AVAILABLE or not values[0].is_cuda:
        return _full_attn_res_compute(values, query, weight, hidden_size, eps, use_rmsnorm)
    if any(not value.is_contiguous() for value in values):
        return _full_attn_res_compute(values, query, weight, hidden_size, eps, use_rmsnorm)

    reference = values[0]
    sequence_length, batch_size, local_hidden_size = reference.shape
    if local_hidden_size != hidden_size:
        raise ValueError(f"expected hidden size {hidden_size}, got {local_hidden_size}")

    num_tokens = sequence_length * batch_size
    num_depths = len(values)
    block_h = 1024
    num_hidden_blocks = triton.cdiv(hidden_size, block_h)

    query_work = query.to(device=reference.device, dtype=reference.dtype).contiguous()
    weight_work = weight.to(device=reference.device, dtype=reference.dtype).contiguous()
    partial_dot = torch.empty(
        (num_depths, num_tokens, num_hidden_blocks), device=reference.device, dtype=torch.float32
    )
    partial_sq = torch.empty_like(partial_dot)

    grid = (num_tokens, num_hidden_blocks)
    for depth, value in enumerate(values):
        value_2d = value.reshape(num_tokens, hidden_size)
        _attn_res_reduce_kernel[grid](
            value_2d,
            query_work,
            weight_work,
            partial_dot[depth],
            partial_sq[depth],
            hidden_size,
            num_hidden_blocks,
            block_h,
            use_rmsnorm,
        )

    dot = partial_dot.sum(dim=-1)
    if use_rmsnorm:
        inv_rms = torch.rsqrt(partial_sq.sum(dim=-1) / hidden_size + eps)
        logits = dot * inv_rms
    else:
        logits = dot

    depth_weights = torch.softmax(logits, dim=0).to(dtype=reference.dtype)
    output = torch.zeros((num_tokens, hidden_size), device=reference.device, dtype=reference.dtype)

    for depth, value in enumerate(values):
        value_2d = value.reshape(num_tokens, hidden_size)
        _attn_res_accum_kernel[grid](
            value_2d,
            depth_weights[depth].contiguous(),
            output,
            hidden_size,
            block_h,
        )

    return output.reshape(sequence_length, batch_size, hidden_size)


def _full_attn_res_triton_backward(
    grad_output: Tensor,
    values: Sequence[Tensor],
    query: Tensor,
    weight: Tensor,
    hidden_size: int,
    eps: float,
    use_rmsnorm: bool,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Compute Full AttnRes backward gradients with Triton kernels."""

    reference = values[0]
    sequence_length, batch_size, local_hidden_size = reference.shape
    if local_hidden_size != hidden_size:
        raise ValueError(f"expected hidden size {hidden_size}, got {local_hidden_size}")

    num_tokens = sequence_length * batch_size
    num_depths = len(values)
    block_h = 1024
    num_hidden_blocks = triton.cdiv(hidden_size, block_h)

    grad_output_2d = grad_output.contiguous().reshape(num_tokens, hidden_size)
    query_work = query.to(device=reference.device, dtype=reference.dtype).contiguous()
    weight_work = weight.to(device=reference.device, dtype=reference.dtype).contiguous()
    partial_dot = torch.empty(
        (num_depths, num_tokens, num_hidden_blocks), device=reference.device, dtype=torch.float32
    )
    partial_sq = torch.empty_like(partial_dot)
    partial_gv = torch.empty_like(partial_dot)

    grid = (num_tokens, num_hidden_blocks)
    for depth, value in enumerate(values):
        value_2d = value.reshape(num_tokens, hidden_size)
        _attn_res_backward_reduce_kernel[grid](
            value_2d,
            grad_output_2d,
            query_work,
            weight_work,
            partial_dot[depth],
            partial_sq[depth],
            partial_gv[depth],
            hidden_size,
            num_hidden_blocks,
            block_h,
            use_rmsnorm,
        )

    raw_dot = partial_dot.sum(dim=-1)
    if use_rmsnorm:
        inv_rms = torch.rsqrt(partial_sq.sum(dim=-1) / hidden_size + eps)
        logits = raw_dot * inv_rms
    else:
        inv_rms = torch.empty_like(raw_dot)
        logits = raw_dot

    alpha = torch.softmax(logits, dim=0).to(dtype=reference.dtype)
    gv = partial_gv.sum(dim=-1)
    grad_output_dot_output = (alpha.float() * gv).sum(dim=0, keepdim=True)
    beta = (alpha.float() * (gv - grad_output_dot_output)).to(dtype=reference.dtype)

    grad_query = torch.zeros_like(query, dtype=torch.float32)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    grad_values = []
    param_block_t = 16
    param_block_h = 128
    param_grid = (triton.cdiv(num_tokens, param_block_t), triton.cdiv(hidden_size, param_block_h))

    for depth, value in enumerate(values):
        value_2d = value.reshape(num_tokens, hidden_size)
        grad_value_2d = torch.empty_like(value_2d)
        _attn_res_backward_apply_kernel[grid](
            value_2d,
            grad_output_2d,
            query_work,
            weight_work,
            alpha[depth].contiguous(),
            beta[depth].contiguous(),
            raw_dot[depth].contiguous(),
            inv_rms[depth].contiguous(),
            grad_value_2d,
            hidden_size,
            block_h,
            use_rmsnorm,
        )
        _attn_res_backward_param_kernel[param_grid](
            value_2d,
            query_work,
            weight_work,
            beta[depth].contiguous(),
            inv_rms[depth].contiguous(),
            grad_query,
            grad_weight,
            num_tokens,
            hidden_size,
            param_block_t,
            param_block_h,
            use_rmsnorm,
        )
        grad_values.append(grad_value_2d.reshape_as(value))

    if not use_rmsnorm:
        grad_weight.zero_()

    return grad_query.to(dtype=query.dtype), grad_weight.to(dtype=weight.dtype), grad_values


class _FullAttnResCheckpointed(torch.autograd.Function):
    """Full AttnRes with custom backward recomputation.

    This keeps the Python/PyTorch implementation numerically aligned with the
    default path, but avoids saving depth logits, softmax weights, and RMSNorm
    intermediates from forward. Backward recomputes them under autograd.
    """

    @staticmethod
    def forward(ctx, query, weight, hidden_size, eps, use_rmsnorm, *values):
        ctx.hidden_size = hidden_size
        ctx.eps = eps
        ctx.use_rmsnorm = use_rmsnorm
        ctx.save_for_backward(query, weight, *values)
        with torch.no_grad():
            return _full_attn_res_compute(values, query, weight, hidden_size, eps, use_rmsnorm)

    @staticmethod
    def backward(ctx, grad_output):
        query, weight, *values = ctx.saved_tensors

        query_recompute = query.detach().requires_grad_(True)
        weight_recompute = weight.detach().requires_grad_(True)
        values_recompute = [value.detach().requires_grad_(True) for value in values]

        with torch.enable_grad():
            output = _full_attn_res_compute(
                values_recompute,
                query_recompute,
                weight_recompute,
                ctx.hidden_size,
                ctx.eps,
                ctx.use_rmsnorm,
            )

        grads = torch.autograd.grad(
            output,
            (query_recompute, weight_recompute, *values_recompute),
            grad_output,
            allow_unused=True,
        )

        query_grad, weight_grad, *value_grads = grads
        if query_grad is None:
            query_grad = torch.zeros_like(query)
        if weight_grad is None:
            weight_grad = torch.zeros_like(weight)
        value_grads = [
            torch.zeros_like(value) if grad is None else grad
            for value, grad in zip(values, value_grads)
        ]
        return query_grad, weight_grad, None, None, None, *value_grads


class _FullAttnResTritonCheckpointed(torch.autograd.Function):
    """Triton forward with checkpointed PyTorch backward recomputation."""

    @staticmethod
    def forward(ctx, query, weight, hidden_size, eps, use_rmsnorm, *values):
        ctx.hidden_size = hidden_size
        ctx.eps = eps
        ctx.use_rmsnorm = use_rmsnorm
        ctx.save_for_backward(query, weight, *values)
        with torch.no_grad():
            return _full_attn_res_triton_forward(
                values, query, weight, hidden_size, eps, use_rmsnorm
            )

    @staticmethod
    def backward(ctx, grad_output):
        query, weight, *values = ctx.saved_tensors

        query_recompute = query.detach().requires_grad_(True)
        weight_recompute = weight.detach().requires_grad_(True)
        values_recompute = [value.detach().requires_grad_(True) for value in values]

        with torch.enable_grad():
            output = _full_attn_res_compute(
                values_recompute,
                query_recompute,
                weight_recompute,
                ctx.hidden_size,
                ctx.eps,
                ctx.use_rmsnorm,
            )

        grads = torch.autograd.grad(
            output,
            (query_recompute, weight_recompute, *values_recompute),
            grad_output,
            allow_unused=True,
        )

        query_grad, weight_grad, *value_grads = grads
        if query_grad is None:
            query_grad = torch.zeros_like(query)
        if weight_grad is None:
            weight_grad = torch.zeros_like(weight)
        value_grads = [
            torch.zeros_like(value) if grad is None else grad
            for value, grad in zip(values, value_grads)
        ]
        return query_grad, weight_grad, None, None, None, *value_grads


class _FullAttnResTritonBackward(torch.autograd.Function):
    """Triton forward and Triton backward recomputation."""

    @staticmethod
    def forward(ctx, query, weight, hidden_size, eps, use_rmsnorm, *values):
        ctx.hidden_size = hidden_size
        ctx.eps = eps
        ctx.use_rmsnorm = use_rmsnorm
        ctx.save_for_backward(query, weight, *values)
        with torch.no_grad():
            return _full_attn_res_triton_forward(
                values, query, weight, hidden_size, eps, use_rmsnorm
            )

    @staticmethod
    def backward(ctx, grad_output):
        query, weight, *values = ctx.saved_tensors
        grad_query, grad_weight, grad_values = _full_attn_res_triton_backward(
            grad_output,
            values,
            query,
            weight,
            ctx.hidden_size,
            ctx.eps,
            ctx.use_rmsnorm,
        )
        return grad_query, grad_weight, None, None, None, *grad_values


class FullAttnRes(torch.nn.Module):
    """Depth-wise attention over previous residual-producing values.

    Args:
        hidden_size: Size of the hidden dimension.
        eps: Epsilon used by the internal RMSNorm over keys.

    Inputs are previous values shaped [S, B, H]. The module stacks them as
    [N, S, B, H], applies RMSNorm to produce keys, scores each depth position
    with one learned pseudo-query, and returns a weighted sum shaped [S, B, H].
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        use_rmsnorm: bool = True,
        implementation: Literal['torch', 'checkpointed', 'triton', 'triton_bwd'] = 'torch',
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if implementation not in ('torch', 'checkpointed', 'triton', 'triton_bwd'):
            raise ValueError(
                "FullAttnRes implementation must be 'torch', 'checkpointed', "
                "'triton', or 'triton_bwd', "
                f"got {implementation}"
            )

        self.hidden_size = hidden_size
        self.eps = eps
        self.use_rmsnorm = use_rmsnorm
        self.implementation = implementation
        self.query = torch.nn.Parameter(torch.zeros(hidden_size))
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def _rms_norm(self, values: Tensor) -> Tensor:
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return normalized * self.weight.to(dtype=values.dtype)

    def forward(self, values: Sequence[Tensor]) -> Tensor:
        """Apply Full Attention Residuals to previous values.

        Args:
            values: Non-empty sequence of tensors shaped [S, B, H].

        Returns:
            Tensor shaped [S, B, H], with dtype matching the first value.
        """

        if len(values) == 0:
            raise ValueError("FullAttnRes requires at least one value")

        reference = values[0]
        if reference.dim() != 3:
            raise ValueError(
                "FullAttnRes values must have shape [sequence, batch, hidden], "
                f"got {tuple(reference.shape)}"
            )
        if reference.size(-1) != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {reference.size(-1)}"
            )

        for index, value in enumerate(values[1:], start=1):
            if value.shape != reference.shape:
                raise ValueError(
                    "all FullAttnRes values must have the same shape; "
                    f"value 0 has {tuple(reference.shape)}, value {index} has {tuple(value.shape)}"
                )

        if self.implementation == 'checkpointed' and torch.is_grad_enabled():
            return _FullAttnResCheckpointed.apply(
                self.query,
                self.weight,
                self.hidden_size,
                self.eps,
                self.use_rmsnorm,
                *values,
            )
        if self.implementation == 'triton' and torch.is_grad_enabled():
            return _FullAttnResTritonCheckpointed.apply(
                self.query,
                self.weight,
                self.hidden_size,
                self.eps,
                self.use_rmsnorm,
                *values,
            )
        if self.implementation == 'triton_bwd' and torch.is_grad_enabled():
            if (
                _TRITON_AVAILABLE
                and reference.is_cuda
                and all(value.is_contiguous() for value in values)
            ):
                return _FullAttnResTritonBackward.apply(
                    self.query,
                    self.weight,
                    self.hidden_size,
                    self.eps,
                    self.use_rmsnorm,
                    *values,
                )
            return _FullAttnResCheckpointed.apply(
                self.query,
                self.weight,
                self.hidden_size,
                self.eps,
                self.use_rmsnorm,
                *values,
            )

        return _full_attn_res_compute(
            values,
            self.query,
            self.weight,
            self.hidden_size,
            self.eps,
            self.use_rmsnorm,
        )
