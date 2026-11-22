# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-head matmul of token-major activations, without layout copies.

Absorbed MLA multiplies every head of a token-major activation ``x[..., n, d]`` by that head's
weight: the query absorption ``einsum("...nd,ndk->...nk", q, w_uk)`` and the value
up-projection ``einsum("...nc,ndc->...nd", o, w_uv)``. ``torch.einsum`` runs them as one
batched GEMM over the heads, so it first copies ``x`` to head-major ``[n, tokens, d]`` (``x`` is
often a slice of a wider projection) and returns a head-major result that the caller then
copies back to token-major. A strided-batched GEMM can read and write the token-major layout
directly: head ``h`` of every token is a matrix with row stride ``n * width`` and batch stride
``width``. :func:`head_batched_matmul` does that in the forward and the backward.
"""

from typing import Tuple

import torch
from torch import Tensor


def _token_major_bmm(a: Tensor, b: Tensor, out_width: int) -> Tensor:
    """``[tokens, n, d] x [n, d, k] -> [tokens, n, k]`` with the GEMM batched over ``n``.

    ``a`` may be any view whose last dimension is contiguous; ``b`` any view whose per-head
    matrices are row- or column-major. The output is a new contiguous token-major tensor.
    """
    out = torch.empty(a.size(0), a.size(1), out_width, dtype=a.dtype, device=a.device)
    torch.bmm(a.transpose(0, 1), b, out=out.transpose(0, 1))
    return out


class _HeadBatchedMatmul(torch.autograd.Function):
    """``out[t, h] = x[t, h] @ w[h]`` for token-major ``x`` ``[tokens, n, d]``, ``w`` ``[n, d, k]``."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
        """Strided-batched GEMM over the heads, reading and writing token-major rows."""
        ctx.save_for_backward(x, weight)
        return _token_major_bmm(x, weight, weight.size(-1))

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        """``dx[t, h] = dout[t, h] @ w[h]^T`` (token-major) and ``dw[h] = x[:, h]^T @ dout[:, h]``."""
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = None
        if grad_out.stride(-1) != 1:
            grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_x = _token_major_bmm(grad_out, weight.transpose(1, 2), weight.size(1))
        if ctx.needs_input_grad[1]:
            grad_weight = torch.bmm(x.permute(1, 2, 0), grad_out.transpose(0, 1))
        return grad_x, grad_weight


def head_batched_matmul(x: Tensor, weight: Tensor) -> Tensor:
    """``torch.einsum("...nd,ndk->...nk", x, weight)`` with no layout copy of ``x`` or the result.

    Args:
        x: token-major activation ``[..., n, d]``; its last dimension must be contiguous, so a
            slice of a wider projection (e.g. the no-PE part of the query) works as is.
        weight: per-head matrices ``[n, d, k]``, row- or column-major per head (a transposed
            view such as ``w.transpose(1, 2)`` works as is).

    Returns:
        A contiguous token-major ``[..., n, k]`` tensor. Falls back to ``torch.einsum`` for
        inputs a strided-batched GEMM cannot read in place (CPU tensors, or a last dimension
        that is not contiguous).
    """
    lead = x.shape[:-2]
    n, d = x.shape[-2:]
    if (
        not x.is_cuda
        or x.stride(-1) != 1
        or weight.dim() != 3
        or weight.size(0) != n
        or weight.size(1) != d
        or weight.dtype != x.dtype
    ):
        return torch.einsum("...nd,ndk->...nk", x, weight).contiguous()
    try:
        x_rows = x.view(-1, n, d)
    except RuntimeError:
        return torch.einsum("...nd,ndk->...nk", x, weight).contiguous()
    return _HeadBatchedMatmul.apply(x_rows, weight).view(*lead, n, weight.size(-1))
