# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from .tilelang_sparse_mla_bwd import HAVE_TILELANG as HAVE_TILELANG_SPARSE_MLA_BWD
from .tilelang_sparse_mla_bwd import sparse_mla_bwd, sparse_mla_delta
from .tilelang_sparse_mla_fwd import HAVE_TILELANG as HAVE_TILELANG_SPARSE_MLA_FWD
from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

HAVE_TILELANG_SPARSE_MLA = HAVE_TILELANG_SPARSE_MLA_BWD and HAVE_TILELANG_SPARSE_MLA_FWD


def _valid_head_mask(indices, num_heads):
    valid_groups = indices.ge(0).any(dim=-1)
    kv_group = valid_groups.size(1)
    if kv_group == num_heads:
        return valid_groups
    if num_heads % kv_group != 0:
        raise RuntimeError(
            f"SparseMLA heads must be divisible by kv_group, got heads={num_heads}, "
            f"kv_group={kv_group}"
        )
    return valid_groups.repeat_interleave(num_heads // kv_group, dim=1)


def _zero_invalid_heads(tensor, valid_heads):
    zero = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    return torch.where(valid_heads.unsqueeze(-1), tensor, zero)


class SparseMLA(torch.autograd.Function):
    """Autograd wrapper around tilelang sparse-MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        """
        Args:
            q: Query tensor (seq_len, heads, dim_plus_tail_dim)
            kv: Key-Value tensor (seq_len_kv, kv_group, dim_plus_tail_dim)
            indices: Sparse indices tensor (seq_len, kv_group, topk)

        Returns:
            out: Output tensor (seq_len, heads, dim)
        """
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        valid_heads = _valid_head_mask(indices, q.size(1))
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
        tl_out = _zero_invalid_heads(tl_out, valid_heads)
        lse_zero = torch.zeros((), dtype=tl_lse.dtype, device=tl_lse.device)
        tl_lse = torch.where(valid_heads, tl_lse, lse_zero)

        # Do not save tl_out/tl_lse: backward recomputes them just long enough to form
        # delta and run the kernel. Saved inputs still go through autograd's saved-tensor
        # hooks/offload path and retain_graph can recompute these tensors again.
        ctx.save_for_backward(q, kv, indices, valid_heads)

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        q, kv, indices, valid_heads = ctx.saved_tensors
        scaling = ctx.scaling
        grad_output = grad_output.contiguous()
        grad_output = _zero_invalid_heads(grad_output, valid_heads)
        with torch.no_grad():
            tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
            tl_out = _zero_invalid_heads(tl_out, valid_heads)
            lse_zero = torch.zeros((), dtype=tl_lse.dtype, device=tl_lse.device)
            tl_lse = torch.where(valid_heads, tl_lse, lse_zero)
        delta = sparse_mla_delta(tl_out, grad_output)
        del tl_out

        tl_dq, tl_dkv = sparse_mla_bwd(
            q, kv, None, grad_output, indices, tl_lse, sm_scale=scaling, delta=delta
        )
        tl_dq = _zero_invalid_heads(tl_dq, valid_heads)
        del tl_lse

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None


if not HAVE_TILELANG_SPARSE_MLA:
    SparseMLA = None
