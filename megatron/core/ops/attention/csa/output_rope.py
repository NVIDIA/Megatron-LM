# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA output rotation and alias-safe backward reconstruction.

The attention Function owns and rotates its output in place. Backward reconstructs
both the raw output and its gradient into private storage. Reconstruction trades
one additional low-precision rotation for not retaining a second output activation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    _autotuned_mla_rope_bwd_inplace_kernel,
    _autotuned_mla_rope_fwd_inplace_kernel,
    _get_thd_token_idx,
)


@triton.jit
def _mla_rope_reconstruct_output_kernel(
    DO,
    OUT,
    COS,
    SIN,
    nope_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    position_ids,
    stride_x_seq,
    stride_x_nheads,
    stride_cos_seq,
    stride_sin_seq,
    cp_rank,
    cp_size,
    INVERSE: tl.constexpr,
    REMOVE_INTERLEAVING: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
):
    """Undo the output rotation and copy its non-rotary prefix into OUT."""
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if position_ids is not None:
        token_idx = tl.load(position_ids + pid_m)
    elif cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * stride_cos_seq + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * stride_sin_seq + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(
        COS + token_idx * stride_cos_seq + emb_dim // 2 + tl.arange(0, emb_dim // 2)
    )
    sin_right = tl.load(
        SIN + token_idx * stride_sin_seq + emb_dim // 2 + tl.arange(0, emb_dim // 2)
    )
    if INVERSE:
        sin_left = -sin_left
        sin_right = -sin_right
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    row_off = pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
    DO = DO + row_off
    OUT = OUT + row_off

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + nope_dim
    mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    if REMOVE_INTERLEAVING:
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        x_left = tl.load(DO + x_1_off, mask=mask)
        x_right = tl.load(DO + x_2_off, mask=mask)
    else:
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(DO + x_left_off, mask=mask)
        x_right = tl.load(DO + x_right_off, mask=mask)
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1

    x_1 = x_left * cos_left + x_right * sin_right
    x_2 = -x_left * sin_left + x_right * cos_right

    if not REMOVE_INTERLEAVING:
        # The split input and interleaved output layouts alias. Finish all loads
        # before any warp stores to the overlapping destination addresses.
        tl.debug_barrier()
    tl.store(OUT + x_1_off, x_1, mask=mask)
    tl.store(OUT + x_2_off, x_2, mask=mask)

    # Preserve the non-rotary prefix in the same launch as output reconstruction.
    nope_idx = tl.arange(0, BLOCK_NOPE)[None, :]
    nope_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + nope_idx
    nope_mask = mask & (nope_idx < nope_dim)
    tl.store(OUT + nope_off, tl.load(DO + nope_off, mask=nope_mask), mask=nope_mask)


_autotuned_mla_rope_reconstruct_output_kernel = triton.autotune(
    configs=[triton.Config({"BLOCK_H": heads}) for heads in (1, 2, 4, 8, 16, 32)],
    key=["emb_dim", "head_num", "nope_dim"],
)(_mla_rope_reconstruct_output_kernel)


def _raw_output_rope(
    t,
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
    inverse=False,
    remove_interleaving=False,
    position_ids=None,
    undo=False,
):
    if cu_seqlens_q is None:
        _, batch_size, nheads, head_dim = t.shape
        x = t.view(-1, nheads, head_dim)
        seq_num = None
    else:
        x = t
        batch_size = None
        seq_num = cu_seqlens_q.numel() - 1
    if x.stride(-1) != 1 or x.shape[-1] != nope_dim + emb_dim:
        raise ValueError('Output RoPE requires contiguous head dimensions.')
    kernel = (
        _autotuned_mla_rope_bwd_inplace_kernel if undo else _autotuned_mla_rope_fwd_inplace_kernel
    )
    grid = lambda meta: (x.shape[0], triton.cdiv(x.shape[1], meta['BLOCK_H']))
    kernel[grid](
        x,
        cos,
        sin,
        nope_dim,
        emb_dim,
        x.shape[1],
        batch_size,
        seq_num,
        cu_seqlens_q,
        position_ids,
        x.stride(0),
        x.stride(1),
        cos.stride(0),
        sin.stride(0),
        cp_rank,
        cp_size,
        INVERSE=inverse,
        REMOVE_INTERLEAVING=remove_interleaving,
    )
    return t


def mla_rope_apply_raw_(t, cos, sin, nope_dim, emb_dim, **kwargs):
    """Rotate an attention-owned output without adding an autograd node."""
    return _raw_output_rope(t, cos, sin, nope_dim, emb_dim, **kwargs)


def mla_rope_unapply_raw(
    t,
    cos,
    sin,
    nope_dim,
    emb_dim,
    out=None,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
    inverse=False,
    remove_interleaving=False,
    position_ids=None,
):
    """Undo output RoPE, fusing non-rotary copy when rebuilding saved output."""
    if out is None:
        return _raw_output_rope(
            t,
            cos,
            sin,
            nope_dim,
            emb_dim,
            cu_seqlens_q=cu_seqlens_q,
            cp_rank=cp_rank,
            cp_size=cp_size,
            inverse=inverse,
            remove_interleaving=remove_interleaving,
            position_ids=position_ids,
            undo=True,
        )
    if out.shape != t.shape or out.stride() != t.stride() or out.dtype != t.dtype:
        raise ValueError('Output reconstruction requires matching tensor metadata.')
    if cu_seqlens_q is None:
        _, batch_size, nheads, head_dim = t.shape
        x, y = t.view(-1, nheads, head_dim), out.view(-1, nheads, head_dim)
        seq_num = None
    else:
        x, y = t, out
        batch_size, seq_num = None, cu_seqlens_q.numel() - 1
    grid = lambda meta: (x.shape[0], triton.cdiv(x.shape[1], meta["BLOCK_H"]))
    _autotuned_mla_rope_reconstruct_output_kernel[grid](
        x,
        y,
        cos,
        sin,
        nope_dim,
        emb_dim,
        x.shape[1],
        batch_size,
        seq_num,
        cu_seqlens_q,
        position_ids,
        x.stride(0),
        x.stride(1),
        cos.stride(0),
        sin.stride(0),
        cp_rank,
        cp_size,
        INVERSE=inverse,
        REMOVE_INTERLEAVING=remove_interleaving,
        BLOCK_NOPE=triton.next_power_of_2(nope_dim),
    )
    return out


@dataclass
class OutputRopeParams:
    """Inverse-RoPE applied to the attention output inside the fused Functions.

    DSv4 undoes the query-side RoPE on the attention output.  Folding that into
    the sparse-attention ``Function`` lets the rotation happen in place on the
    tensor the Function already saves for backward, instead of on a private
    clone, which removes one full ``(rows, heads, v_head_dim)`` buffer per layer
    from the forward-to-backward live set.

    The rotation is orthogonal only when the cos/sin were built with
    ``mscale=1.0`` (DSv4's "pure rotation" contract).  That is what makes
    :meth:`undo` an inverse up to floating-point rounding, so backward can
    recover the pre-RoPE output and gradient.

    ``sbhd_batch_size`` is required for the sbhd layout, where the flat
    ``(sq * b, ...)`` rows are sq-major and the kernel derives the token index
    as ``row // batch_size``.  For thd it must be None and ``cu_seqlens_q``
    supplies the segment boundaries instead.
    """

    cos: Tensor
    sin: Tensor
    nope_dim: int
    pos_dim: int
    cu_seqlens_q: Optional[Tensor] = None
    cp_rank: int = 0
    cp_size: int = 1
    position_ids: Optional[Tensor] = None
    sbhd_batch_size: Optional[int] = None

    def _kernel_view(self, out_flat: Tensor) -> Tensor:
        """View a flat ``(rows, heads, head_dim)`` output the way the kernel wants it."""
        if out_flat.shape[-1] != self.nope_dim + self.pos_dim:
            raise ValueError(
                "Output inverse-RoPE expects head_dim == nope_dim + pos_dim, got "
                f"{out_flat.shape[-1]} != {self.nope_dim} + {self.pos_dim}."
            )
        if self.cu_seqlens_q is not None:
            return out_flat
        if self.sbhd_batch_size is None:
            raise ValueError("sbhd_batch_size is required when cu_seqlens_q is None.")
        rows, nheads, head_dim = out_flat.shape
        return out_flat.view(rows // self.sbhd_batch_size, self.sbhd_batch_size, nheads, head_dim)

    def _kernel_kwargs(self) -> dict:
        return dict(
            cu_seqlens_q=self.cu_seqlens_q,
            cp_rank=self.cp_rank,
            cp_size=self.cp_size,
            inverse=True,
            remove_interleaving=True,
            position_ids=self.position_ids,
        )

    def apply_(self, out_flat: Tensor) -> Tensor:
        """Inverse-RoPE a ``(rows, heads, head_dim)`` attention output in place."""
        mla_rope_apply_raw_(
            self._kernel_view(out_flat),
            self.cos,
            self.sin,
            self.nope_dim,
            self.pos_dim,
            **self._kernel_kwargs(),
        )
        return out_flat

    def undo(self, t: Tensor, out: Optional[Tensor] = None) -> Tensor:
        """Recover the pre-RoPE values of ``t``, in place or into ``out``.

        Pass ``out`` whenever ``t`` is aliased by a tensor that another autograd
        node still expects to hold the rotated values, which is the case for the
        saved attention output.
        """
        mla_rope_unapply_raw(
            self._kernel_view(t),
            self.cos,
            self.sin,
            self.nope_dim,
            self.pos_dim,
            out=None if out is None else self._kernel_view(out),
            **self._kernel_kwargs(),
        )
        return t if out is None else out


def undo_output_rope_for_backward(
    out_rope: Optional[OutputRopeParams], out_flat: Tensor, dO_flat: Tensor
) -> Tuple[Tensor, Tensor]:
    """Reconstruct cuDNN inputs without mutating saved output or upstream gradient.

    Either input may be shared with another autograd consumer. ``contiguous()``
    alone does not establish exclusive ownership, so use separate destinations.
    """
    if out_rope is None:
        return out_flat, dO_flat
    o_unrot = torch.empty_like(out_flat)
    out_rope.undo(out_flat, out=o_unrot)
    dO_flat = dO_flat.contiguous()
    do_unrot = torch.empty_like(dO_flat)
    out_rope.undo(dO_flat, out=do_unrot)
    return o_unrot, do_unrot
