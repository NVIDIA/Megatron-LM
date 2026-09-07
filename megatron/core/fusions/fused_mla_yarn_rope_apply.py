# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


@triton.jit
def _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
    # Cast ``pid_m`` and ``cu_seqlens`` loads to a single shared dtype so
    # the loop-body reassignments don't surface as
    # "initial value is int32 but redefined as int64" in newer Triton
    # versions (which promote ``// Python_int`` to int64).
    pid_m = pid_m.to(tl.int64)
    token_idx = tl.full((), -1, dtype=tl.int64)
    this_seq_len = tl.full((), 0, dtype=tl.int64)
    seq_idx = 0
    last_cum_seqlen = tl.load(cu_seqlens).to(tl.int64) // cp_size
    while seq_idx < seq_num:
        cur_cum_seqlen = tl.load(cu_seqlens + seq_idx + 1).to(tl.int64) // cp_size
        if token_idx == -1 and cur_cum_seqlen > pid_m:
            token_idx = pid_m - last_cum_seqlen
            this_seq_len = cur_cum_seqlen - last_cum_seqlen
        last_cum_seqlen = cur_cum_seqlen
        seq_idx += 1
    # Padding tokens beyond cu_seqlens[-1] (from THD CUDA-graph padding)
    # never match any sequence, leaving token_idx == -1.  Clamp to 0 so
    # the cos/sin table loads stay in-bounds; the wrong RoPE result is
    # harmless because padding positions are excluded by loss_mask.
    if token_idx == -1:
        token_idx = tl.full((), 0, dtype=tl.int64)
    if cp_size > 1:
        first_cp_seg = (this_seq_len + 1) // 2
        second_cp_seg = this_seq_len // 2
        if token_idx < first_cp_seg:
            token_idx = token_idx + cp_rank * first_cp_seg
        else:
            token_idx = (
                token_idx
                - first_cp_seg
                + cp_size * first_cp_seg
                + (cp_size - cp_rank - 1) * second_cp_seg
            )
    return token_idx


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["Q"],
)
@triton.jit
def _mla_rope_fwd_inplace_kernel(
    Q,
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
    ROPE_FIRST: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Forward pass: apply RoPE inplace to the leading emb_dim elements when ROPE_FIRST is true,
    otherwise to the trailing emb_dim elements.
    Reads from interleaved layout, writes back to interleaved layout.

    Input:
        Q: [seq_len, batch_size, head_num, nope_dim + emb_dim]
            or [total_seq_len, head_num, nope_dim + emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size: batch size for sbhd format, not used for thd format
        seq_num: number of sequences for thd format, not used for sbhd format
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
    """
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

    Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    rope_offset = 0 if ROPE_FIRST else nope_dim
    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + rope_offset
    mask = x_off < head_num * stride_x_nheads
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_2_off = x_1_off + 1
    x_1 = tl.load(Q + x_1_off, mask=mask)
    x_2 = tl.load(Q + x_2_off, mask=mask)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right

    if REMOVE_INTERLEAVING:
        tl.store(Q + x_1_off, x_left, mask=mask)
        tl.store(Q + x_2_off, x_right, mask=mask)
    else:
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        tl.store(Q + x_left_off, x_left, mask=mask)
        tl.store(Q + x_right_off, x_right, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["DO_OUT"],
)
@triton.jit
def _mla_rope_bwd_kernel(
    DO_IN,
    DO_OUT,
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
    ROPE_FIRST: tl.constexpr,
    COPY_NOPE: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Backward pass: inverse RoPE on the leading emb_dim elements when ROPE_FIRST is true,
    otherwise on the trailing emb_dim elements.
    Reads from interleaved layout, writes to interleaved layout.

    ``DO_IN`` and ``DO_OUT`` may alias, which gives the in-place form. When they
    are distinct buffers, ``COPY_NOPE`` must be set so the leading ``nope_dim``
    elements, which the rotation never touches, are copied across and ``DO_OUT``
    is left fully populated. ``BLOCK_NOPE`` is then the next power of two at or
    above ``nope_dim``.

    Input:
        DO_IN: [seq_len, batch_size, head_num, nope_dim + emb_dim]
            or [total_seq_len, head_num, nope_dim + emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size, seq_num, and cu_seqlens_q are the same as in the forward pass

    Output:
        DO_OUT: same shape and strides as DO_IN
    """
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
    DO_IN = DO_IN + row_off
    DO_OUT = DO_OUT + row_off

    head_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads
    rope_offset = 0 if ROPE_FIRST else nope_dim
    x_off = head_off + rope_offset
    mask = x_off < head_num * stride_x_nheads
    if REMOVE_INTERLEAVING:
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        x_left = tl.load(DO_IN + x_1_off, mask=mask)
        x_right = tl.load(DO_IN + x_2_off, mask=mask)
    else:
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(DO_IN + x_left_off, mask=mask)
        x_right = tl.load(DO_IN + x_right_off, mask=mask)
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1

    x_1 = x_left * cos_left + x_right * sin_right
    x_2 = -x_left * sin_left + x_right * cos_right

    tl.store(DO_OUT + x_1_off, x_1, mask=mask)
    tl.store(DO_OUT + x_2_off, x_2, mask=mask)

    if COPY_NOPE:
        nope_idx = tl.arange(0, BLOCK_NOPE)[None, :]
        nope_off = head_off + (emb_dim if ROPE_FIRST else 0) + nope_idx
        nope_mask = mask & (nope_idx < nope_dim)
        tl.store(
            DO_OUT + nope_off,
            tl.load(DO_IN + nope_off, mask=nope_mask),
            mask=nope_mask,
        )


def _flatten_rope_input(x, cu_seqlens_q, position_ids):
    """Normalize an sbhd or thd RoPE input to a ``(rows, head_num, head_dim)`` view.

    Returns ``(x_flat, batch_size, seq_num)``.  ``batch_size`` is only meaningful
    for sbhd and ``seq_num`` only for thd; the unused one is None, matching what
    the kernels expect.
    """
    if cu_seqlens_q is None:
        # sbhd
        assert position_ids is None
        _max_seqlen, batch_size, nheads, headdim = x.shape
        return x.view(-1, nheads, headdim), batch_size, None
    # thd
    total_seqlen = x.shape[0]
    if position_ids is not None:
        assert position_ids.shape == (total_seqlen,)
    return x, None, len(cu_seqlens_q) - 1


def _check_rope_layout(x, cos, sin, nope_dim, emb_dim):
    """Validate the contiguity and head-dim invariants both RoPE kernels rely on."""
    assert x.stride(-1) == 1
    assert cos.stride(-1) == 1
    assert sin.stride(-1) == 1
    assert x.shape[-1] == nope_dim + emb_dim
    assert emb_dim % 4 == 0


def mla_rope_apply_raw_(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    emb_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    inverse: bool = False,
    remove_interleaving: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    rope_first: bool = False,
) -> torch.Tensor:
    """Apply the MLA RoPE rotation to ``t`` in place, bypassing autograd.

    Same kernel and semantics as :func:`fused_mla_rope_inplace`, but without the
    autograd node.  Intended for callers that already run inside a custom
    ``torch.autograd.Function`` and take care of the backward themselves.

    Args:
        t: [seq_len, batch_size, head_num, nope_dim + emb_dim]
            or [total_seq_len, head_num, nope_dim + emb_dim]
        cos/sin: [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
        inverse: if True, apply the inverse rotation
        remove_interleaving: if True, output RoPE dims in non-interleaved layout
        position_ids: optional thd row positions overriding the CP row mapping
        rope_first: if True, rotate the leading emb_dim elements instead of the trailing ones

    Returns:
        t: the same tensor, modified in place
    """
    x, batch_size, seq_num = _flatten_rope_input(t, cu_seqlens_q, position_ids)
    _check_rope_layout(x, cos, sin, nope_dim, emb_dim)
    total_seqlen, nheads, _ = x.shape

    grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
    _mla_rope_fwd_inplace_kernel[grid](
        x,
        cos,
        sin,
        nope_dim,
        emb_dim,
        nheads,
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
        ROPE_FIRST=rope_first,
    )
    return t


def mla_rope_unapply_raw(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    emb_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    inverse: bool = False,
    remove_interleaving: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    rope_first: bool = False,
) -> torch.Tensor:
    """Undo :func:`mla_rope_apply_raw_`, bypassing autograd.

    This is the exact transpose of the forward rotation, so for a unit-magnitude
    rotation (``mscale == 1.0``) it is also its exact inverse.  Pass the same
    ``inverse`` and ``remove_interleaving`` flags that were used to apply it.

    With ``out=None`` the un-rotation happens in place.  Pass ``out`` when ``t``
    is aliased by a tensor another autograd node still expects to hold the
    rotated values; the leading ``nope_dim`` elements are copied across so
    ``out`` is fully populated in a single pass.

    Returns:
        ``t`` when ``out`` is None, otherwise ``out``.
    """
    x, batch_size, seq_num = _flatten_rope_input(t, cu_seqlens_q, position_ids)
    _check_rope_layout(x, cos, sin, nope_dim, emb_dim)
    total_seqlen, nheads, _ = x.shape

    if out is None:
        y = x
    else:
        assert out.shape == t.shape
        assert out.stride() == t.stride()
        assert out.dtype == t.dtype
        y, _, _ = _flatten_rope_input(out, cu_seqlens_q, position_ids)

    grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
    _mla_rope_bwd_kernel[grid](
        x,
        y,
        cos,
        sin,
        nope_dim,
        emb_dim,
        nheads,
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
        ROPE_FIRST=rope_first,
        COPY_NOPE=out is not None,
        BLOCK_NOPE=triton.next_power_of_2(nope_dim) if out is not None else 1,
    )
    return t if out is None else out


class _FusedMLARoPEInplace(torch.autograd.Function):
    """
    Autograd function for applying RoPE inplace to either end of a multi-head tensor.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        rotary_interleaved=False,
        inverse=False,
        remove_interleaving=False,
        position_ids=None,
        rope_first=False,
    ):
        """
        Forward function for _FusedMLARoPEInplace.

        Args:
            q: [seq_len, batch_size, head_num, nope_dim + emb_dim]
                or [total_seq_len, head_num, nope_dim + emb_dim]
            cos/sin: [max_seq_len, 1, 1, emb_dim]
            cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
            rotary_interleaved: whether to apply RoPE interleaved, only supports False for now
            inverse: if True, negate sin inside the kernel to apply the inverse rotation
            rope_first: if True, rotate the leading emb_dim elements instead of the trailing ones
        """
        assert not rotary_interleaved
        mla_rope_apply_raw_(
            q,
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
            rope_first=rope_first,
        )
        ctx.save_for_backward(cos, sin, *(() if position_ids is None else (position_ids,)))
        ctx.has_position_ids = position_ids is not None
        ctx.nope_dim = nope_dim
        ctx.emb_dim = emb_dim
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.rotary_interleaved = rotary_interleaved
        ctx.inverse = inverse
        ctx.remove_interleaving = remove_interleaving
        ctx.rope_first = rope_first
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        return q

    @staticmethod
    def backward(ctx, grad):
        """
        Backward function for _FusedMLARoPEInplace.

        Args:
            grad: [seq_len, batch_size, head_num, nope_dim + emb_dim]
                or [total_seq_len, head_num, nope_dim + emb_dim]
        """
        if ctx.has_position_ids:
            cos, sin, position_ids = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
            position_ids = None
        if ctx.cu_seqlens_q is None or ctx.has_position_ids:
            grad = grad.contiguous()

        mla_rope_unapply_raw(
            grad,
            cos,
            sin,
            ctx.nope_dim,
            ctx.emb_dim,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cp_rank=ctx.cp_rank,
            cp_size=ctx.cp_size,
            inverse=ctx.inverse,
            remove_interleaving=ctx.remove_interleaving,
            position_ids=position_ids,
            rope_first=ctx.rope_first,
        )
        return grad, None, None, None, None, None, None, None, None, None, None, None, None


def fused_mla_rope_inplace(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    emb_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
    inverse: bool = False,
    remove_interleaving: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    rope_first: bool = False,
):
    """
    Fused RoPE applied inplace to emb_dim elements at either end of a tensor,
    leaving the nope_dim elements unchanged.
    It supports both sbhd and thd input formats.

    When ``inverse=True`` the rotation is reversed, which is useful for
    undoing RoPE on the attention output.

    For the notations below, seq_len is the length of the sequence per batch for sbhd format,
    total_seq_len is the total length of the sequences for thd format.
    max_seq_len is the maximum length of the sequences in the input tensor.

    Args:
        t: [seq_len, batch_size, head_num, nope_dim + emb_dim]
            or [total_seq_len, head_num, nope_dim + emb_dim]
        cos/sin: [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
        rotary_interleaved: whether to apply RoPE interleaved, only supports False for now
        inverse: if True, apply the inverse rotation
        remove_interleaving: if True, output RoPE dims in non-interleaved layout
        position_ids: optional THD row positions. When supplied, these positions
            replace the built-in CP row-to-position mapping.
        rope_first: if True, rotate the leading emb_dim elements instead of the trailing ones.

    Returns:
        t: inplace modified input tensor
    """
    return _FusedMLARoPEInplace.apply(
        t,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        rotary_interleaved,
        inverse,
        remove_interleaving,
        position_ids,
        rope_first,
    )


def fused_mla_rope_out_of_place(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    emb_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
    inverse: bool = False,
    remove_interleaving: bool = False,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the fused RoPE kernel without modifying the input tensor.

    Use this wrapper when an upstream autograd function may have retained its
    output for backward. The underlying kernel remains in-place, so a private
    copy is required to keep the retained tensor unchanged.
    """
    return fused_mla_rope_inplace(
        t.clone(),
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q=cu_seqlens_q,
        cp_rank=cp_rank,
        cp_size=cp_size,
        rotary_interleaved=rotary_interleaved,
        inverse=inverse,
        remove_interleaving=remove_interleaving,
        position_ids=position_ids,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["nope_dim", "emb_dim", "head_num"],
)
@triton.jit
def _mla_rope_concat_fwd_kernel(
    NOPE,
    ROPE,
    OUTPUT,
    COS,
    SIN,
    nope_dim: tl.constexpr,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens,
    stride_nope_seq,
    stride_nope_batch,
    stride_nope_head,
    stride_nope_dim,
    stride_rope_seq,
    stride_rope_batch,
    stride_rope_head,
    stride_rope_dim,
    stride_output_seq,
    stride_output_head,
    stride_cos_seq,
    stride_sin_seq,
    cp_rank,
    cp_size,
    IS_THD: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROT_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Concatenate the non-RoPE part with a rotated positional part."""
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if IS_THD:
        token_idx = _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size)
        nope_row = NOPE + pid_m * stride_nope_seq
        rope_row = ROPE + pid_m * stride_rope_seq
    else:
        seq_idx = pid_m // batch_size
        batch_idx = pid_m % batch_size
        token_idx = seq_idx
        nope_row = NOPE + seq_idx * stride_nope_seq + batch_idx * stride_nope_batch
        rope_row = ROPE + seq_idx * stride_rope_seq + batch_idx * stride_rope_batch

    head_idx = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_idx < head_num

    nope_idx = tl.arange(0, NOPE_BLOCK)
    nope_mask = head_mask[:, None] & (nope_idx[None, :] < nope_dim)
    nope_offsets = head_idx[:, None] * stride_nope_head + nope_idx[None, :] * stride_nope_dim
    nope = tl.load(nope_row + nope_offsets, mask=nope_mask)

    rope_idx = tl.arange(0, ROT_BLOCK)
    rope_mask = head_mask[:, None] & (rope_idx[None, :] < emb_dim // 2)
    rope_base = head_idx[:, None] * stride_rope_head
    rope_pair = rope_idx[None, :] * 2 * stride_rope_dim
    x_1 = tl.load(rope_row + rope_base + rope_pair, mask=rope_mask)
    x_2 = tl.load(rope_row + rope_base + rope_pair + stride_rope_dim, mask=rope_mask)

    cos_left = tl.load(COS + token_idx * stride_cos_seq + rope_idx, mask=rope_idx < emb_dim // 2)
    sin_left = tl.load(SIN + token_idx * stride_sin_seq + rope_idx, mask=rope_idx < emb_dim // 2)
    cos_right = tl.load(
        COS + token_idx * stride_cos_seq + emb_dim // 2 + rope_idx, mask=rope_idx < emb_dim // 2
    )
    sin_right = tl.load(
        SIN + token_idx * stride_sin_seq + emb_dim // 2 + rope_idx, mask=rope_idx < emb_dim // 2
    )
    x_left = x_1 * cos_left[None, :] - x_2 * sin_left[None, :]
    x_right = x_2 * cos_right[None, :] + x_1 * sin_right[None, :]

    output_row = OUTPUT + pid_m * stride_output_seq
    output_base = head_idx[:, None] * stride_output_head
    output_nope_offsets = output_base + nope_idx[None, :]
    tl.store(output_row + output_nope_offsets, nope, mask=nope_mask)
    output_rope_offsets = output_base + nope_dim + rope_idx[None, :]
    tl.store(output_row + output_rope_offsets, x_left, mask=rope_mask)
    tl.store(output_row + output_rope_offsets + emb_dim // 2, x_right, mask=rope_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["nope_dim", "emb_dim", "head_num"],
)
@triton.jit
def _mla_rope_concat_bwd_kernel(
    DOUTPUT,
    DNOPE,
    DROPE,
    COS,
    SIN,
    nope_dim: tl.constexpr,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens,
    stride_doutput_seq,
    stride_doutput_head,
    stride_dnope_seq,
    stride_dnope_head,
    stride_drope_seq,
    stride_drope_head,
    stride_cos_seq,
    stride_sin_seq,
    cp_rank,
    cp_size,
    IS_THD: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROT_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Split the output gradient and apply the inverse rotation."""
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if IS_THD:
        token_idx = _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size)
    else:
        token_idx = pid_m // batch_size

    head_idx = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_idx < head_num
    doutput_row = DOUTPUT + pid_m * stride_doutput_seq
    doutput_base = head_idx[:, None] * stride_doutput_head

    nope_idx = tl.arange(0, NOPE_BLOCK)
    nope_mask = head_mask[:, None] & (nope_idx[None, :] < nope_dim)
    nope_offsets = doutput_base + nope_idx[None, :]
    dnope = tl.load(doutput_row + nope_offsets, mask=nope_mask)
    dnope_row = DNOPE + pid_m * stride_dnope_seq
    dnope_offsets = head_idx[:, None] * stride_dnope_head + nope_idx[None, :]
    tl.store(dnope_row + dnope_offsets, dnope, mask=nope_mask)

    rope_idx = tl.arange(0, ROT_BLOCK)
    rope_mask = head_mask[:, None] & (rope_idx[None, :] < emb_dim // 2)
    rope_offsets = doutput_base + nope_dim + rope_idx[None, :]
    dx_left = tl.load(doutput_row + rope_offsets, mask=rope_mask)
    dx_right = tl.load(doutput_row + rope_offsets + emb_dim // 2, mask=rope_mask)

    cos_left = tl.load(COS + token_idx * stride_cos_seq + rope_idx, mask=rope_idx < emb_dim // 2)
    sin_left = tl.load(SIN + token_idx * stride_sin_seq + rope_idx, mask=rope_idx < emb_dim // 2)
    cos_right = tl.load(
        COS + token_idx * stride_cos_seq + emb_dim // 2 + rope_idx, mask=rope_idx < emb_dim // 2
    )
    sin_right = tl.load(
        SIN + token_idx * stride_sin_seq + emb_dim // 2 + rope_idx, mask=rope_idx < emb_dim // 2
    )
    dx_1 = dx_left * cos_left[None, :] + dx_right * sin_right[None, :]
    dx_2 = -dx_left * sin_left[None, :] + dx_right * cos_right[None, :]

    drope_row = DROPE + pid_m * stride_drope_seq
    drope_base = head_idx[:, None] * stride_drope_head
    drope_pair = rope_idx[None, :] * 2
    tl.store(drope_row + drope_base + drope_pair, dx_1, mask=rope_mask)
    tl.store(drope_row + drope_base + drope_pair + 1, dx_2, mask=rope_mask)


class _FusedMLARoPEConcat(torch.autograd.Function):
    """Autograd wrapper for fused MLA packing and RoPE."""

    @staticmethod
    def forward(ctx, nope, rope, cos, sin, cu_seqlens, cp_rank, cp_size):
        """Pack non-positional and RoPE channels while applying rotary embeddings."""
        assert nope.ndim in (3, 4)
        assert rope.ndim == nope.ndim
        assert nope.shape[:-1] == rope.shape[:-1]
        assert nope.dtype == rope.dtype
        assert nope.device == rope.device == cos.device == sin.device
        assert cos.stride(-1) == 1 and sin.stride(-1) == 1

        is_thd = nope.ndim == 3
        if is_thd:
            total_seqlen, nheads, nope_dim = nope.shape
            seq_num = len(cu_seqlens) - 1
            batch_size = 1
            nope_strides = (nope.stride(0), 0, nope.stride(1), nope.stride(2))
            rope_strides = (rope.stride(0), 0, rope.stride(1), rope.stride(2))
        else:
            max_seqlen, batch_size, nheads, nope_dim = nope.shape
            total_seqlen = max_seqlen * batch_size
            seq_num = 0
            assert cu_seqlens is None
            nope_strides = nope.stride()
            rope_strides = rope.stride()

        emb_dim = rope.size(-1)
        assert emb_dim % 4 == 0
        nope_block = triton.next_power_of_2(nope_dim)
        rot_block = triton.next_power_of_2(emb_dim // 2)

        output = nope.new_empty(total_seqlen, nheads, nope_dim + emb_dim)
        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        _mla_rope_concat_fwd_kernel[grid](
            nope,
            rope,
            output,
            cos,
            sin,
            nope_dim,
            emb_dim,
            nheads,
            batch_size,
            seq_num,
            cu_seqlens,
            *nope_strides,
            *rope_strides,
            output.stride(0),
            output.stride(1),
            cos.stride(0),
            sin.stride(0),
            cp_rank,
            cp_size,
            IS_THD=is_thd,
            NOPE_BLOCK=nope_block,
            ROT_BLOCK=rot_block,
        )

        ctx.save_for_backward(cos, sin)
        ctx.input_shape = nope.shape
        ctx.cu_seqlens = cu_seqlens
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.nope_dim = nope_dim
        ctx.emb_dim = emb_dim
        ctx.is_thd = is_thd
        if not is_thd:
            output = output.view(max_seqlen, batch_size, nheads, nope_dim + emb_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients for the packed non-positional and RoPE inputs."""
        cos, sin = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if ctx.is_thd:
            total_seqlen, nheads, _ = grad_output.shape
            batch_size = 1
            seq_num = len(ctx.cu_seqlens) - 1
        else:
            max_seqlen, batch_size, nheads, _ = grad_output.shape
            total_seqlen = max_seqlen * batch_size
            grad_output = grad_output.view(total_seqlen, nheads, -1)
            seq_num = 0

        dnope = grad_output.new_empty(total_seqlen, nheads, ctx.nope_dim)
        drope = grad_output.new_empty(total_seqlen, nheads, ctx.emb_dim)
        nope_block = triton.next_power_of_2(ctx.nope_dim)
        rot_block = triton.next_power_of_2(ctx.emb_dim // 2)
        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        _mla_rope_concat_bwd_kernel[grid](
            grad_output,
            dnope,
            drope,
            cos,
            sin,
            ctx.nope_dim,
            ctx.emb_dim,
            nheads,
            batch_size,
            seq_num,
            ctx.cu_seqlens,
            grad_output.stride(0),
            grad_output.stride(1),
            dnope.stride(0),
            dnope.stride(1),
            drope.stride(0),
            drope.stride(1),
            cos.stride(0),
            sin.stride(0),
            ctx.cp_rank,
            ctx.cp_size,
            IS_THD=ctx.is_thd,
            NOPE_BLOCK=nope_block,
            ROT_BLOCK=rot_block,
        )
        dnope = dnope.view(ctx.input_shape)
        drope = drope.view(*ctx.input_shape[:-1], ctx.emb_dim)
        return dnope, drope, None, None, None, None, None


def fused_mla_rope_concat(
    nope: torch.Tensor,
    rope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> torch.Tensor:
    """Pack MLA's non-positional and positional parts while applying RoPE.

    ``nope`` and ``rope`` may use arbitrary input strides. The returned SBHD
    or THD tensor is contiguous and stores the rotated positional channels
    after the non-positional channels.
    """
    return _FusedMLARoPEConcat.apply(nope, rope, cos, sin, cu_seqlens, cp_rank, cp_size)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "k_dim", "v_dim", "head_num"],
)
@triton.jit
def _mla_rope_fwd_kv_split_kernel(
    KV,
    K_POS_EMB,
    O_KEY,
    O_VALUE,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_kv_seq,
    stride_kv_nheads,
    stride_emb_seq,
    stride_k_seq,
    stride_k_nheads,
    stride_v_seq,
    stride_v_nheads,
    cp_rank,
    cp_size,
    REMOVE_INTERLEAVING: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Forward pass: split KV into key and value, apply RoPE to k_pos_emb,
    and concatenate the result onto key.

    Input:
        KV: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        K_POS_EMB: [seq_len, batch_size, emb_dim] or [total_seq_len, emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size: batch size for sbhd format, not used for thd format
        seq_num: number of sequences for thd format, not used for sbhd format
        cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format

    Output:
        O_KEY: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        O_VALUE: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    KV_ptr = KV + pid_m * stride_kv_seq + pid_head * BLOCK_H * stride_kv_nheads
    kv_off = tl.arange(0, BLOCK_H)[:, None] * stride_kv_nheads
    mask = kv_off < head_num * stride_kv_nheads
    k_in_off = kv_off + tl.arange(0, k_dim)[None, :]
    v_in_off = kv_off + k_dim + tl.arange(0, v_dim)[None, :]
    k = tl.load(KV_ptr + k_in_off, mask=mask)
    v = tl.load(KV_ptr + v_in_off, mask=mask)

    K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads
    V_ptr = O_VALUE + pid_m * stride_v_seq + pid_head * BLOCK_H * stride_v_nheads

    k_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads + tl.arange(0, k_dim)[None, :]
    v_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_v_nheads + tl.arange(0, v_dim)[None, :]
    tl.store(K_ptr + k_out_off, k, mask=mask)
    tl.store(V_ptr + v_out_off, v, mask=mask)

    EMB = K_POS_EMB + pid_m * stride_emb_seq
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
    x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right
    x_left = x_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    x_right = x_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    if REMOVE_INTERLEAVING:
        x_1_off = (
            tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
            + k_dim
            + tl.arange(0, emb_dim // 2)[None, :] * 2
        )
        x_2_off = x_1_off + 1
        tl.store(K_ptr + x_1_off, x_left, mask=mask)
        tl.store(K_ptr + x_2_off, x_right, mask=mask)
    else:
        x_left_off = (
            tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
            + k_dim
            + tl.arange(0, emb_dim // 2)[None, :]
        )
        x_right_off = x_left_off + emb_dim // 2
        tl.store(K_ptr + x_left_off, x_left, mask=mask)
        tl.store(K_ptr + x_right_off, x_right, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "k_dim", "v_dim", "head_num"],
)
@triton.jit
def _mla_rope_bwd_kv_split_kernel(
    dK,
    dV,
    dKV,
    dEMB,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_dk_seq,
    stride_dk_nheads,
    stride_dv_seq,
    stride_dv_nheads,
    stride_dkv_seq,
    stride_dkv_nheads,
    stride_demb_seq,
    cp_rank,
    cp_size,
    REMOVE_INTERLEAVING: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Backward pass for the KV-split RoPE.

    Input:
        dK: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        dV: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size, seq_num, and cu_seqlens_kv are the same as in the forward pass

    Output:
        dKV: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        dEMB: [seq_len, batch_size, emb_dim] or [total_seq_len, emb_dim]
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    dKV_ptr = dKV + pid_m * stride_dkv_seq + pid_head * BLOCK_H * stride_dkv_nheads
    dkv_off = tl.arange(0, BLOCK_H)[:, None] * stride_dkv_nheads
    mask = dkv_off < head_num * stride_dkv_nheads
    dk_out_off = dkv_off + tl.arange(0, k_dim)[None, :]
    dv_out_off = dkv_off + k_dim + tl.arange(0, v_dim)[None, :]

    dK_ptr = dK + pid_m * stride_dk_seq + pid_head * BLOCK_H * stride_dk_nheads
    dV_ptr = dV + pid_m * stride_dv_seq + pid_head * BLOCK_H * stride_dv_nheads
    dk_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + tl.arange(0, k_dim)[None, :]
    dv_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dv_nheads + tl.arange(0, v_dim)[None, :]
    dk = tl.load(dK_ptr + dk_in_off, mask=mask)
    dv = tl.load(dV_ptr + dv_in_off, mask=mask)
    tl.store(dKV_ptr + dk_out_off, dk, mask=mask)
    tl.store(dKV_ptr + dv_out_off, dv, mask=mask)

    if pid_head == 0:
        x_left_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        x_right_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
            dK_ptr = dK + pid_m * stride_dk_seq + i * BLOCK_H * stride_dk_nheads
            x_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + k_dim
            mask = x_off < head_num * stride_dk_nheads
            if REMOVE_INTERLEAVING:
                x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
                x_2_off = x_1_off + 1
                x_left = tl.load(dK_ptr + x_1_off, mask=mask)
                x_right = tl.load(dK_ptr + x_2_off, mask=mask)
            else:
                x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
                x_right_off = x_left_off + emb_dim // 2
                x_left = tl.load(dK_ptr + x_left_off, mask=mask)
                x_right = tl.load(dK_ptr + x_right_off, mask=mask)
            x_left_accum += x_left
            x_right_accum += x_right
        x_left_accum = tl.sum(x_left_accum, axis=0)
        x_right_accum = tl.sum(x_right_accum, axis=0)
        x_left_accum = x_left_accum.to(dEMB.dtype.element_ty)
        x_right_accum = x_right_accum.to(dEMB.dtype.element_ty)

        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

        x_1 = x_left_accum * cos_left + x_right_accum * sin_right
        x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
        dEMB_ptr = dEMB + pid_m * stride_demb_seq
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2, x_1)
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2 + 1, x_2)


class _FusedMLARoPEKVSplit(torch.autograd.Function):
    """
    Autograd function for applying RoPE to MLA's key and value.
    Splits KV, applies RoPE to k_pos_emb, concatenates onto key.
    """

    @staticmethod
    def forward(
        ctx,
        kv,
        k_pos_emb,
        cos,
        sin,
        emb_dim,
        k_dim,
        v_dim,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
        rotary_interleaved=False,
        remove_interleaving=False,
    ):
        """
        Forward function for _FusedMLARoPEKVSplit.

        Args:
            kv: [seq_len, batch_size, head_num, k_dim + v_dim]
                or [total_seq_len, head_num, k_dim + v_dim]
            k_pos_emb: [seq_len, batch_size, 1, emb_dim] or [total_seq_len, 1, emb_dim]
            cos/sin: [max_seq_len, 1, 1, emb_dim]
            cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format
            rotary_interleaved: whether to apply RoPE interleaved, only supports False for now
        """
        assert not rotary_interleaved
        max_seqlen = None
        batch_size = None
        seq_num = None
        if cu_seqlens_kv is None:
            # sbhd
            max_seqlen, batch_size, nheads, headdim = kv.shape
            kv = kv.view(-1, nheads, headdim)
            k_pos_emb = k_pos_emb.view(-1, emb_dim)
            total_seqlen = kv.shape[0]
        else:
            # thd
            seq_num = len(cu_seqlens_kv) - 1
            total_seqlen, nheads, headdim = kv.shape
        assert headdim == k_dim + v_dim
        assert kv.stride(-1) == 1
        assert k_pos_emb.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert emb_dim % 4 == 0

        o_key = kv.new_empty(total_seqlen, nheads, emb_dim + k_dim)
        o_value = kv.new_empty(total_seqlen, nheads, v_dim)

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        _mla_rope_fwd_kv_split_kernel[grid](
            kv,
            k_pos_emb,
            o_key,
            o_value,
            cos,
            sin,
            emb_dim,
            k_dim,
            v_dim,
            nheads,
            batch_size,
            seq_num,
            cu_seqlens_kv,
            kv.stride(0),
            kv.stride(1),
            k_pos_emb.stride(0),
            o_key.stride(0),
            o_key.stride(1),
            o_value.stride(0),
            o_value.stride(1),
            cp_rank,
            cp_size,
            REMOVE_INTERLEAVING=remove_interleaving,
        )
        ctx.save_for_backward(cos, sin)
        ctx.remove_interleaving = remove_interleaving
        ctx.rotary_interleaved = rotary_interleaved
        ctx.emb_dim = emb_dim
        ctx.k_dim = k_dim
        ctx.v_dim = v_dim
        ctx.cu_seqlens_kv = cu_seqlens_kv
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        if cu_seqlens_kv is None:
            o_key = o_key.view(max_seqlen, -1, nheads, emb_dim + k_dim)
            o_value = o_value.view(max_seqlen, -1, nheads, v_dim)
        return o_key, o_value

    @staticmethod
    def backward(ctx, dk, dv):
        """
        Backward function for _FusedMLARoPEKVSplit.

        Args:
            dk: [seq_len, batch_size, head_num, emb_dim + k_dim]
                or [total_seq_len, head_num, emb_dim + k_dim]
            dv: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
        """
        cos, sin = ctx.saved_tensors
        max_seqlen = None
        batch_size = None
        seq_num = None
        if ctx.cu_seqlens_kv is None:
            # sbhd
            max_seqlen, batch_size, nheads, _ = dk.shape
            dk = dk.contiguous().view(-1, nheads, ctx.emb_dim + ctx.k_dim)
            dv = dv.contiguous().view(-1, nheads, ctx.v_dim)
            total_seqlen = dk.shape[0]
        else:
            # thd
            seq_num = len(ctx.cu_seqlens_kv) - 1
            total_seqlen, nheads, _ = dk.shape
        assert dk.stride(-1) == 1
        assert dv.stride(-1) == 1

        d_kv = dk.new_empty(total_seqlen, nheads, ctx.k_dim + ctx.v_dim)
        d_emb = dk.new_empty(total_seqlen, 1, ctx.emb_dim)

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        _mla_rope_bwd_kv_split_kernel[grid](
            dk,
            dv,
            d_kv,
            d_emb,
            cos,
            sin,
            ctx.emb_dim,
            ctx.k_dim,
            ctx.v_dim,
            nheads,
            batch_size,
            seq_num,
            ctx.cu_seqlens_kv,
            dk.stride(0),
            dk.stride(1),
            dv.stride(0),
            dv.stride(1),
            d_kv.stride(0),
            d_kv.stride(1),
            d_emb.stride(0),
            ctx.cp_rank,
            ctx.cp_size,
            REMOVE_INTERLEAVING=ctx.remove_interleaving,
        )
        if ctx.cu_seqlens_kv is None:
            d_kv = d_kv.view(max_seqlen, batch_size, nheads, ctx.k_dim + ctx.v_dim)
            d_emb = d_emb.view(max_seqlen, batch_size, 1, ctx.emb_dim)
        return d_kv, d_emb, None, None, None, None, None, None, None, None, None, None


def fused_mla_rope_kv_split(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    emb_dim: int,
    k_dim: int,
    v_dim: int,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
    remove_interleaving: bool = False,
):
    """
    Fused function for applying RoPE to MLA's key and value.
    It splits the input tensor kv into key and value,
    and concatenates the processed RoPE to the key.

    For the notations below, seq_len is the length of sequence per batch for sbhd format,
    total_seq_len is the total length of the sequences for thd format.
    max_seq_len is the maximum length of the sequences in the input tensor.

    Args:
        kv: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        k_pos_emb: [seq_len, batch_size, 1, emb_dim] or [total_seq_len, 1, emb_dim]
        cos/sin: [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format
        rotary_interleaved: whether to apply RoPE interleaved, only supports False for now
        remove_interleaving: if True, output RoPE dims in non-interleaved layout

    Returns:
        key: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        value: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
    """
    return _FusedMLARoPEKVSplit.apply(
        kv,
        k_pos_emb,
        cos,
        sin,
        emb_dim,
        k_dim,
        v_dim,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
        rotary_interleaved,
        remove_interleaving,
    )


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated, prefer the new names above)
# ---------------------------------------------------------------------------
fused_apply_mla_rope_for_q = fused_mla_rope_inplace
fused_apply_mla_rope_for_kv = fused_mla_rope_kv_split
