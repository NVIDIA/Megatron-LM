# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton fused multimodal RoPE apply.

The fused path consumes the raw three-axis mRoPE frequencies with shape
``[3, batch, seq, rotary_dim / 2]`` and applies the rotation directly to a BSHD
tensor. It supports both Qwen2-VL section-based mRoPE and Qwen3.5-VL
stride-3 interleaved mRoPE layouts.
"""

from __future__ import annotations

from typing import List, Optional
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


def _smallest_power_of_2_at_least(x: int) -> int:
    block = 1
    while block < x:
        block *= 2
    return block


def _expected_interleaved_mrope_section(half_rotary_dim: int) -> tuple[int, int, int]:
    return ((half_rotary_dim + 2) // 3, (half_rotary_dim + 1) // 3, half_rotary_dim // 3)


def _validate_mrope_section(
    mrope_section: List[int], half_rotary_dim: int, interleaved_mrope: bool
) -> tuple[int, int, int]:
    assert len(mrope_section) == 3, f"mrope_section must have length 3, got {mrope_section}"

    sec_t, sec_h, sec_w = (int(section) for section in mrope_section)
    assert (
        min(sec_t, sec_h, sec_w) >= 0
    ), f"mrope_section values must be non-negative, got {mrope_section}"
    assert half_rotary_dim > 0, "raw mRoPE rotary dim must be greater than 0"
    assert (
        sec_t + sec_h + sec_w == half_rotary_dim
    ), f"mrope_section {mrope_section} must sum to rotary_dim / 2 = {half_rotary_dim}"
    if interleaved_mrope:
        expected = _expected_interleaved_mrope_section(half_rotary_dim)
        assert (sec_t, sec_h, sec_w) == expected, (
            f"interleaved mRoPE with rotary_dim / 2 = {half_rotary_dim} requires "
            f"mrope_section {list(expected)}, got {mrope_section}"
        )
    return sec_t, sec_h, sec_w


def _validate_mrope_inputs(
    t: torch.Tensor, freqs: torch.Tensor, mrope_section: List[int], interleaved_mrope: bool
) -> tuple[int, int, int, int, int, int, int, int]:
    assert t.dim() == 4, f"t must have shape [seq, batch, heads, head_dim], got {t.shape}"
    assert freqs.dim() == 4, (
        "raw mRoPE freqs must have shape [3, batch, seq, rotary_dim / 2], " f"got {freqs.shape}"
    )

    seq, batch, heads, head_dim = t.shape
    axes, freq_batch, freq_seq, half_rotary_dim = freqs.shape
    assert axes == 3, f"raw mRoPE freqs first dimension must be 3, got {axes}"
    assert (
        freq_batch == batch and freq_seq == seq
    ), f"freqs shape {tuple(freqs.shape)} is incompatible with t shape {tuple(t.shape)}"

    sec_t, sec_h, sec_w = _validate_mrope_section(mrope_section, half_rotary_dim, interleaved_mrope)

    rotary_dim = half_rotary_dim * 2
    assert (
        rotary_dim <= head_dim
    ), f"raw mRoPE rotary dim {rotary_dim} exceeds input head dim {head_dim}"
    return seq, batch, heads, head_dim, half_rotary_dim, sec_t, sec_h, sec_w


def _validate_mrope_thd_inputs(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    mrope_section: List[int],
    interleaved_mrope: bool,
    cp_size: int,
) -> tuple[int, int, int, int, int, int, int]:
    assert t.dim() == 3, f"t must have shape [tokens, heads, head_dim], got {t.shape}"
    assert freqs.dim() == 4, (
        "raw mRoPE freqs must have shape [3, 1, total_seqlen, rotary_dim / 2], "
        f"got {freqs.shape}"
    )
    assert cu_seqlens.dim() == 1, f"cu_seqlens must be 1D, got {cu_seqlens.shape}"

    tokens, heads, head_dim = t.shape
    axes, freq_batch, freq_seq, half_rotary_dim = freqs.shape
    assert axes == 3, f"raw mRoPE freqs first dimension must be 3, got {axes}"
    assert freq_batch == 1, (
        "raw mRoPE THD freqs must have singleton batch dimension, " f"got {freqs.shape}"
    )
    assert freq_seq == tokens * cp_size, (
        "raw mRoPE THD freqs sequence length must match local tokens times cp_size, "
        f"got freqs.shape[2]={freq_seq}, tokens={tokens}, cp_size={cp_size}"
    )

    sec_t, sec_h, sec_w = _validate_mrope_section(mrope_section, half_rotary_dim, interleaved_mrope)
    rotary_dim = half_rotary_dim * 2
    assert (
        rotary_dim <= head_dim
    ), f"raw mRoPE rotary dim {rotary_dim} exceeds input head dim {head_dim}"
    return tokens, heads, head_dim, half_rotary_dim, sec_t, sec_h, sec_w


def get_fused_mrope_unavailable_reason(
    t: Optional[torch.Tensor] = None,
    freqs: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
) -> Optional[str]:
    """Return why fused mRoPE cannot run, or None when it is launchable."""
    if not HAVE_TRITON:
        return "Triton is not available"
    if rotary_interleaved:
        return "rotary_interleaved=True is not supported"
    if t is None or freqs is None:
        return None
    if not t.is_cuda or not freqs.is_cuda:
        return "Triton fused mRoPE requires CUDA tensors"
    if t.device != freqs.device:
        return (
            "Triton fused mRoPE requires t and freqs on the same device, "
            f"got {t.device} and {freqs.device}"
        )
    if freqs.dtype != torch.float32:
        return f"raw mRoPE freqs must be float32, got {freqs.dtype}"
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"input dtype {t.dtype} is not supported"
    if t.stride(-1) != 1:
        return f"input head dimension must be contiguous, got stride {t.stride()}"
    try:
        capability = torch.cuda.get_device_capability(t.device)
    except RuntimeError as exc:
        return f"could not query CUDA device capability: {exc}"
    if capability < (7, 0):
        return f"requires CUDA compute capability >= 7.0, got {capability[0]}.{capability[1]}"
    if t.dtype == torch.bfloat16 and capability < (8, 0):
        return (
            "requires CUDA compute capability >= 8.0 for bfloat16 inputs, "
            f"got {capability[0]}.{capability[1]}"
        )
    return None


def get_fused_mrope_thd_unavailable_reason(
    t: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    freqs: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Optional[str]:
    """Return why fused THD mRoPE cannot run, or None when it is launchable."""
    if not HAVE_TRITON:
        return "Triton is not available"
    if rotary_interleaved:
        return "rotary_interleaved=True is not supported"
    if cp_size < 1:
        return f"cp_size must be positive, got {cp_size}"
    if cp_rank < 0 or cp_rank >= cp_size:
        return f"cp_rank must be in [0, {cp_size}), got {cp_rank}"
    if t is None or cu_seqlens is None or freqs is None:
        return None
    if t.dim() != 3:
        return (
            f"THD fused mRoPE expects t with shape [tokens, heads, head_dim], got {tuple(t.shape)}"
        )
    if freqs.dim() != 4:
        return (
            "raw mRoPE THD freqs must have shape [3, 1, total_seqlen, rotary_dim / 2], "
            f"got {tuple(freqs.shape)}"
        )
    if cu_seqlens.dim() != 1:
        return f"cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}"
    if not t.is_cuda or not freqs.is_cuda or not cu_seqlens.is_cuda:
        return "Triton fused THD mRoPE requires CUDA tensors"
    if t.device != freqs.device or t.device != cu_seqlens.device:
        return (
            "Triton fused THD mRoPE requires t, freqs, and cu_seqlens on the same device, "
            f"got {t.device}, {freqs.device}, and {cu_seqlens.device}"
        )
    if freqs.dtype != torch.float32:
        return f"raw mRoPE freqs must be float32, got {freqs.dtype}"
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"input dtype {t.dtype} is not supported"
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        return f"cu_seqlens dtype {cu_seqlens.dtype} is not supported"
    if t.stride(-1) != 1:
        return f"input head dimension must be contiguous, got stride {t.stride()}"
    if freqs.shape[0] != 3 or freqs.shape[1] != 1:
        return (
            "raw mRoPE THD freqs must have shape [3, 1, total_seqlen, rotary_dim / 2], "
            f"got {tuple(freqs.shape)}"
        )
    if cp_size > 1 and freqs.shape[2] % cp_size != 0:
        return (
            "raw mRoPE THD freqs sequence length must be divisible by context parallel size, "
            f"got freqs.shape[2]={freqs.shape[2]}, cp_size={cp_size}"
        )
    # Per-sequence divisibility is validated at the THD packing boundary by
    # rope_utils._get_thd_cp_splits(). Keep this reminder here because a future
    # direct fused-kernel path would need the same check before launch, but doing
    # it here would require a cu_seqlens GPU->CPU sync on the hot availability path:
    #
    # cu_seqlens_list = cu_seqlens.tolist()
    # for seq_start, seq_end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:]):
    #     if cp_size > 1 and (seq_end - seq_start) % cp_size != 0:
    #         return "THD sequence lengths must be divisible by context parallel size"
    if freqs.shape[2] != t.shape[0] * cp_size:
        return (
            "raw mRoPE THD freqs sequence length must match local tokens times cp_size, "
            f"got freqs.shape[2]={freqs.shape[2]}, tokens={t.shape[0]}, cp_size={cp_size}"
        )
    try:
        capability = torch.cuda.get_device_capability(t.device)
    except RuntimeError as exc:
        return f"could not query CUDA device capability: {exc}"
    if capability < (7, 0):
        return f"requires CUDA compute capability >= 7.0, got {capability[0]}.{capability[1]}"
    if t.dtype == torch.bfloat16 and capability < (8, 0):
        return (
            "requires CUDA compute capability >= 8.0 for bfloat16 inputs, "
            f"got {capability[0]}.{capability[1]}"
        )
    return None


def can_launch_fused_mrope(
    t: Optional[torch.Tensor] = None,
    freqs: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
) -> bool:
    """Return whether the Triton fused mRoPE kernel can be launched."""
    return get_fused_mrope_unavailable_reason(t, freqs, rotary_interleaved) is None


def can_launch_fused_mrope_thd(
    t: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    freqs: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> bool:
    """Return whether the Triton fused THD mRoPE kernel can be launched."""
    return (
        get_fused_mrope_thd_unavailable_reason(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=rotary_interleaved,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        is None
    )


def mrope_freqs_to_rotary_emb(
    freqs: torch.Tensor,
    mrope_section: List[int],
    interleaved_mrope: bool = False,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """Convert raw mRoPE freqs to the unfused RoPE embedding layout.

    Args:
        freqs: Raw mRoPE frequencies with shape ``[3, batch, seq, rotary_dim / 2]``.
        mrope_section: Temporal, height, and width channel sections.
        interleaved_mrope: Use Qwen3.5-VL stride-3 T/H/W layout when True. Use
            Qwen2-VL section layout when False.
        rotary_interleaved: Use adjacent-pair RoPE layout when True. This is
            available for reference conversion; fused Triton currently supports
            split-half layout only.

    Returns:
        Tensor with shape ``[seq, batch, 1, rotary_dim]``.
    """
    assert freqs.dim() == 4, (
        "raw mRoPE freqs must have shape [3, batch, seq, rotary_dim / 2], " f"got {freqs.shape}"
    )
    assert freqs.size(0) == 3, f"raw mRoPE freqs first dimension must be 3, got {freqs.size(0)}"
    assert len(mrope_section) == 3, f"mrope_section must have length 3, got {mrope_section}"

    half_rotary_dim = freqs.size(-1)
    sec_t, sec_h, sec_w = _validate_mrope_section(mrope_section, half_rotary_dim, interleaved_mrope)

    if interleaved_mrope:
        freqs_out = freqs[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = int(mrope_section[dim_idx]) * 3
            idx = slice(offset, length, 3)
            freqs_out[..., idx] = freqs[dim_idx, ..., idx]
        if rotary_interleaved:
            batch = freqs_out.shape[0]
            emb = torch.stack(
                (freqs_out.reshape(batch, -1, 1), freqs_out.reshape(batch, -1, 1)), dim=-1
            )
            emb = emb.view(batch, freqs_out.shape[1], -1)
        else:
            emb = torch.cat((freqs_out, freqs_out), dim=-1)
    else:
        if rotary_interleaved:
            batch = freqs.shape[1]
            emb = torch.stack(
                (freqs.reshape(3, batch, -1, 1), freqs.reshape(3, batch, -1, 1)), dim=-1
            ).view(3, batch, freqs.shape[2], -1)
            mrope_section_doubled = list(mrope_section) * 2
            emb = torch.cat(
                [chunk[i % 3] for i, chunk in enumerate(emb.split(mrope_section_doubled, dim=-1))],
                dim=-1,
            )
        else:
            freqs_out = torch.empty_like(freqs[0])
            freqs_out[..., :sec_t] = freqs[0, ..., :sec_t]
            freqs_out[..., sec_t : sec_t + sec_h] = freqs[1, ..., sec_t : sec_t + sec_h]
            freqs_out[..., sec_t + sec_h :] = freqs[2, ..., sec_t + sec_h :]
            emb = torch.cat((freqs_out, freqs_out), dim=-1)
    return emb[..., None, :].transpose(0, 1).contiguous()


@triton.jit
def _mrope_axis(
    k,
    SEC_T: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
    INTERLEAVED_MROPE: tl.constexpr,
):
    if INTERLEAVED_MROPE:
        rem = k % 3
        section_idx = k // 3
        is_h = (rem == 1) & (section_idx < SEC_H)
        is_w = (rem == 2) & (section_idx < SEC_W)
        return tl.where(is_h, 1, tl.where(is_w, 2, 0))

    is_h = (k >= SEC_T) & (k < (SEC_T + SEC_H))
    is_w = k >= (SEC_T + SEC_H)
    return tl.where(is_h, 1, tl.where(is_w, 2, 0))


@triton.jit
def _fused_mrope_kernel(
    T,
    FREQS,
    OUT,
    t_s_seq,
    t_s_batch,
    t_s_head,
    t_s_dim,
    f_s_axis,
    f_s_batch,
    f_s_seq,
    f_s_dim,
    o_s_seq,
    o_s_batch,
    o_s_head,
    o_s_dim,
    HEAD_DIM: tl.constexpr,
    HALF_ROTARY_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
    SEC_T: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
    INTERLEAVED_MROPE: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_PASS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    k = tl.arange(0, BLOCK_HALF)
    mask = k < HALF_ROTARY_DIM

    axis = _mrope_axis(k, SEC_T, SEC_H, SEC_W, INTERLEAVED_MROPE)

    freqs_offset = axis * f_s_axis + batch_idx * f_s_batch + seq_idx * f_s_seq + k * f_s_dim
    freqs = tl.load(FREQS + freqs_offset, mask=mask, other=0.0)
    # Match PyTorch pointwise dtype semantics: cast cos/sin before the multiply.
    cos_v = tl.cos(freqs).to(OUT.dtype.element_ty)
    sin_v = tl.sin(freqs).to(OUT.dtype.element_ty)
    if INVERSE:
        sin_v = -sin_v

    t_base = T + seq_idx * t_s_seq + batch_idx * t_s_batch + head_idx * t_s_head
    out_base = OUT + seq_idx * o_s_seq + batch_idx * o_s_batch + head_idx * o_s_head

    if ROTARY_INTERLEAVED:
        lo_offset = (2 * k) * t_s_dim
        hi_offset = (2 * k + 1) * t_s_dim
        out_lo_offset = (2 * k) * o_s_dim
        out_hi_offset = (2 * k + 1) * o_s_dim
    else:
        lo_offset = k * t_s_dim
        hi_offset = (k + HALF_ROTARY_DIM) * t_s_dim
        out_lo_offset = k * o_s_dim
        out_hi_offset = (k + HALF_ROTARY_DIM) * o_s_dim

    t_lo = tl.load(t_base + lo_offset, mask=mask, other=0.0).to(OUT.dtype.element_ty)
    t_hi = tl.load(t_base + hi_offset, mask=mask, other=0.0).to(OUT.dtype.element_ty)

    lo_cos = (t_lo * cos_v).to(OUT.dtype.element_ty)
    hi_sin = (t_hi * sin_v).to(OUT.dtype.element_ty)
    hi_cos = (t_hi * cos_v).to(OUT.dtype.element_ty)
    lo_sin = (t_lo * sin_v).to(OUT.dtype.element_ty)

    out_lo = (lo_cos - hi_sin).to(OUT.dtype.element_ty)
    out_hi = (hi_cos + lo_sin).to(OUT.dtype.element_ty)

    tl.store(out_base + out_lo_offset, out_lo, mask=mask)
    tl.store(out_base + out_hi_offset, out_hi, mask=mask)

    if PASS_DIM > 0:
        pass_idx = tl.arange(0, BLOCK_PASS)
        pass_mask = pass_idx < PASS_DIM
        src_dim = 2 * HALF_ROTARY_DIM + pass_idx
        pass_values = tl.load(t_base + src_dim * t_s_dim, mask=pass_mask, other=0.0)
        tl.store(out_base + src_dim * o_s_dim, pass_values, mask=pass_mask)


@triton.jit
def _fused_mrope_thd_kernel(
    T,
    CU_SEQLENS,
    FREQS,
    OUT,
    t_s_token,
    t_s_head,
    t_s_dim,
    cu_s_idx,
    f_s_axis,
    f_s_seq,
    f_s_dim,
    o_s_token,
    o_s_head,
    o_s_dim,
    NUM_SEQS,
    HEAD_DIM: tl.constexpr,
    HALF_ROTARY_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
    SEC_T: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
    INTERLEAVED_MROPE: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    INVERSE: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    FP32_COMPUTE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_PASS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    freq_seq_idx = token_idx
    seq_i = 0
    while seq_i < NUM_SEQS:
        global_start = tl.load(CU_SEQLENS + seq_i * cu_s_idx)
        global_end = tl.load(CU_SEQLENS + (seq_i + 1) * cu_s_idx)
        local_start = global_start // CP_SIZE
        local_end = global_end // CP_SIZE
        in_seq = (token_idx >= local_start) & (token_idx < local_end)
        local_offset = token_idx - local_start

        if CP_SIZE > 1:
            local_seq_len = local_end - local_start
            first_cp_seg = (local_seq_len + 1) // 2
            second_cp_seg = local_seq_len // 2
            first_freq_idx = global_start + CP_RANK * first_cp_seg + local_offset
            second_freq_idx = (
                global_end - (CP_RANK + 1) * second_cp_seg + (local_offset - first_cp_seg)
            )
            seq_freq_idx = tl.where(local_offset < first_cp_seg, first_freq_idx, second_freq_idx)
        else:
            seq_freq_idx = global_start + local_offset

        freq_seq_idx = tl.where(in_seq, seq_freq_idx, freq_seq_idx)
        seq_i += 1

    k = tl.arange(0, BLOCK_HALF)
    mask = k < HALF_ROTARY_DIM
    axis = _mrope_axis(k, SEC_T, SEC_H, SEC_W, INTERLEAVED_MROPE)

    freqs_offset = axis * f_s_axis + freq_seq_idx * f_s_seq + k * f_s_dim
    freqs = tl.load(FREQS + freqs_offset, mask=mask, other=0.0)
    if FP32_COMPUTE:
        cos_v = tl.cos(freqs)
        sin_v = tl.sin(freqs)
    else:
        cos_v = tl.cos(freqs).to(OUT.dtype.element_ty)
        sin_v = tl.sin(freqs).to(OUT.dtype.element_ty)
    if INVERSE:
        sin_v = -sin_v

    t_base = T + token_idx * t_s_token + head_idx * t_s_head
    out_base = OUT + token_idx * o_s_token + head_idx * o_s_head

    if ROTARY_INTERLEAVED:
        lo_offset = (2 * k) * t_s_dim
        hi_offset = (2 * k + 1) * t_s_dim
        out_lo_offset = (2 * k) * o_s_dim
        out_hi_offset = (2 * k + 1) * o_s_dim
    else:
        lo_offset = k * t_s_dim
        hi_offset = (k + HALF_ROTARY_DIM) * t_s_dim
        out_lo_offset = k * o_s_dim
        out_hi_offset = (k + HALF_ROTARY_DIM) * o_s_dim

    if FP32_COMPUTE:
        t_lo = tl.load(t_base + lo_offset, mask=mask, other=0.0).to(tl.float32)
        t_hi = tl.load(t_base + hi_offset, mask=mask, other=0.0).to(tl.float32)
    else:
        t_lo = tl.load(t_base + lo_offset, mask=mask, other=0.0).to(OUT.dtype.element_ty)
        t_hi = tl.load(t_base + hi_offset, mask=mask, other=0.0).to(OUT.dtype.element_ty)

    if FP32_COMPUTE:
        lo_cos = t_lo * cos_v
        hi_sin = t_hi * sin_v
        hi_cos = t_hi * cos_v
        lo_sin = t_lo * sin_v
    else:
        lo_cos = (t_lo * cos_v).to(OUT.dtype.element_ty)
        hi_sin = (t_hi * sin_v).to(OUT.dtype.element_ty)
        hi_cos = (t_hi * cos_v).to(OUT.dtype.element_ty)
        lo_sin = (t_lo * sin_v).to(OUT.dtype.element_ty)

    if FP32_COMPUTE:
        out_lo = lo_cos - hi_sin
        out_hi = hi_cos + lo_sin
    else:
        out_lo = (lo_cos - hi_sin).to(OUT.dtype.element_ty)
        out_hi = (hi_cos + lo_sin).to(OUT.dtype.element_ty)

    tl.store(out_base + out_lo_offset, out_lo, mask=mask)
    tl.store(out_base + out_hi_offset, out_hi, mask=mask)

    if PASS_DIM > 0:
        pass_idx = tl.arange(0, BLOCK_PASS)
        pass_mask = pass_idx < PASS_DIM
        src_dim = 2 * HALF_ROTARY_DIM + pass_idx
        pass_values = tl.load(t_base + src_dim * t_s_dim, mask=pass_mask, other=0.0)
        tl.store(out_base + src_dim * o_s_dim, pass_values, mask=pass_mask)


def _launch_fused_mrope(
    t: torch.Tensor,
    freqs: torch.Tensor,
    mrope_section: List[int],
    interleaved_mrope: bool,
    rotary_interleaved: bool,
    inverse: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    unavailable_reason = get_fused_mrope_unavailable_reason(t, freqs, rotary_interleaved)
    assert unavailable_reason is None, unavailable_reason

    seq, batch, heads, head_dim, half_rotary_dim, sec_t, sec_h, sec_w = _validate_mrope_inputs(
        t, freqs, mrope_section, interleaved_mrope
    )

    if out is None:
        out = torch.empty_like(t)
    else:
        assert out.shape == t.shape and out.dtype == t.dtype
        assert (
            out.stride(-1) == 1
        ), f"fused mRoPE requires output contiguous head dimension, got {out.stride()}"

    block_half = _smallest_power_of_2_at_least(half_rotary_dim)
    pass_dim = head_dim - (2 * half_rotary_dim)
    block_pass = _smallest_power_of_2_at_least(max(pass_dim, 1))

    grid = (seq, batch, heads)
    _fused_mrope_kernel[grid](
        t,
        freqs,
        out,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        t.stride(3),
        freqs.stride(0),
        freqs.stride(1),
        freqs.stride(2),
        freqs.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        HEAD_DIM=head_dim,
        HALF_ROTARY_DIM=half_rotary_dim,
        PASS_DIM=pass_dim,
        SEC_T=sec_t,
        SEC_H=sec_h,
        SEC_W=sec_w,
        INTERLEAVED_MROPE=interleaved_mrope,
        ROTARY_INTERLEAVED=rotary_interleaved,
        INVERSE=inverse,
        BLOCK_HALF=block_half,
        BLOCK_PASS=block_pass,
        num_warps=4,
    )
    return out


def _launch_fused_mrope_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    mrope_section: List[int],
    interleaved_mrope: bool,
    rotary_interleaved: bool,
    inverse: bool,
    cp_size: int,
    cp_rank: int,
    fp32_compute: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    unavailable_reason = get_fused_mrope_thd_unavailable_reason(
        t,
        cu_seqlens,
        freqs,
        rotary_interleaved=rotary_interleaved,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )
    assert unavailable_reason is None, unavailable_reason

    tokens, heads, head_dim, half_rotary_dim, sec_t, sec_h, sec_w = _validate_mrope_thd_inputs(
        t, cu_seqlens, freqs, mrope_section, interleaved_mrope, cp_size
    )

    if out is None:
        out = torch.empty_like(t)
    else:
        assert out.shape == t.shape and out.dtype == t.dtype
        assert (
            out.stride(-1) == 1
        ), f"fused THD mRoPE requires output contiguous head dimension, got {out.stride()}"

    block_half = _smallest_power_of_2_at_least(half_rotary_dim)
    pass_dim = head_dim - (2 * half_rotary_dim)
    block_pass = _smallest_power_of_2_at_least(max(pass_dim, 1))
    num_seqs = cu_seqlens.numel() - 1

    grid = (tokens, heads)
    _fused_mrope_thd_kernel[grid](
        t,
        cu_seqlens,
        freqs,
        out,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        cu_seqlens.stride(0),
        freqs.stride(0),
        freqs.stride(2),
        freqs.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_seqs,
        HEAD_DIM=head_dim,
        HALF_ROTARY_DIM=half_rotary_dim,
        PASS_DIM=pass_dim,
        SEC_T=sec_t,
        SEC_H=sec_h,
        SEC_W=sec_w,
        INTERLEAVED_MROPE=interleaved_mrope,
        ROTARY_INTERLEAVED=rotary_interleaved,
        INVERSE=inverse,
        CP_SIZE=cp_size,
        CP_RANK=cp_rank,
        FP32_COMPUTE=fp32_compute,
        BLOCK_HALF=block_half,
        BLOCK_PASS=block_pass,
        num_warps=4,
    )
    return out


class _FusedMRoPE(torch.autograd.Function):
    """Autograd wrapper for fused mRoPE.

    The raw frequency table is generated from position IDs and inverse frequencies,
    so gradients are only propagated to the rotated tensor.
    """

    @staticmethod
    def forward(ctx, t, freqs, mrope_section, interleaved_mrope, rotary_interleaved):
        assert not freqs.requires_grad, "fused mRoPE expects non-gradient raw frequency tensors"
        ctx.mrope_section = tuple(int(section) for section in mrope_section)
        ctx.interleaved_mrope = bool(interleaved_mrope)
        ctx.rotary_interleaved = bool(rotary_interleaved)
        ctx.save_for_backward(freqs)
        return _launch_fused_mrope(
            t,
            freqs,
            ctx.mrope_section,
            ctx.interleaved_mrope,
            ctx.rotary_interleaved,
            inverse=False,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (freqs,) = ctx.saved_tensors
        grad_input = _launch_fused_mrope(
            grad_output.contiguous(),
            freqs,
            ctx.mrope_section,
            ctx.interleaved_mrope,
            ctx.rotary_interleaved,
            inverse=True,
        )
        return grad_input, None, None, None, None


class _FusedMRoPETHD(torch.autograd.Function):
    """Autograd wrapper for fused THD mRoPE."""

    @staticmethod
    def forward(
        ctx,
        t,
        cu_seqlens,
        freqs,
        mrope_section,
        interleaved_mrope,
        rotary_interleaved,
        cp_size,
        cp_rank,
        fp32_compute,
    ):
        assert not freqs.requires_grad, "fused THD mRoPE expects non-gradient raw frequency tensors"
        ctx.mrope_section = tuple(int(section) for section in mrope_section)
        ctx.interleaved_mrope = bool(interleaved_mrope)
        ctx.rotary_interleaved = bool(rotary_interleaved)
        ctx.cp_size = int(cp_size)
        ctx.cp_rank = int(cp_rank)
        ctx.fp32_compute = bool(fp32_compute)
        ctx.save_for_backward(cu_seqlens, freqs)
        return _launch_fused_mrope_thd(
            t,
            cu_seqlens,
            freqs,
            ctx.mrope_section,
            ctx.interleaved_mrope,
            ctx.rotary_interleaved,
            inverse=False,
            cp_size=ctx.cp_size,
            cp_rank=ctx.cp_rank,
            fp32_compute=ctx.fp32_compute,
        )

    @staticmethod
    def backward(ctx, grad_output):
        cu_seqlens, freqs = ctx.saved_tensors
        grad_input = _launch_fused_mrope_thd(
            grad_output.contiguous(),
            cu_seqlens,
            freqs,
            ctx.mrope_section,
            ctx.interleaved_mrope,
            ctx.rotary_interleaved,
            inverse=True,
            cp_size=ctx.cp_size,
            cp_rank=ctx.cp_rank,
            fp32_compute=ctx.fp32_compute,
        )
        return grad_input, None, None, None, None, None, None, None, None


def fused_apply_mrope(
    t: torch.Tensor,
    freqs: torch.Tensor,
    mrope_section: List[int],
    interleaved_mrope: bool = False,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """Apply multimodal RoPE with a fused Triton kernel.

    Args:
        t: Input tensor with shape ``[seq, batch, heads, head_dim]``.
        freqs: Raw mRoPE frequencies with shape ``[3, batch, seq, rotary_dim / 2]``.
        mrope_section: Temporal, height, and width channel sections.
        interleaved_mrope: Use Qwen3.5-VL stride-3 T/H/W layout when True. Use
            Qwen2-VL section layout when False.
        rotary_interleaved: Must be False. The integrated fused mRoPE path
            currently supports split-half RoPE layout.

    Returns:
        Rotated tensor with the same shape and dtype as ``t``.
    """
    return _FusedMRoPE.apply(t, freqs, mrope_section, interleaved_mrope, rotary_interleaved)


def fused_apply_mrope_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    mrope_section: List[int],
    interleaved_mrope: bool = False,
    rotary_interleaved: bool = False,
    cp_size: int = 1,
    cp_rank: int = 0,
    fp32_compute: bool = False,
) -> torch.Tensor:
    """Apply multimodal RoPE to THD-packed tensors with a fused Triton kernel.

    Args:
        t: Input tensor with shape ``[total_tokens, heads, head_dim]``.
        cu_seqlens: Global cumulative sequence lengths for the packed batch.
        freqs: Raw mRoPE frequencies with shape ``[3, 1, total_seqlen, rotary_dim / 2]``.
        mrope_section: Temporal, height, and width channel sections.
        interleaved_mrope: Use Qwen3.5-VL stride-3 T/H/W layout when True.
        rotary_interleaved: Must be False.
        cp_size: Context parallel world size for THD token mapping.
        cp_rank: Context parallel rank for THD token mapping.
        fp32_compute: Apply the rotary math in fp32 and cast directly to output dtype.

    Returns:
        Rotated tensor with the same shape and dtype as ``t``.
    """
    return _FusedMRoPETHD.apply(
        t,
        cu_seqlens,
        freqs,
        mrope_section,
        interleaved_mrope,
        rotary_interleaved,
        cp_size,
        cp_rank,
        fp32_compute,
    )


def is_fused_mrope_available() -> bool:
    """Return whether the Triton mRoPE fusion can be used on this host.

    This does not check tensor device, dtype, stride, or CUDA capability. Use
    ``can_launch_fused_mrope`` or ``get_fused_mrope_unavailable_reason`` with
    tensors before dispatching to the fused kernel.
    """
    if not torch.cuda.is_available():
        return False
    return can_launch_fused_mrope()
