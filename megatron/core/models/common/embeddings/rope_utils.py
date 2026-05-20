# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

import logging

import torch
from torch import Tensor

from megatron.core import parallel_state

logger = logging.getLogger(__name__)

try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
except ImportError:
    fused_apply_rotary_pos_emb = None


try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
except ImportError:
    fused_apply_rotary_pos_emb_thd = None


try:
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash
except ImportError:
    apply_rotary_emb_flash = None


__all__ = [
    'apply_rotary_pos_emb',
    'apply_rotary_emb_flash',
    'apply_rotary_pos_emb_with_cos_sin',
    'fused_apply_rotary_pos_emb',
    'fused_apply_rotary_pos_emb_thd',
    'get_pos_emb_on_this_cp_rank',
]


def get_pos_emb_on_this_cp_rank(
    pos_emb: Tensor, seq_dim: int, cp_group: torch.distributed.ProcessGroup
) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
        cp_group (torch.distributed.ProcessGroup): The context parallel group
    """
    if cp_group is None:
        raise ValueError("cp_group must be provided to get positional embedding per CP rank")
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
    multi_latent_attention: Optional[bool] = None,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]
        rotary_interleaved (bool): Whether to apply interleaving in the rotate half function.
        mla_rotary_interleaved (bool): Whether to apply MLA-style interleaving for RoPE.
        mscale (float): The scaling factor for the RoPE.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated. Please use mla_rotary_interleaved instead.",
            DeprecationWarning,
        )
        mla_rotary_interleaved = multi_latent_attention

    # Some callers may pass freqs with an extra singleton axis, e.g.
    # t: [s, b, d] and freqs: [s, 1, 1, d]. In that case, broadcasting would
    # accidentally expand to [s, s, b, d]. Squeeze the extra singleton axis to
    # keep freqs rank aligned with t.
    if freqs.dim() == t.dim() + 1 and freqs.size(-2) == 1:
        freqs = freqs.squeeze(-2)

    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if mla_rotary_interleaved:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
    if inverse:
        sin_ = -sin_

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    # Fallback to original permutation
    # DSv4 applies rope on V and O, so we need to uninterleave the tensor.
    # The existing MLA code is safe because the dot product is permutation-invariant.
    if mla_rotary_interleaved and mla_output_remove_interleaving:
        x1, x2 = torch.chunk(t, 2, dim=-1)
        t = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)

    return torch.cat((t, t_pass), dim=-1)


def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor, offset: int = 0
) -> Tensor:
    """Get the correct frequency slice for this context parallel rank with optional sequence offset.

    Args:
        cp_rank: Current context parallel rank
        cp_size: Total context parallel size
        x: Input tensor for current sequence
        freqs: Frequency tensor - either full batch positions or max sequence length
        offset: Starting position offset for this sequence in the original batch (default: 0)

    Returns:
        Tensor: Frequency slice corresponding to this CP rank's portion of the sequence

    Note:
        This function supports two modes based on the offset parameter:
        1. offset > 0: Exact mapping mode - freqs contains all positions across all sequences.
           The offset ensures each sequence gets frequencies from its actual position within
           the overall batch. Critical for non-1D RoPE in VLMs where spatial positions matter.
        2. offset = 0: Traditional mode - freqs contains only max sequence length positions.
           All sequences use frequencies starting from position 0, preserving backward
           compatibility.
    """
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        # Apply offset to both forward and backward segments for context parallelism
        # offset=0: traditional behavior, freqs[0:cp_seg] and freqs[...]
        # offset>0: exact mapping, freqs[offset+0:offset+cp_seg] and freqs[offset+...]
        return torch.cat(
            [
                freqs[offset + cp_rank * cp_seg : offset + (cp_rank + 1) * cp_seg],
                freqs[
                    offset
                    + full_seqlen
                    - (cp_rank + 1) * cp_seg : offset
                    + full_seqlen
                    - cp_rank * cp_seg
                ],
            ]
        )
    else:
        # For single context parallel rank:
        # offset=0: use freqs[0:x.size(0)] (traditional)
        # offset>0: use freqs[offset:offset+x.size(0)] (exact mapping)
        return freqs[offset : offset + x.size(0)]


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
    cp_group: torch.distributed.ProcessGroup = None,
    multi_latent_attention: Optional[bool] = None,
    max_seqlen: Optional[int] = None,
) -> Tensor:
    """Apply RoPE for `thd` format using pure CUDA ops (CUDA Graph compatible).

    Replaces the original Python-loop + .tolist() implementation with vectorized
    CUDA operations. No GPU->CPU syncs, compatible with CUDA Graph capture.

    Args:
        t (Tensor): Input tensor of shape [total_tokens, h, d]
        cu_seqlens (Tensor): Cumulative sequence lengths, shape [num_seqs + 1], int32.
        freqs (Tensor): RoPE frequencies, shape [max_s, 1, 1, d] or [total_tokens, 1, 1, d]
        cp_group: Context parallel group
        max_seqlen: Global max sequence length for this packed batch when known.

    Returns:
        Tensor: Shape [total_tokens, h, d]. Input with RoPE applied.
    """
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated. Please use mla_rotary_interleaved instead.",
            DeprecationWarning,
        )
        mla_rotary_interleaved = multi_latent_attention

    if cp_group is None:
        raise ValueError("cp_group must be provided for THD format RoPE")
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    total_tokens = t.shape[0]
    device = t.device

    token_pos = torch.arange(total_tokens, device=device, dtype=torch.int64)

    # `cu_seqlens` describes the global packed sequence. With CP, `t` is already
    # CP-partitioned, so build a local cumulative-length view before assigning
    # local tokens to packed sequences.
    cu_seqlens_i64 = cu_seqlens.to(torch.int64)
    global_seq_lens = cu_seqlens_i64[1:] - cu_seqlens_i64[:-1]
    local_seq_lens = global_seq_lens // cp_size if cp_size > 1 else global_seq_lens
    local_cu_seqlens = torch.zeros_like(cu_seqlens_i64)
    local_cu_seqlens[1:] = torch.cumsum(local_seq_lens, dim=0)

    # `searchsorted(..., right=True) - 1` returns the local sequence index. The
    # clamp guards padded tokens that sit beyond the final real local token; they
    # get a harmless frequency and are later masked out.
    seq_idx = torch.searchsorted(local_cu_seqlens, token_pos, right=True) - 1
    seq_idx = seq_idx.clamp(min=0, max=cu_seqlens.shape[0] - 2)

    local_seq_start = local_cu_seqlens[seq_idx]
    local_pos = token_pos - local_seq_start
    local_seq_len = local_seq_lens[seq_idx]
    global_seq_start = cu_seqlens_i64[seq_idx]

    if cp_size > 1:
        cp_seg = local_seq_len // 2
        full_seqlen = local_seq_len * cp_size
        is_first_half = local_pos < cp_seg
        freq_pos = torch.where(
            is_first_half,
            cp_rank * cp_seg + local_pos,
            full_seqlen - (cp_rank + 1) * cp_seg + (local_pos - cp_seg),
        )
    else:
        freq_pos = local_pos.to(torch.int64)

    if max_seqlen is None:
        exact_packed_freqs = freqs.dim() >= 1 and cp_size == 1 and freqs.size(0) > total_tokens
    else:
        exact_packed_freqs = freqs.dim() >= 1 and freqs.size(0) > max_seqlen
    if exact_packed_freqs:
        # `freqs` covers all positions across all sequences (used for non-1D
        # RoPE / VLMs); shift by the per-sequence start offset so each token
        # samples its absolute position. When `freqs` only spans one max-len
        # sequence, no shift is needed.
        freq_pos = freq_pos + global_seq_start

    # Same rationale as the seq_idx clamp above: padded positions can index
    # past `freqs`; they receive a known wrong-but-harmless freq that gets
    # masked away. If you suspect a real out-of-range bug, swap clamp for an
    # assert during development.
    freq_pos = freq_pos.clamp(min=0, max=freqs.shape[0] - 1)
    freqs_packed = freqs[freq_pos]

    return _apply_rotary_pos_emb_bshd(
        t.unsqueeze(1),
        freqs_packed,
        rotary_interleaved=rotary_interleaved,
        mla_rotary_interleaved=mla_rotary_interleaved,
        mscale=mscale,
        inverse=inverse,
        mla_output_remove_interleaving=mla_output_remove_interleaving,
    ).squeeze(1)


def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
    mla_rotary_interleaved: bool = False,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
    max_seqlen: Optional[int] = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    global fused_apply_rotary_pos_emb, fused_apply_rotary_pos_emb_thd

    # Keep for backward compatibility. Will deprecate in the future.
    if cp_group is None:
        cp_group = parallel_state.get_context_parallel_group()

    if config.apply_rope_fusion:
        if cu_seqlens is None:
            # NOTE: TE backends do not support mRoPE in bshd format when bs > 1.
            use_unfused = False
            if config.mrope_section is not None and freqs.shape[1] > 1:
                # TODO: Add a check in TransformerConfig and remove this unfused implementation.
                warnings.warn(
                    "apply_rope_fusion does not support mRoPE in bshd format when bs > 1. "
                    "Please set apply_rope_fusion to false. This will become an error in v0.16."
                )
                use_unfused = True
            if mscale != 1.0:
                warnings.warn(
                    f"mscale={mscale} is not supported by TE's fused RoPE. "
                    "Using unfused implementation."
                )
                use_unfused = True
            if mla_rotary_interleaved:
                warnings.warn(
                    "apply_rope_fusion does not support MLA-style interleaving in RoPE."
                    "Using unfused implementation."
                )
                use_unfused = True
            if inverse:
                warnings.warn(
                    "inverse RoPE is not supported by TE's fused RoPE. "
                    "Using unfused implementation."
                )
                use_unfused = True
            if not use_unfused:
                assert fused_apply_rotary_pos_emb is not None, "apply_rope_fusion is not available."
                return fused_apply_rotary_pos_emb(t, freqs, interleaved=config.rotary_interleaved)
        else:
            assert fused_apply_rotary_pos_emb_thd is not None, "apply_rope_fusion is not available."
            return fused_apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                cp_size=cp_group.size(),
                cp_rank=cp_group.rank(),
                interleaved=config.rotary_interleaved,
            )
    # use unfused implementation
    if cu_seqlens is None:
        return _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
        )
    else:
        return _apply_rotary_pos_emb_thd(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            cp_group=cp_group,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
            max_seqlen=max_seqlen,
        )


def apply_rotary_pos_emb_with_cos_sin(
    t: Tensor, cos: Tensor, sin: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """
    This function applies rotary positional embedding to the target tensor t
    using precomputed cos and sin of size (seq_len, d_rot / 2)
    """
    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    if apply_rotary_emb_flash is None:
        # Combine cos and sin into freqs
        freqs = torch.stack([cos, sin], dim=-1).flatten(start_dim=-2)

        # Expand freqs to match t's shape
        while freqs.dim() < t.dim():
            freqs = freqs.unsqueeze(1)
        freqs = freqs.expand(t.shape[:-1] + (-1,))

        y = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=False,
            mscale=1.0,
        )
    else:
        # Use Flash Attention's optimized kernel for rotary embedding
        t = t.permute(1, 0, 2, 3)
        y = apply_rotary_emb_flash(t, cos, sin, rotary_interleaved)
        y = y.permute(1, 0, 2, 3)

    return y
