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
_ROPE_FUSION_FALLBACK_WARNINGS: set[str] = set()

try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
except ImportError:
    fused_apply_rotary_pos_emb = None


try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
except ImportError:
    fused_apply_rotary_pos_emb_thd = None


try:
    from megatron.core.fusions.fused_mrope import (
        can_launch_fused_mrope_thd,
        fused_apply_mrope,
        fused_apply_mrope_thd,
        get_fused_mrope_thd_unavailable_reason,
        get_fused_mrope_unavailable_reason,
        is_fused_mrope_available,
        mrope_freqs_to_rotary_emb,
    )
except ImportError:
    can_launch_fused_mrope_thd = None
    fused_apply_mrope = None
    fused_apply_mrope_thd = None
    get_fused_mrope_thd_unavailable_reason = None
    get_fused_mrope_unavailable_reason = None
    is_fused_mrope_available = None
    mrope_freqs_to_rotary_emb = None


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
    'can_launch_fused_mrope_thd',
    'fused_apply_mrope',
    'fused_apply_mrope_thd',
    'get_fused_mrope_thd_unavailable_reason',
    'get_fused_mrope_unavailable_reason',
    'is_fused_mrope_available',
    'mrope_freqs_to_rotary_emb',
    'get_pos_emb_on_this_cp_rank',
]


def _is_raw_mrope_freqs(t: Tensor, freqs: Tensor, config: TransformerConfig) -> bool:
    """Return whether freqs is the raw 3-axis mRoPE tensor for fused apply."""
    if config.mrope_section is None or freqs.dim() != 4 or freqs.shape[0] != 3:
        return False
    if sum(config.mrope_section) != freqs.shape[-1] or freqs.shape[-1] * 2 > t.shape[-1]:
        return False
    if t.dim() == 4:
        return freqs.shape[1] == t.shape[1] and freqs.shape[2] == t.shape[0]
    if t.dim() == 3:
        return freqs.shape[1] == 1
    return False


def _is_raw_mrope_freqs_thd(
    t: Tensor, freqs: Tensor, cu_seqlens: Tensor, config: TransformerConfig, cp_size: int
) -> bool:
    """Return whether freqs is raw mRoPE for THD layout, or fail on raw-like bad shapes."""
    if config.mrope_section is None or freqs.dim() != 4 or freqs.shape[0] != 3:
        return False
    if t.dim() != 3:
        raise ValueError(
            f"raw mRoPE THD expects t with shape [tokens, heads, head_dim], got {tuple(t.shape)}"
        )
    if sum(config.mrope_section) != freqs.shape[-1] or freqs.shape[-1] * 2 > t.shape[-1]:
        return False

    if freqs.shape[1] != 1:
        raise ValueError(
            "raw mRoPE THD freqs must have singleton batch dimension with shape "
            f"[3, 1, total_seqlen, rotary_dim / 2], got {tuple(freqs.shape)}"
        )
    if cp_size > 1 and freqs.shape[2] % cp_size != 0:
        raise ValueError(
            "raw mRoPE THD freqs sequence length must be divisible by context parallel size, "
            f"got freqs.shape[2]={freqs.shape[2]}, cp_size={cp_size}"
        )
    expected_total_seqlen = t.shape[0] * cp_size
    if freqs.shape[2] != expected_total_seqlen:
        raise ValueError(
            "raw mRoPE THD freqs sequence length must match local tokens times cp_size, "
            f"got freqs.shape[2]={freqs.shape[2]}, tokens={t.shape[0]}, cp_size={cp_size}"
        )
    if cu_seqlens.dim() != 1:
        raise ValueError(f"raw mRoPE THD cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}")
    return True


def _raw_mrope_freqs_to_emb(freqs: Tensor, config: TransformerConfig) -> Tensor:
    assert mrope_freqs_to_rotary_emb is not None, "mRoPE frequency conversion is unavailable."
    return mrope_freqs_to_rotary_emb(
        freqs,
        config.mrope_section,
        interleaved_mrope=config.mrope_interleaved,
        rotary_interleaved=config.rotary_interleaved,
    )


def _warn_rope_fusion_fallback_once(key: str, message: str) -> None:
    if key in _ROPE_FUSION_FALLBACK_WARNINGS:
        return
    _ROPE_FUSION_FALLBACK_WARNINGS.add(key)
    warnings.warn(message, stacklevel=2)


def _fused_mrope_unavailable_warning_key(reason: str, thd: bool = False) -> str:
    prefix = "triton-mrope-thd-unavailable" if thd else "triton-mrope-unavailable"
    reason_lower = reason.lower()
    if "triton is not available" in reason_lower:
        category = "import"
    elif "cuda tensors" in reason_lower or "same device" in reason_lower:
        category = "device"
    elif "dtype" in reason_lower or "float32" in reason_lower:
        category = "dtype"
    elif "stride" in reason_lower or "contiguous" in reason_lower:
        category = "stride"
    elif "capability" in reason_lower:
        category = "capability"
    elif "rotary_interleaved" in reason_lower:
        category = "rotary-interleaved"
    else:
        category = "other"
    return f"{prefix}-{category}"


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
        first_cp_seg = (x.size(0) + 1) // 2
        second_cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        # Apply offset to both forward and backward segments for context parallelism
        # offset=0: traditional behavior, freqs[0:first_cp_seg] and freqs[...]
        # offset>0: exact mapping, freqs[offset+0:offset+first_cp_seg] and freqs[offset+...]
        return torch.cat(
            [
                freqs[offset + cp_rank * first_cp_seg : offset + (cp_rank + 1) * first_cp_seg],
                freqs[
                    offset
                    + full_seqlen
                    - (cp_rank + 1) * second_cp_seg : offset
                    + full_seqlen
                    - cp_rank * second_cp_seg
                ],
            ]
        )
    else:
        # For single context parallel rank:
        # offset=0: use freqs[0:x.size(0)] (traditional)
        # offset>0: use freqs[offset:offset+x.size(0)] (exact mapping)
        return freqs[offset : offset + x.size(0)]


def _get_thd_raw_mrope_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor, offset: int = 0
) -> Tensor:
    """Get raw mRoPE frequency slices for this CP rank in THD layout."""
    if cp_size > 1:
        first_cp_seg = (x.size(0) + 1) // 2
        second_cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        return torch.cat(
            [
                freqs[
                    :, :, offset + cp_rank * first_cp_seg : offset + (cp_rank + 1) * first_cp_seg
                ],
                freqs[
                    :,
                    :,
                    offset
                    + full_seqlen
                    - (cp_rank + 1) * second_cp_seg : offset
                    + full_seqlen
                    - cp_rank * second_cp_seg,
                ],
            ],
            dim=2,
        )
    else:
        return freqs[:, :, offset : offset + x.size(0)]


def _get_thd_cp_splits(cu_seqlens: Tensor, cp_size: int) -> tuple[list[int], list[int]]:
    """Return global sequence offsets and per-rank sequence lengths for THD CP fallback."""
    cu_seqlens_list = cu_seqlens.tolist()
    local_seqlens = []
    for seq_start, seq_end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:]):
        seq_len = seq_end - seq_start
        if cp_size > 1 and seq_len % cp_size != 0:
            raise ValueError(
                "THD sequence lengths must be divisible by context parallel size, "
                f"got sequence length {seq_len}, cp_size={cp_size}"
            )
        local_seqlens.append(seq_len // cp_size)
    return cu_seqlens_list, local_seqlens


def _pack_thd_raw_mrope_freqs(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    cp_group: torch.distributed.ProcessGroup,
    total_seqlen: Optional[int] = None,
) -> Tensor:
    """Pack raw mRoPE freqs into the same local token order as THD tensor ``t``."""
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    cu_seqlens_list, seqlens = _get_thd_cp_splits(cu_seqlens, cp_size)
    sequence_splits = torch.split(t, seqlens)
    if total_seqlen is None:
        total_seqlen = cu_seqlens_list[-1]
        assert freqs.size(2) == total_seqlen, (
            f"raw mRoPE THD freqs sequence length {freqs.size(2)} must match "
            f"cu_seqlens[-1] = {total_seqlen}"
        )

    freq_slices = []
    for i, x in enumerate(sequence_splits):
        seq_start_offset = cu_seqlens_list[i]
        freq_slices.append(
            _get_thd_raw_mrope_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs, seq_start_offset)
        )

    packed_freqs = torch.cat(freq_slices, dim=2)
    assert packed_freqs.shape[2] == t.shape[0], (
        f"packed raw mRoPE freqs sequence length {packed_freqs.shape[2]} "
        f"does not match THD tensor length {t.shape[0]}"
    )
    return packed_freqs.contiguous()


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
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
        cp_group (torch.distributed.ProcessGroup): The context parallel group

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
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
    cu_seqlens_list, seqlens = _get_thd_cp_splits(cu_seqlens, cp_size)

    # Handle two different frequency tensor formats:
    # 1. If freqs.size(0) == cu_seqlens_list[-1]: freqs contains all positions across all sequences
    #    -> Use offset-based mapping for exact positional correspondence
    # 2. Otherwise: freqs contains only max sequence length positions
    #    -> Use traditional mapping without offsets (map first :seqlen part)
    if freqs.dim() >= 1 and freqs.size(0) == cu_seqlens_list[-1]:
        # CASE 1: Exact mapping with offsets
        # Build packed freqs in one pass, then apply once to the whole packed tensor
        sequence_splits = torch.split(t, seqlens)
        freq_slices = []
        for i, x in enumerate(sequence_splits):
            # cu_seqlens[i] is the starting offset of this sequence in the original batch
            seq_start_offset = cu_seqlens_list[i]
            freq_slices.append(
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs, seq_start_offset)
            )

        freqs_packed = torch.cat(freq_slices, dim=0)

        return _apply_rotary_pos_emb_bshd(
            t.unsqueeze(1),
            freqs_packed,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
        ).squeeze(1)
    else:
        # CASE 2: Traditional mapping without offsets
        # Build packed freqs for all sequences using the standard mapping, then apply once
        sequence_splits = torch.split(t, seqlens)
        freqs_packed = torch.cat(
            [_get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs) for x in sequence_splits],
            dim=0,
        )

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
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    global fused_apply_rotary_pos_emb, fused_apply_rotary_pos_emb_thd

    # Keep for backward compatibility. Will deprecate in the future.
    if cp_group is None:
        cp_group = parallel_state.get_context_parallel_group()

    is_raw_mrope_freqs = (
        _is_raw_mrope_freqs(t, freqs, config)
        if cu_seqlens is None
        else _is_raw_mrope_freqs_thd(t, freqs, cu_seqlens, config, cp_group.size())
    )

    if config.apply_rope_fusion:
        if cu_seqlens is None:
            force_unfused_mrope = False
            if is_raw_mrope_freqs:
                unavailable_reason = None
                can_try_fused_mrope = (
                    fused_apply_mrope is not None
                    and get_fused_mrope_unavailable_reason is not None
                    and not mla_rotary_interleaved
                    and not inverse
                    and mscale == 1.0
                )
                if can_try_fused_mrope:
                    unavailable_reason = get_fused_mrope_unavailable_reason(
                        t, freqs, config.rotary_interleaved
                    )
                use_fused_mrope = can_try_fused_mrope and unavailable_reason is None
                if use_fused_mrope:
                    return fused_apply_mrope(
                        t,
                        freqs,
                        config.mrope_section,
                        interleaved_mrope=config.mrope_interleaved,
                        rotary_interleaved=config.rotary_interleaved,
                    )

                if unavailable_reason is not None:
                    _warn_rope_fusion_fallback_once(
                        _fused_mrope_unavailable_warning_key(unavailable_reason),
                        f"Triton fused mRoPE is unavailable: {unavailable_reason}. "
                        "Using unfused implementation.",
                    )
                    force_unfused_mrope = True
                unavailable_is_rotary_interleaved = (
                    unavailable_reason is not None
                    and "rotary_interleaved" in unavailable_reason.lower()
                )
                if mscale != 1.0:
                    _warn_rope_fusion_fallback_once(
                        "triton-mrope-mscale",
                        f"mscale={mscale} is not supported by Triton fused mRoPE. "
                        "Using unfused implementation.",
                    )
                    force_unfused_mrope = True
                if mla_rotary_interleaved:
                    _warn_rope_fusion_fallback_once(
                        "triton-mrope-mla-rotary-interleaved",
                        "Triton fused mRoPE does not support MLA-style interleaving in RoPE. "
                        "Using unfused implementation.",
                    )
                    force_unfused_mrope = True
                if inverse:
                    _warn_rope_fusion_fallback_once(
                        "triton-mrope-inverse",
                        "inverse RoPE is not supported by Triton fused mRoPE. "
                        "Using unfused implementation.",
                    )
                    force_unfused_mrope = True
                if config.rotary_interleaved and not unavailable_is_rotary_interleaved:
                    _warn_rope_fusion_fallback_once(
                        "triton-mrope-rotary-interleaved",
                        "Triton fused mRoPE currently supports rotary_interleaved=False. "
                        "Using unfused implementation.",
                    )
                    force_unfused_mrope = True
                freqs = _raw_mrope_freqs_to_emb(freqs, config)
                is_raw_mrope_freqs = False
                if force_unfused_mrope:
                    return _apply_rotary_pos_emb_bshd(
                        t,
                        freqs,
                        rotary_interleaved=config.rotary_interleaved,
                        mla_rotary_interleaved=mla_rotary_interleaved,
                        mscale=mscale,
                        inverse=inverse,
                        mla_output_remove_interleaving=mla_output_remove_interleaving,
                    )

            # NOTE: TE backends do not support mRoPE in bshd format when bs > 1.
            use_unfused = False
            if config.mrope_section is not None and freqs.shape[1] > 1:
                # TODO: Add a check in TransformerConfig and remove this unfused implementation.
                _warn_rope_fusion_fallback_once(
                    "te-mrope-bshd-batch",
                    "Transformer Engine fused RoPE does not support mRoPE in bshd format when "
                    "bs > 1 without raw mRoPE freqs. Using unfused implementation.",
                )
                use_unfused = True
            if mscale != 1.0:
                _warn_rope_fusion_fallback_once(
                    "te-rope-mscale",
                    f"mscale={mscale} is not supported by TE's fused RoPE. "
                    "Using unfused implementation.",
                )
                use_unfused = True
            if mla_rotary_interleaved:
                _warn_rope_fusion_fallback_once(
                    "te-rope-mla-rotary-interleaved",
                    "apply_rope_fusion does not support MLA-style interleaving in RoPE. "
                    "Using unfused implementation.",
                )
                use_unfused = True
            if inverse:
                _warn_rope_fusion_fallback_once(
                    "te-rope-inverse",
                    "inverse RoPE is not supported by TE's fused RoPE. "
                    "Using unfused implementation.",
                )
                use_unfused = True
            if fused_apply_rotary_pos_emb is None:
                _warn_rope_fusion_fallback_once(
                    "te-rope-unavailable",
                    "Transformer Engine fused RoPE is unavailable. Using unfused implementation.",
                )
                use_unfused = True
            if not use_unfused:
                return fused_apply_rotary_pos_emb(t, freqs, interleaved=config.rotary_interleaved)
        else:
            if is_raw_mrope_freqs:
                use_fused_mrope_thd = (
                    fused_apply_mrope_thd is not None
                    and can_launch_fused_mrope_thd is not None
                    and get_fused_mrope_thd_unavailable_reason is not None
                    and mscale == 1.0
                    and not mla_rotary_interleaved
                    and not inverse
                    and not config.rotary_interleaved
                )
                if use_fused_mrope_thd:
                    unavailable_reason = get_fused_mrope_thd_unavailable_reason(
                        t,
                        cu_seqlens,
                        freqs,
                        rotary_interleaved=config.rotary_interleaved,
                        cp_size=cp_group.size(),
                        cp_rank=cp_group.rank(),
                    )
                    if unavailable_reason is None:
                        return fused_apply_mrope_thd(
                            t,
                            cu_seqlens,
                            freqs,
                            config.mrope_section,
                            interleaved_mrope=config.mrope_interleaved,
                            rotary_interleaved=config.rotary_interleaved,
                            cp_size=cp_group.size(),
                            cp_rank=cp_group.rank(),
                        )
                    _warn_rope_fusion_fallback_once(
                        _fused_mrope_unavailable_warning_key(unavailable_reason, thd=True),
                        f"Triton fused mRoPE for THD layout is unavailable: "
                        f"{unavailable_reason}. Using unfused implementation.",
                    )
                else:
                    has_unsupported_option = False
                    if mscale != 1.0:
                        _warn_rope_fusion_fallback_once(
                            "triton-mrope-thd-mscale",
                            f"mscale={mscale} is not supported by Triton fused mRoPE for THD "
                            "layout. Using unfused implementation.",
                        )
                        has_unsupported_option = True
                    if mla_rotary_interleaved:
                        _warn_rope_fusion_fallback_once(
                            "triton-mrope-thd-mla-rotary-interleaved",
                            "Triton fused mRoPE for THD layout does not support MLA-style "
                            "interleaving in RoPE. Using unfused implementation.",
                        )
                        has_unsupported_option = True
                    if inverse:
                        _warn_rope_fusion_fallback_once(
                            "triton-mrope-thd-inverse",
                            "inverse RoPE is not supported by Triton fused mRoPE for THD layout. "
                            "Using unfused implementation.",
                        )
                        has_unsupported_option = True
                    if config.rotary_interleaved:
                        _warn_rope_fusion_fallback_once(
                            "triton-mrope-thd-rotary-interleaved",
                            "Triton fused mRoPE for THD layout currently supports "
                            "rotary_interleaved=False. Using unfused implementation.",
                        )
                        has_unsupported_option = True
                    if not has_unsupported_option:
                        _warn_rope_fusion_fallback_once(
                            "triton-mrope-thd-unavailable",
                            "Triton fused mRoPE for THD layout is unavailable. "
                            "Using unfused implementation.",
                        )
                freqs = _raw_mrope_freqs_to_emb(freqs, config)
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
                )
            use_unfused_thd = False
            if mscale != 1.0:
                _warn_rope_fusion_fallback_once(
                    "te-rope-thd-mscale",
                    f"mscale={mscale} is not supported by TE's fused RoPE for THD layout. "
                    "Using unfused implementation.",
                )
                use_unfused_thd = True
            if mla_rotary_interleaved:
                _warn_rope_fusion_fallback_once(
                    "te-rope-thd-mla-rotary-interleaved",
                    "TE fused RoPE for THD layout does not support MLA-style interleaving "
                    "in RoPE. Using unfused implementation.",
                )
                use_unfused_thd = True
            if inverse:
                _warn_rope_fusion_fallback_once(
                    "te-rope-thd-inverse",
                    "inverse RoPE is not supported by TE's fused RoPE for THD layout. "
                    "Using unfused implementation.",
                )
                use_unfused_thd = True
            if fused_apply_rotary_pos_emb_thd is None:
                _warn_rope_fusion_fallback_once(
                    "te-rope-thd-unavailable",
                    "Transformer Engine fused RoPE for THD layout is unavailable. "
                    "Using unfused implementation.",
                )
                use_unfused_thd = True
            if use_unfused_thd:
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
                )
            return fused_apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                cp_size=cp_group.size(),
                cp_rank=cp_group.rank(),
                interleaved=config.rotary_interleaved,
            )
    # use unfused implementation
    if is_raw_mrope_freqs:
        freqs = _raw_mrope_freqs_to_emb(freqs, config)

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
