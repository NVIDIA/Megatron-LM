# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""RoPE helpers for MLite attention primitives."""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor


def get_pos_emb_on_this_cp_rank(
    pos_emb: Tensor, seq_dim: int, cp_group: torch.distributed.ProcessGroup
) -> Tensor:
    if cp_group is None:
        raise ValueError("cp_group must be provided to get positional embedding per CP rank")
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device=pos_emb.device, dtype=torch.long
    )
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    return pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
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
    multi_latent_attention: Optional[bool] = None,
) -> Tensor:
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated. Use mla_rotary_interleaved instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mla_rotary_interleaved = multi_latent_attention

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if mla_rotary_interleaved:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor, offset: int = 0
) -> Tensor:
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
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
    return freqs[offset : offset + x.size(0)]


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
    multi_latent_attention: Optional[bool] = None,
) -> Tensor:
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated. Use mla_rotary_interleaved instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mla_rotary_interleaved = multi_latent_attention

    if cp_group is None:
        raise ValueError("cp_group must be provided for THD format RoPE")
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()

    sequence_splits = torch.split(t, seqlens)
    if freqs.dim() >= 1 and freqs.size(0) == cu_seqlens[-1]:
        freq_slices = []
        for i, x in enumerate(sequence_splits):
            seq_start_offset = cu_seqlens[i].item()
            freq_slices.append(
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs, seq_start_offset)
            )
        freqs_packed = torch.cat(freq_slices, dim=0)
    else:
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
    ).squeeze(1)


__all__ = [
    "_apply_rotary_pos_emb_bshd",
    "_apply_rotary_pos_emb_thd",
    "_rotate_half",
    "get_pos_emb_on_this_cp_rank",
]
