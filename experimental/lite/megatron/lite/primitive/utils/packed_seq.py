# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed sequence parameter containers for THD-format primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class PackedSeqParams:
    """Parameters consumed by TE DotProductAttention and MLite THD helpers."""

    qkv_format: str | None = "thd"
    cu_seqlens_q: Tensor | None = None
    cu_seqlens_kv: Tensor | None = None
    cu_seqlens_q_padded: Tensor | None = None
    cu_seqlens_kv_padded: Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_kv: int | None = None
    local_cp_size: int | None = None
    cp_group: Any | None = None
    total_tokens: int | None = None
    seq_idx: Tensor | None = None
    cp_rank: int | None = None

    def __post_init__(self) -> None:
        cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        if isinstance(cu_seqlens, Tensor) and self.total_tokens is not None:
            total_tokens_tensor = torch.tensor(
                [self.total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
            )
            cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
            seq_lengths = (cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]).clamp(min=0)
            self.seq_idx = (
                torch.repeat_interleave(
                    torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths
                )
                .to(torch.int32)
                .unsqueeze(0)
            )

    @staticmethod
    def from_cu_seqlens(cu_seqlens: Tensor, max_seqlen: int) -> PackedSeqParams:
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
        )


__all__ = ["PackedSeqParams"]
