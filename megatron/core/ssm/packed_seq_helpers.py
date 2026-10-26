# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for SSM mixers handling packed (THD-format) sequences.

Lifted from `MambaMixer._create_packed_seq_idx` so GDP, KDA, DPv2, GDN can
share a single reference implementation (avoids drift across mixers).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import is_causal_conv1d_min_version


def get_cu_seqlens(packed_seq_params: PackedSeqParams) -> torch.Tensor:
    """Pick the right cu_seqlens tensor (padded if available)."""
    if packed_seq_params.cu_seqlens_q_padded is not None:
        return packed_seq_params.cu_seqlens_q_padded
    return packed_seq_params.cu_seqlens_q


def build_packed_seq_idx(packed_seq_params: PackedSeqParams, total_tokens: int) -> torch.Tensor:
    """Build the per-token sequence index tensor used by varlen kernels.

    For ``packed_seq_params.cu_seqlens_q[_padded]`` of the form
    ``[0, 5, 7, 11]`` and ``total_tokens=16`` returns
    ``[0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]``  (shape ``[1, total_tokens]``,
    int32). The trailing chunk after ``cu_seqlens[-1]`` is treated as one
    extra sequence so the output covers every token in the pack. If
    ``cu_seqlens[-1] == total_tokens`` no extra index is added.

    This is the per-token tensor consumed by ``causal_conv1d_fn(seq_idx=...)``
    and by Mamba's fused conv+SSM kernel as ``seq_idx``.

    ``total_tokens`` must equal the *post-parallelism-gather* sequence length
    that the kernel will actually consume — not the caller's
    ``hidden_states.shape[0]`` which may be sequence-parallel-sharded and/or
    context-parallel-sliced. The robust pattern (mirrors ``mamba_mixer.py``)
    is to call this *after* ``in_proj`` (SP all-gather) and ``pre_conv_ssm``
    (CP all-to-all), passing the post-gather tensor's seq dim — that way
    the helper is agnostic to TP/SP/CP shapes upstream.
    """
    cu_seqlens = get_cu_seqlens(packed_seq_params)
    total_tokens_tensor = torch.tensor(
        [total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
    seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
    seq_idx = torch.repeat_interleave(
        torch.arange(seq_lengths.numel(), device=cu_seqlens.device),
        seq_lengths,
        output_size=total_tokens,
    )
    return seq_idx.to(torch.int32).unsqueeze(0)


def check_fla_sequence_packing_support() -> Tuple[bool, Optional[str]]:
    """Lighter sibling of `_check_mamba_sequence_packing_support` for FLA-backed mixers.

    GDP/KDA/DPv2/GDN reach into FLA's chunk_kda / chunk_gated_delta_product /
    chunk_gated_delta_rule, all of which manage their own variable-length
    state internally. The only shared external dependency is the causal
    conv1d kernel — `causal_conv1d_fn(seq_idx=...)` was added in 1.4.0 and
    is required to reset the conv state at packed-document boundaries.

    Mamba2's stricter `mamba_ssm` minimums (used by `mamba_split_conv1d_scan_combined`)
    do not apply.
    """
    conv1d_min = "1.4.0"
    if not is_causal_conv1d_min_version(conv1d_min):
        return False, f"causal_conv1d >= {conv1d_min} is required for packed sequences"
    return True, None
