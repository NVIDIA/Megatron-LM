# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class PackedSeqParams:
    '''
    parameters to TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None

    def __post_init__(self):
        # Pre-compute seq_idx for Mamba mixer CUDA graph compatibility.
        # Stored as a non-field attribute so dataclasses.fields() won't include it, preventing it
        # from being forwarded to TE attention.
        cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        if isinstance(cu_seqlens, Tensor) and cu_seqlens.numel() > 1:
            # cu_seqlens follows the standard flash attention convention:
            # [0, len1, len1+len2, ..., total] with shape [num_seqs + 1].
            # Compute individual sequence lengths directly from consecutive diffs.
            seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            if (seq_lengths >= 0).all():
                self.seq_idx = (
                    torch.repeat_interleave(
                        torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths
                    )
                    .to(torch.int32)
                    .unsqueeze(0)
                )
            else:
                self.seq_idx = None
        else:
            self.seq_idx = None
