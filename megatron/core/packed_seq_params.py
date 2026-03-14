# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


# Maximum number of packed sequences supported by CUDA graph capture.
# cu_seqlens tensors are padded to this length + 1 for fixed-shape graph inputs.
# Override at runtime with --cuda-graph-max-packed-seqs.
CUDA_GRAPH_MAX_PACKED_SEQS: int = 2048


# Module-level cache for shared CUDA graph buffer tensors.
# Key: (tag, seq_length, max_seqs, device_id) -> shared buffer dict (or tensor for seq_idx).
# All layers with the same key share the SAME dict and SAME underlying tensor objects.
# Updating these tensors once per micro-batch propagates to ALL layers' CUDA graphs.
_CG_SHARED_BUFFERS: dict = {}


@dataclass
class PackedSeqParams:
    '''
    Parameters for TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format.
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    # Tensor versions of max_seqlen for CUDA graph buffer updates (avoids int->tensor inside CG).
    max_seqlen_q_tensor: Tensor = None
    max_seqlen_kv_tensor: Tensor = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    total_tokens: int = None
    # Pre-computed seq_idx for Mamba. When set, mamba_mixer reads it directly,
    # avoiding dynamic allocations that are forbidden inside CUDA graph capture.
    seq_idx: Optional[Tensor] = None

    def __post_init__(self):
        """Pre-compute seq_idx for Mamba mixer.

        Converts cu_seqlens into a per-token sequence index tensor. For example,
        cu_seqlens=[0, 5, 7, 11] with total_tokens=16 produces:
        [0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]

        An extra sequence index is appended for tokens beyond the last cu_seqlens entry.
        """
        if self.seq_idx is not None:
            return  # Already set (e.g. CG dummy PSP with pre-allocated buffer)

        cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        if isinstance(cu_seqlens, Tensor) and self.total_tokens is not None:
            total_tokens_tensor = torch.tensor(
                [self.total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
            )
            cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
            seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
            # Pass output_size to avoid a GPU->CPU sync that repeat_interleave
            # performs when the output length is unknown.
            self.seq_idx = (
                torch.repeat_interleave(
                    torch.arange(seq_lengths.numel(), device=cu_seqlens.device),
                    seq_lengths,
                    output_size=self.total_tokens,
                )
                .to(torch.int32)
                .unsqueeze(0)  # Add a batch dimension
            )

    # ----------------------------------------------------------------
    # CUDA graph padding utilities
    # ----------------------------------------------------------------

    @staticmethod
    def pad_cu_seqlens(cu_seqlens: Tensor, target_len: int) -> Tensor:
        """Pad cu_seqlens to a fixed length using the last element as fill value.

        CUDA graphs require fixed-shape inputs. By padding cu_seqlens to a
        constant size (bucket_size + 1), the graph captures a single shape and
        replays it for all batches that fit within the bucket.
        """
        actual_len = cu_seqlens.shape[0]
        if actual_len >= target_len:
            return cu_seqlens[:target_len]
        padded = cu_seqlens.new_empty(target_len)
        padded[:actual_len] = cu_seqlens
        padded[actual_len:] = cu_seqlens[-1]
        return padded

    def ensure_cg_padded(self, target_len: int) -> None:
        """Lazily compute and cache padded cu_seqlens for CUDA graph replay.

        Called per-layer during CG replay but computes padding only once per
        micro-batch (the PSP object is reused across all layers in the same
        iteration). Subsequent calls are a no-op because the cache is stored
        on the PSP instance itself.
        """
        if getattr(self, '_cg_pad_target', None) == target_len:
            return  # Already cached for this target_len
        self._cg_pad_target = target_len
        self._cg_padded_q = PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_q, target_len)
        self._cg_padded_kv = PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_kv, target_len)
        self._cg_padded_qp = (
            PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_q_padded, target_len)
            if self.cu_seqlens_q_padded is not None
            else None
        )
        self._cg_padded_kvp = (
            PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_kv_padded, target_len)
            if self.cu_seqlens_kv_padded is not None
            else None
        )

    # ----------------------------------------------------------------
    # Shared CUDA graph buffer management
    # ----------------------------------------------------------------

    @classmethod
    def get_or_create_shared_cg_buffers(
        cls,
        seq_length: int,
        max_seqs: int,
        device: torch.device,
        *,
        tag: str = 'attn',
    ) -> Dict[str, Tensor]:
        """Return the shared PSP buffer dict for CUDA graph replay.

        All layers with the same (tag, seq_length, max_seqs, device) share the
        SAME dict object and therefore the SAME underlying tensor objects.
        Updating the tensors once per micro-batch (via copy_()) propagates to
        all layers' captured CUDA graphs simultaneously.
        """
        key = (tag, seq_length, max_seqs, int(device.index or 0))
        if key not in _CG_SHARED_BUFFERS:
            _, buffers = cls.create_dummy_for_cuda_graph(seq_length, max_seqs=max_seqs)
            # Object-identity gate; None forces first update.
            buffers['_last_updated_psp'] = None
            _CG_SHARED_BUFFERS[key] = buffers
        return _CG_SHARED_BUFFERS[key]

    @classmethod
    def get_or_create_shared_seq_idx_buffer(
        cls, total_tokens: int, device: torch.device
    ) -> Tensor:
        """Return the shared seq_idx buffer tensor for Mamba CUDA graph replay."""
        key = ('seq_idx', total_tokens, int(device.index or 0))
        if key not in _CG_SHARED_BUFFERS:
            _CG_SHARED_BUFFERS[key] = torch.zeros(
                1, total_tokens, dtype=torch.int32, device=device
            )
        return _CG_SHARED_BUFFERS[key]

    @classmethod
    def create_dummy_for_cuda_graph(
        cls, seq_length: int, max_seqs: int = CUDA_GRAPH_MAX_PACKED_SEQS
    ) -> Tuple[PackedSeqParams, Dict[str, Tensor]]:
        """Create a dummy PackedSeqParams for CUDA graph capture.

        Returns the dummy PSP and a dict of tensor buffer references that can
        be updated via copy_() during graph replay.
        """
        cu_seqlens_len = max_seqs + 1
        device = torch.cuda.current_device()
        dtype = torch.int32

        cu_seqlens_q = torch.zeros(cu_seqlens_len, dtype=dtype, device=device)
        cu_seqlens_q[1:] = seq_length
        cu_seqlens_kv = cu_seqlens_q.clone()
        cu_seqlens_q_padded = cu_seqlens_q.clone()
        cu_seqlens_kv_padded = cu_seqlens_q.clone()
        max_seqlen_q_tensor = torch.tensor([seq_length], dtype=dtype, device=device)
        max_seqlen_kv_tensor = torch.tensor([seq_length], dtype=dtype, device=device)

        psp = cls(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=seq_length,
            max_seqlen_kv=seq_length,
            max_seqlen_q_tensor=max_seqlen_q_tensor,
            max_seqlen_kv_tensor=max_seqlen_kv_tensor,
        )
        buffers = {
            'cu_seqlens_q': cu_seqlens_q,
            'cu_seqlens_kv': cu_seqlens_kv,
            'cu_seqlens_q_padded': cu_seqlens_q_padded,
            'cu_seqlens_kv_padded': cu_seqlens_kv_padded,
            'max_seqlen_q_tensor': max_seqlen_q_tensor,
            'max_seqlen_kv_tensor': max_seqlen_kv_tensor,
        }
        return psp, buffers
