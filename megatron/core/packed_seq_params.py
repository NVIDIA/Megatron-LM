# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
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
    total_tokens: int = None
    seq_idx: Tensor = None

    def __post_init__(self):
        """Pre-compute seq_idx for Mamba mixer CUDA graph compatibility.

        If total_tokens is 16 (for example), this method takes packed_seq_params.cu_seqlens_q_padded
        (or cu_seqlens_q) which is of the form [0, 5, 7, 11] and returns a tensor of the form
        [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        which is [0]*(5-0) + [1]*(7-5) + [2]*(11-7) + [3]*(16-11)
        In the above example, there are three sequences in the pack.
        In general, the output has an additional sequence index (e.g. 0, 1, 2, 3) so that any tokens
        beyond the last padded input sequence are accounted for as an extra sequence. However, If
        cu_seqlens_q_padded[-1] == max_seqlen then this additional sequence index will not be
        included.
        """
        cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        if isinstance(cu_seqlens, Tensor) and self.total_tokens is not None:
            total_tokens_tensor = torch.tensor(
                [self.total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
            )
            # Example: [0, 5, 7, 11] -> [0, 5, 7, 11, 16]
            cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
            # Example: [0, 5, 7, 11, 16] -> [5, 2, 4, 5]
            seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
            # Clamp to non-negative: cu_seqlens_q_padded may not be strictly
            # monotonic when context parallelism slices sequences across ranks,
            # or when padded cumulative lengths exceed total_tokens (e.g. the
            # appended total_tokens sentinel is smaller than cu_seqlens[-1]
            # due to padding). In either case the diff can go negative, which
            # causes torch.repeat_interleave to fail.
            seq_lengths = seq_lengths.clamp(min=0)
            # Example: [5, 2, 4, 5] -> [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
            self.seq_idx = (
                torch.repeat_interleave(
                    torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths
                )
                .to(torch.int32)
                .unsqueeze(0)  # Add a batch dimension
            )


def pad_thd_for_cuda_graph(
    tokens: Optional[Tensor],
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    position_ids: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    max_seqlen: int,
    max_num_seqs: int,
) -> Tuple[
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    PackedSeqParams,
    Optional[Tensor],
]:
    """Pad THD batch data to fixed sizes for CUDA Graph compatibility.

    CUDA Graph requires static tensor shapes. This function pads:
    - tokens, labels, loss_mask, position_ids along dim=-1 to max_seqlen
    - cu_seqlens tensors to (max_num_seqs + 1) entries, filled with actual_T
    - Generates padding_mask for MoE aux loss exclusion

    Returns:
        Padded (tokens, labels, loss_mask, position_ids, packed_seq_params, padding_mask)
        padding_mask: [1, max_seqlen] bool tensor, True at padding positions.
    """

    def _pad_seq_tensor(t, target_len):
        if t is None:
            return None
        actual_len = t.shape[-1]
        if actual_len >= target_len:
            return t
        return F.pad(t, (0, target_len - actual_len), value=0)

    def _pad_cu_seqlens(cu_seqlens, target_entries):
        if cu_seqlens is None:
            return None
        actual_entries = cu_seqlens.shape[0]
        if actual_entries >= target_entries:
            return cu_seqlens
        pad_value = cu_seqlens[-1].item()
        padded = torch.full(
            (target_entries,), pad_value, dtype=cu_seqlens.dtype, device=cu_seqlens.device
        )
        padded[:actual_entries] = cu_seqlens
        return padded

    actual_T = None
    mask_device = None
    for candidate in (tokens, labels, loss_mask, position_ids):
        if candidate is not None:
            actual_T = candidate.shape[-1]
            mask_device = candidate.device
            break
    actual_T_is_local = actual_T is not None
    if actual_T is None:
        assert packed_seq_params.cu_seqlens_q is not None, (
            "packed_seq_params.cu_seqlens_q must be available to derive padding_mask "
            "when tokens/labels/loss_mask/position_ids are all None."
        )
        actual_T = int(packed_seq_params.cu_seqlens_q[-1].item())
        mask_device = packed_seq_params.cu_seqlens_q.device

    if actual_T is not None and packed_seq_params.cu_seqlens_q is not None:
        _cu = packed_seq_params.cu_seqlens_q
        _individual_lens = _cu[1:] - _cu[:-1]
        _max_individual = int(_individual_lens.max().item()) if _individual_lens.numel() > 0 else 0
        from megatron.core import parallel_state

        _cp_size = (
            packed_seq_params.local_cp_size
            if packed_seq_params.local_cp_size is not None
            else parallel_state.get_context_parallel_world_size()
        )
        _global_max_seqlen = max_seqlen * _cp_size
        assert _max_individual <= _global_max_seqlen, (
            f"Individual request length ({_max_individual}) exceeds the global max sequence length "
            f"({_global_max_seqlen} = max_seqlen_per_dp_cp_rank {max_seqlen} * cp_size {_cp_size}). "
            f"Each request must fit within the CUDA Graph static buffer after CP partitioning. "
            f"Increase --max-seqlen-per-dp-cp-rank or --seq-length, or filter out overlong requests."
        )

    tokens = _pad_seq_tensor(tokens, max_seqlen)
    labels = _pad_seq_tensor(labels, max_seqlen)
    loss_mask = _pad_seq_tensor(loss_mask, max_seqlen)
    position_ids = _pad_seq_tensor(position_ids, max_seqlen)

    target_cu_entries = max_num_seqs + 1
    padded_params = PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=_pad_cu_seqlens(packed_seq_params.cu_seqlens_q, target_cu_entries),
        cu_seqlens_kv=_pad_cu_seqlens(packed_seq_params.cu_seqlens_kv, target_cu_entries),
        cu_seqlens_q_padded=_pad_cu_seqlens(
            packed_seq_params.cu_seqlens_q_padded, target_cu_entries
        ),
        cu_seqlens_kv_padded=_pad_cu_seqlens(
            packed_seq_params.cu_seqlens_kv_padded, target_cu_entries
        ),
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
    )

    from megatron.core import parallel_state

    cp_size = (
        packed_seq_params.local_cp_size
        if packed_seq_params.local_cp_size is not None
        else parallel_state.get_context_parallel_world_size()
    )
    cp_rank = parallel_state.get_context_parallel_rank() if cp_size > 1 else 0

    if cp_size > 1:
        from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

        if actual_T_is_local:
            local_actual_T = int(actual_T)
            local_max_seqlen = int(max_seqlen)
        else:
            local_actual_T = int(
                get_thd_partitioned_indices(
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None
                    else packed_seq_params.cu_seqlens_q,
                    int(actual_T),
                    cp_size,
                    cp_rank,
                ).numel()
            )
            local_max_seqlen = int(
                get_thd_partitioned_indices(
                    padded_params.cu_seqlens_q_padded
                    if padded_params.cu_seqlens_q_padded is not None
                    else padded_params.cu_seqlens_q,
                    max_seqlen,
                    cp_size,
                    cp_rank,
                ).numel()
            )
        padding_mask = (
            torch.arange(local_max_seqlen, device=mask_device).unsqueeze(0) >= local_actual_T
        )
    else:
        padding_mask = torch.arange(max_seqlen, device=mask_device).unsqueeze(0) >= actual_T

    return tokens, labels, loss_mask, position_ids, padded_params, padding_mask
