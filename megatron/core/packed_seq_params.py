# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


def pad_cu_seqlens_for_cuda_graph(cu_seqlens: Tensor, target_num_seqs: int) -> Optional[Tensor]:
    """Create PackedSeqParams for a single bin to enable proper attention masking in TE.
    TODO(helenn/RL): unify with `create_packed_seq_params_for_bin` in rl/sequence_packing_utils.py.

    When using Transformer Engine with sequence packing, we need to provide cu_seqlens
    (cumulative sequence lengths) so that TE knows the boundaries between sequences
    within a packed bin. This prevents attention leakage between unrelated sequences.

    Packed-sequence training produces a variable number of documents per microbatch, so `cu_seqlens`
    (length = num_docs + 1) changes shape from step to step. This helper pads the tensor
    to `target_num_seqs + 1` entries by repeating the final cumulative value. The trailing
    repeated entries indicate zero-length segments which are handled by PackedSeqParams.

    Any trailing padding-to-microbatch-size must already be folded into the last segment
    of `cu_seqlens` before calling this helper. This is done in the SFT dataloader
    (`megatron/training/datasets/sft_dataset.py` rewrites the last entry to `pack_length` when there
    is trailing padding).

    Args:
        cu_seqlens: 1-D int32 cumulative sequence-length tensor of shape (K + 1,) where K is the
            number of real documents. `cu_seqlens[-1]` must equal the total token count of the
            corresponding input tensor (see precondition above).
        target_num_seqs: Target document capacity (the value of --cuda-graph-max-packed-seqs).

    Returns:
        A new 1-D tensor of length target_num_seqs + 1 when K <= target_num_seqs;
        None when K > target_num_seqs so the caller can fall back to an eager forward pass.
    """

    assert cu_seqlens.dim() == 1, f"cu_seqlens must be 1-D, got shape {cu_seqlens.shape}"
    current_len = cu_seqlens.shape[0]
    target_len = target_num_seqs + 1
    if current_len > target_len:
        return None
    if current_len == target_len:
        return cu_seqlens

    # Build the padded tensor without a GPU -> CPU sync: copy the real values, then
    # broadcast-assign the final element into the tail.
    out = torch.empty((target_len,), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    out[:current_len] = cu_seqlens
    out[current_len:] = cu_seqlens[-1]
    return out


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
