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
    total_tokens: int = None
    seq_idx: Tensor = None
    tokens_per_sample: int = None
    pad_between_seqs: bool = None
    seq_aux_loss_sample_ids: Tensor = None
    seq_aux_loss_num_samples: int = None

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

    def get_seq_aux_loss_sample_ids(
        self,
        num_local_tokens: int,
        *,
        cp_size: int,
        tp_size: int,
        tp_rank: int,
        sequence_parallel: bool,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        """Return logical sample ownership aligned with the local THD token stream.

        By default, ownership is derived from padded cumulative sequence boundaries.
        Context parallelism partitions each logical sequence independently, so every
        rank owns ``padded_sequence_length / cp_size`` tokens from each sample. Sequence
        parallelism then selects a contiguous TP-rank slice from that CP-local stream.

        Nonstandard token orderings can provide ``seq_aux_loss_sample_ids`` explicitly.
        Explicit IDs must already be aligned with the local router input, and
        ``seq_aux_loss_num_samples`` must include samples that have no tokens on this rank.

        Args:
            num_local_tokens: Number of tokens in the local router input.
            cp_size: Context-parallel world size that partitioned the THD input.
            tp_size: Tensor-parallel world size.
            tp_rank: Rank in the tensor-parallel group.
            sequence_parallel: Whether the CP-local stream is sequence-parallel sharded.
            device: Device holding the local router input.

        Returns:
            Local sample IDs and the total logical sample count.

        Raises:
            ValueError: If ownership metadata is missing or inconsistent with the local input.
        """
        if cp_size < 1 or tp_size < 1:
            raise ValueError("cp_size and tp_size must be positive.")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}.")

        cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        num_samples = self.seq_aux_loss_num_samples
        if num_samples is None and isinstance(cu_seqlens, Tensor):
            num_samples = cu_seqlens.numel() - 1
        if num_samples is not None and num_samples < 1:
            raise ValueError("seq_aux_loss_num_samples must be positive.")

        if self.seq_aux_loss_sample_ids is not None:
            if num_samples is None:
                raise ValueError(
                    "seq_aux_loss_num_samples is required when explicit sample IDs are "
                    "provided without cumulative sequence boundaries."
                )
            sample_ids = self.seq_aux_loss_sample_ids.reshape(-1).to(
                device=device, dtype=torch.long
            )
            if torch.any((sample_ids < 0) | (sample_ids >= num_samples)):
                raise ValueError(
                    "seq_aux_loss_sample_ids must be in "
                    f"[0, seq_aux_loss_num_samples), got {num_samples} samples."
                )
        else:
            if not isinstance(cu_seqlens, Tensor) or num_samples is None:
                raise ValueError(
                    "Variable-length packed seq_aux_loss requires cu_seqlens_q, "
                    "cu_seqlens_q_padded, or explicit seq_aux_loss_sample_ids."
                )
            if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                raise ValueError(
                    "Packed sequence boundaries for seq_aux_loss must be a one-dimensional "
                    "tensor with at least two entries."
                )

            padded_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device=device, dtype=torch.long)
            if torch.any(padded_lengths < 0):
                raise ValueError(
                    "Packed sequence boundaries for seq_aux_loss must be nondecreasing."
                )
            cp_divisor = 2 * cp_size
            if cp_size > 1 and torch.any(padded_lengths % cp_divisor != 0):
                raise ValueError(
                    "Each padded packed sequence length must be divisible by 2 * cp_size "
                    "to derive context-parallel sample ownership."
                )
            local_lengths = padded_lengths // cp_size
            sample_ids = torch.repeat_interleave(
                torch.arange(num_samples, device=device), local_lengths
            )

            if sequence_parallel:
                if sample_ids.numel() % tp_size != 0:
                    raise ValueError(
                        "The CP-local packed token count must be divisible by tp_size when "
                        "sequence parallelism is enabled."
                    )
                local_size = sample_ids.numel() // tp_size
                sample_ids = sample_ids.narrow(0, tp_rank * local_size, local_size).contiguous()

            self.seq_aux_loss_sample_ids = sample_ids
            self.seq_aux_loss_num_samples = num_samples

        if sample_ids.numel() != num_local_tokens:
            raise ValueError(
                "Packed seq_aux_loss sample ownership does not match the local router input: "
                f"got {sample_ids.numel()} sample IDs for {num_local_tokens} tokens. "
                "Provide explicitly aligned seq_aux_loss_sample_ids for nonstandard "
                "token orderings."
            )
        return sample_ids, num_samples
