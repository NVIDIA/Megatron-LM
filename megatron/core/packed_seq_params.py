# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import operator
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

def _normalize_cu_seqlens(
    cu_seqlens: Tensor,
    *,
    allow_dummy_batch_dim: bool,
    require_nondecreasing: bool,
    validate: bool,
) -> Tensor:
    if not isinstance(cu_seqlens, Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")

    if cu_seqlens.dim() == 2:
        if not allow_dummy_batch_dim:
            raise ValueError("cu_seqlens must be rank 1")
        if cu_seqlens.size(0) != 1:
            raise ValueError("cu_seqlens with a dummy batch dimension must have shape [1, N]")
        cu_seqlens = cu_seqlens.squeeze(0)
    elif cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be rank 1")

    if cu_seqlens.numel() == 0:
        raise ValueError("cu_seqlens must contain at least one cumulative length")

    if cu_seqlens.dtype == torch.bool or cu_seqlens.is_floating_point() or cu_seqlens.is_complex():
        raise TypeError("cu_seqlens must use an integer dtype")

    cu_seqlens = cu_seqlens.to(dtype=torch.int32)

    if validate:
        if cu_seqlens[0].item() != 0:
            raise ValueError("cu_seqlens must start at 0")

        if require_nondecreasing and cu_seqlens.numel() > 1:
            if torch.any(cu_seqlens[1:] < cu_seqlens[:-1]).item():
                raise ValueError("cu_seqlens must be nondecreasing")

    return cu_seqlens


def _normalize_int(value: int | Tensor) -> int:
    if isinstance(value, Tensor):
        if value.dim() == 0:
            value = value.item()
        elif value.dim() == 1 and value.numel() == 1:
            value = value[0].item()
        else:
            raise ValueError("expected a Python int, a scalar tensor, or a tensor with shape [1]")

    if isinstance(value, bool):
        raise TypeError("expected an integer")

    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError("expected an integer or scalar integer tensor") from exc

    if value < 0:
        raise ValueError("expected a non-negative integer")

    return value


def _validate_padded_cu_seqlens_compatibility(cu_seqlens: Tensor, cu_seqlens_padded: Tensor) -> None:
    if cu_seqlens_padded.shape != cu_seqlens.shape:
        raise ValueError("padded cu_seqlens must have the same shape as the unpadded cu_seqlens")

    if cu_seqlens_padded.device != cu_seqlens.device:
        raise ValueError("padded cu_seqlens must be on the same device as the unpadded cu_seqlens")


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

    @classmethod
    def from_cu_seqlens(
        cls,
        *,
        cu_seqlens_q: Tensor,
        max_seqlen_q: int | Tensor,
        cu_seqlens_kv: Tensor | None = None,
        max_seqlen_kv: int | Tensor | None = None,
        cu_seqlens_q_padded: Tensor | None = None,
        cu_seqlens_kv_padded: Tensor | None = None,
        total_tokens: int | Tensor | None = None,
        qkv_format: str = "thd",
        local_cp_size: int | None = None,
        cp_group: dist.ProcessGroup | None = None,
        allow_dummy_batch_dim: bool = True,
        validate: bool = True,
    ) -> "PackedSeqParams":
        """Construct PackedSeqParams from THD cumulative-length metadata.

        Set validate=False when the cumulative-length metadata is already trusted and
        on-device validation overhead should be avoided.
        """
        if qkv_format != "thd":
            raise ValueError(
                f"PackedSeqParams.from_cu_seqlens only supports qkv_format='thd', got {qkv_format!r}"
            )

        cu_seqlens_q = _normalize_cu_seqlens(
            cu_seqlens_q,
            allow_dummy_batch_dim=allow_dummy_batch_dim,
            require_nondecreasing=True,
            validate=validate,
        )
        kv_defaults_from_q = cu_seqlens_kv is None
        if kv_defaults_from_q:
            cu_seqlens_kv = cu_seqlens_q
        else:
            cu_seqlens_kv = _normalize_cu_seqlens(
                cu_seqlens_kv,
                allow_dummy_batch_dim=allow_dummy_batch_dim,
                require_nondecreasing=True,
                validate=validate,
            )

        max_seqlen_q = _normalize_int(max_seqlen_q)
        if max_seqlen_kv is None:
            max_seqlen_kv = max_seqlen_q
        else:
            max_seqlen_kv = _normalize_int(max_seqlen_kv)

        if cu_seqlens_q_padded is not None:
            cu_seqlens_q_padded = _normalize_cu_seqlens(
                cu_seqlens_q_padded,
                allow_dummy_batch_dim=allow_dummy_batch_dim,
                require_nondecreasing=False,
                validate=validate,
            )
            _validate_padded_cu_seqlens_compatibility(cu_seqlens_q, cu_seqlens_q_padded)

        if cu_seqlens_kv_padded is None and kv_defaults_from_q:
            cu_seqlens_kv_padded = cu_seqlens_q_padded
        elif cu_seqlens_kv_padded is not None:
            cu_seqlens_kv_padded = _normalize_cu_seqlens(
                cu_seqlens_kv_padded,
                allow_dummy_batch_dim=allow_dummy_batch_dim,
                require_nondecreasing=False,
                validate=validate,
            )
            _validate_padded_cu_seqlens_compatibility(cu_seqlens_kv, cu_seqlens_kv_padded)

        if total_tokens is None:
            total_tokens = int(cu_seqlens_q[-1].item())
        else:
            total_tokens = _normalize_int(total_tokens)
            if total_tokens < int(cu_seqlens_q[-1].item()):
                raise ValueError("total_tokens must be >= cu_seqlens_q[-1]")

        return cls(
            qkv_format=qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            local_cp_size=local_cp_size,
            cp_group=cp_group,
            total_tokens=total_tokens,
        )

    @classmethod
    def single_sequence(
        cls,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.int32,
        qkv_format: str = "thd",
    ) -> "PackedSeqParams":
        """Construct THD metadata for a single unpacked sequence."""
        seq_len = _normalize_int(seq_len)
        cu_seqlens = torch.tensor([0, seq_len], dtype=dtype, device=device)
        return cls.from_cu_seqlens(
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=seq_len,
            total_tokens=seq_len,
            qkv_format=qkv_format,
            validate=False,
        )

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
