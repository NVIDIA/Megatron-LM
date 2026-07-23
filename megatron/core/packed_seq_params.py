# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass, replace
from typing import Literal, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX = "_packed_seq_params_"

PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
)

PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS = (
    "qkv_format",
    "max_seqlen_q",
    "max_seqlen_kv",
    "local_cp_size",
    "cp_group",
    "pad_between_seqs",
    "cp_partition_mode",
)


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
    cp_partition_mode: Literal["zigzag", "contiguous"] = "zigzag"

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


def _cuda_graph_packed_seq_params_key(field_name: str, prefix: str) -> str:
    return f"{prefix}{field_name}"


def split_packed_seq_params_for_cuda_graph(
    packed_seq_params: PackedSeqParams | None, prefix: str = CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
) -> tuple[dict[str, Tensor | None], dict[str, object]]:
    """Split ``PackedSeqParams`` into graph Tensor inputs and static metadata.

    Transformer Engine CUDA graph inputs must be tensors or ``None``. ``PackedSeqParams`` mixes
    dynamic Tensor fields, such as cumulative sequence lengths, with static metadata, such as THD
    format and max sequence lengths. This helper keeps only the fields TE attention consumes;
    Mamba-only fields such as ``total_tokens`` and ``seq_idx`` stay outside this graph boundary.
    """
    if packed_seq_params is None:
        return {}, {}

    tensor_kwargs = {}
    for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS:
        value = getattr(packed_seq_params, field_name)
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"PackedSeqParams.{field_name} must be a Tensor or None for CUDA graphs, "
                f"got {type(value).__name__}."
            )
        if value is not None:
            tensor_kwargs[_cuda_graph_packed_seq_params_key(field_name, prefix)] = value

    static_metadata = {}
    for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS:
        value = getattr(packed_seq_params, field_name)
        if isinstance(value, Tensor):
            raise TypeError(
                f"PackedSeqParams.{field_name} is static CUDA graph metadata and must not be "
                "a Tensor."
            )
        static_metadata[field_name] = value

    return tensor_kwargs, static_metadata


def has_packed_seq_params_cuda_graph_kwargs(
    kwargs: Mapping[str, object], prefix: str = CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
) -> bool:
    """Return whether ``kwargs`` contains flattened ``PackedSeqParams`` Tensor fields."""
    return any(
        _cuda_graph_packed_seq_params_key(field_name, prefix) in kwargs
        for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS
    )


def build_packed_seq_params_from_cuda_graph_kwargs(
    kwargs: MutableMapping[str, object],
    static_metadata: Mapping[str, object] | None,
    prefix: str = CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    remove_from_kwargs: bool = True,
) -> PackedSeqParams | None:
    """Rebuild ``PackedSeqParams`` from flattened CUDA graph kwargs.

    Args:
        kwargs: Graph kwargs that may contain flattened packed-sequence Tensor fields.
        static_metadata: Non-Tensor metadata produced by
            :func:`split_packed_seq_params_for_cuda_graph`.
        prefix: Prefix used for flattened Tensor fields.
        remove_from_kwargs: Whether to pop consumed flattened fields from ``kwargs``.
    """
    packed_seq_params_kwargs = dict(static_metadata or {})
    found_tensor_field = False
    for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS:
        key = _cuda_graph_packed_seq_params_key(field_name, prefix)
        if key not in kwargs:
            continue
        found_tensor_field = True
        value = kwargs.pop(key) if remove_from_kwargs else kwargs[key]
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"Flattened PackedSeqParams field {key} must be a Tensor or None, "
                f"got {type(value).__name__}."
            )
        packed_seq_params_kwargs[field_name] = value

    if not packed_seq_params_kwargs and not found_tensor_field:
        return None

    return PackedSeqParams(**packed_seq_params_kwargs)


def _pad_seq_tensor(tensor: Optional[Tensor], target_len: int) -> Optional[Tensor]:
    """Pad a token-like tensor along its last dimension with zeros."""
    if tensor is None:
        return None
    actual_len = tensor.shape[-1]
    assert actual_len <= target_len, (
        f"Sequence-length tensor (last dim = {actual_len}) exceeds target ({target_len}); "
        "increase max_seqlen_per_dp_cp_rank or filter overlong samples upstream."
    )
    if actual_len == target_len:
        return tensor
    return F.pad(tensor, (0, target_len - actual_len), value=0)


def _pad_padding_mask(mask: Tensor, target_len: int) -> Tensor:
    """Pad a boolean padding mask with ``True`` along its last dimension."""
    actual_len = mask.shape[-1]
    assert actual_len <= target_len, (
        f"Padding mask length ({actual_len}) exceeds target ({target_len}); "
        "refusing to silently truncate."
    )
    if actual_len == target_len:
        return mask

    pad_shape = list(mask.shape)
    pad_shape[-1] = target_len - actual_len
    tail = torch.ones(pad_shape, dtype=mask.dtype, device=mask.device)
    return torch.cat((mask, tail), dim=-1)


def _pad_cu_seqlens(cu_seqlens: Optional[Tensor], target_entries: int) -> Optional[Tensor]:
    """Pad cumulative-length metadata to a static number of entries."""
    if cu_seqlens is None:
        return None
    actual_entries = cu_seqlens.shape[0]
    assert actual_entries <= target_entries, (
        f"Actual num_seqs ({actual_entries - 1}) exceeds thd_max_packed_sequences "
        f"({target_entries - 1})."
    )
    if actual_entries == target_entries:
        return cu_seqlens
    padded = torch.empty(
        (target_entries,), dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    padded.fill_(cu_seqlens[-1].item())
    padded[:actual_entries] = cu_seqlens
    return padded


def _append_dummy_seq(cu_seqlens: Optional[Tensor], dummy_end: int) -> Optional[Tensor]:
    """Append a cumulative boundary for the post-pack padding tail."""
    if cu_seqlens is None:
        return None
    dummy = torch.full(
        (1,), int(dummy_end), dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    return torch.cat((cu_seqlens, dummy), dim=0)


def _round_up_to_alignment(value: int, alignment: int) -> int:
    assert alignment > 0, f"Packed sequence padding alignment must be > 0, got {alignment}."
    return ((value + alignment - 1) // alignment) * alignment


def get_thd_padding_kwargs(
    pad_packed_seq_alignment: Union[int, Literal["max"]],
    max_seqlen_per_dp_cp_rank: Optional[int],
    thd_max_packed_sequences: Optional[int],
    cuda_graph_static: bool,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Resolve token and cumulative-length padding settings from the config."""
    if cuda_graph_static:
        assert max_seqlen_per_dp_cp_rank is not None
        return None, int(max_seqlen_per_dp_cp_rank), thd_max_packed_sequences

    if pad_packed_seq_alignment == "max":
        assert max_seqlen_per_dp_cp_rank is not None
        return None, int(max_seqlen_per_dp_cp_rank), None

    return int(pad_packed_seq_alignment), None, None


def _resolve_thd_cp_geometry(
    packed_seq_params: PackedSeqParams,
    cp_group: Optional[dist.ProcessGroup] = None,
    cp_size: Optional[int] = None,
    cp_rank: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve THD context-parallel geometry from explicitly threaded state."""
    if cp_group is not None:
        return int(dist.get_world_size(group=cp_group)), int(dist.get_rank(group=cp_group))

    if cp_size is None:
        if packed_seq_params.cp_group is not None:
            return (
                int(dist.get_world_size(group=packed_seq_params.cp_group)),
                int(dist.get_rank(group=packed_seq_params.cp_group)),
            )
        cp_size = packed_seq_params.local_cp_size or 1

    cp_size = int(cp_size)
    if cp_size == 1:
        return 1, 0
    if cp_rank is None:
        raise ValueError(
            "cp_rank or cp_group must be provided when padding THD metadata with cp_size > 1."
        )
    return cp_size, int(cp_rank)


def _resolve_thd_padding_lengths(
    tokens: Optional[Tensor],
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    position_ids: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    target_len: Optional[int],
    alignment: Optional[int],
    cp_group: Optional[dist.ProcessGroup] = None,
    cp_size: Optional[int] = None,
    cp_rank: Optional[int] = None,
    padding_mask: Optional[Tensor] = None,
) -> Tuple[int, int, int, int, torch.device]:
    """Resolve local and global THD padding lengths without changing tensors."""
    cp_size, cp_rank = _resolve_thd_cp_geometry(
        packed_seq_params, cp_group=cp_group, cp_size=cp_size, cp_rank=cp_rank
    )

    local_tensor_len = None
    mask_device = None
    for candidate in (tokens, labels, loss_mask, position_ids, padding_mask):
        if candidate is not None:
            local_tensor_len = int(candidate.shape[-1])
            mask_device = candidate.device
            break

    has_local_tensor = local_tensor_len is not None
    if packed_seq_params.cu_seqlens_q is not None:
        global_actual_len = int(packed_seq_params.cu_seqlens_q[-1].item())
        if mask_device is None:
            mask_device = packed_seq_params.cu_seqlens_q.device
    else:
        assert has_local_tensor, (
            "packed_seq_params.cu_seqlens_q must be available to derive padding_mask "
            "when all token-like tensors are None."
        )
        global_actual_len = local_tensor_len * cp_size

    if has_local_tensor:
        local_actual_len = local_tensor_len
        local_target_len = (
            int(target_len)
            if target_len is not None
            else _round_up_to_alignment(local_actual_len, alignment)
        )
        return (
            local_actual_len,
            global_actual_len,
            local_target_len,
            local_target_len * cp_size,
            mask_device,
        )

    global_target_len = (
        int(target_len) * cp_size
        if target_len is not None
        else _round_up_to_alignment(global_actual_len, alignment)
    )
    if cp_size > 1:
        from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

        partition_cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        local_actual_len = int(
            get_thd_partitioned_indices(
                partition_cu_seqlens, global_actual_len, cp_size, cp_rank
            ).numel()
        )
        local_target_len = int(
            get_thd_partitioned_indices(
                partition_cu_seqlens, global_target_len, cp_size, cp_rank
            ).numel()
        )
    else:
        local_actual_len = global_actual_len
        local_target_len = global_target_len

    return (
        local_actual_len,
        global_actual_len,
        local_target_len,
        global_target_len,
        mask_device,
    )


def pad_sequence_for_thd(
    tokens: Optional[Tensor],
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    position_ids: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    alignment: Optional[int] = None,
    target_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    pad_by_appending_dummy_seq: bool = True,
    padding_mask: Optional[Tensor] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
    cp_size: Optional[int] = None,
    cp_rank: Optional[int] = None,
) -> Tuple[
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    PackedSeqParams,
    Optional[Tensor],
]:
    """Pad packed THD tensors and return a mask for the padding tail."""
    assert (alignment is None) != (
        target_len is None
    ), "Exactly one of alignment or target_len must be provided for THD padding."

    (
        local_actual_len,
        global_actual_len,
        local_target_len,
        global_target_len,
        mask_device,
    ) = _resolve_thd_padding_lengths(
        tokens,
        labels,
        loss_mask,
        position_ids,
        packed_seq_params,
        target_len=target_len,
        alignment=alignment,
        cp_group=cp_group,
        cp_size=cp_size,
        cp_rank=cp_rank,
        padding_mask=padding_mask,
    )

    if packed_seq_params.cu_seqlens_q is not None:
        cu_seqlens = packed_seq_params.cu_seqlens_q
        individual_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_individual = (
            int(individual_lens.max().item()) if individual_lens.numel() > 0 else 0
        )
        assert max_individual <= global_target_len, (
            f"Individual request length ({max_individual}) exceeds the global padded "
            f"capacity ({global_target_len})."
        )

    tokens = _pad_seq_tensor(tokens, local_target_len)
    labels = _pad_seq_tensor(labels, local_target_len)
    loss_mask = _pad_seq_tensor(loss_mask, local_target_len)
    position_ids = _pad_seq_tensor(position_ids, local_target_len)

    cu_seqlens_q = packed_seq_params.cu_seqlens_q
    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
    cu_seqlens_q_padded = packed_seq_params.cu_seqlens_q_padded
    cu_seqlens_kv_padded = packed_seq_params.cu_seqlens_kv_padded

    target_cu_entries = None if max_num_seqs is None else max_num_seqs + 1
    has_dummy_padding_seq = (
        pad_by_appending_dummy_seq and global_target_len > global_actual_len
    )
    dummy_seq_len = (
        global_target_len - global_actual_len if has_dummy_padding_seq else 0
    )
    if has_dummy_padding_seq:
        cu_seqlens_q = _append_dummy_seq(cu_seqlens_q, global_target_len)
        cu_seqlens_kv = _append_dummy_seq(cu_seqlens_kv, global_target_len)
        cu_seqlens_q_padded = _append_dummy_seq(
            cu_seqlens_q_padded, global_target_len
        )
        cu_seqlens_kv_padded = _append_dummy_seq(
            cu_seqlens_kv_padded, global_target_len
        )

    if target_cu_entries is not None:
        cu_seqlens_q = _pad_cu_seqlens(cu_seqlens_q, target_cu_entries)
        cu_seqlens_kv = _pad_cu_seqlens(cu_seqlens_kv, target_cu_entries)
        cu_seqlens_q_padded = _pad_cu_seqlens(
            cu_seqlens_q_padded, target_cu_entries
        )
        cu_seqlens_kv_padded = _pad_cu_seqlens(
            cu_seqlens_kv_padded, target_cu_entries
        )

    padded_params = replace(
        packed_seq_params,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        max_seqlen_q=(
            global_target_len
            if target_cu_entries is not None
            else max(packed_seq_params.max_seqlen_q or 0, dummy_seq_len)
        ),
        max_seqlen_kv=(
            global_target_len
            if target_cu_entries is not None
            else max(packed_seq_params.max_seqlen_kv or 0, dummy_seq_len)
        ),
        total_tokens=local_target_len if target_cu_entries is None else None,
        seq_idx=None,
        pad_between_seqs=(
            False if has_dummy_padding_seq else packed_seq_params.pad_between_seqs
        ),
    )

    tail_padding_mask = (
        torch.arange(local_target_len, device=mask_device).unsqueeze(0)
        >= local_actual_len
    )
    if padding_mask is None:
        padding_mask = tail_padding_mask
    else:
        padding_mask = (
            _pad_padding_mask(padding_mask, local_target_len) | tail_padding_mask
        )

    return tokens, labels, loss_mask, position_ids, padded_params, padding_mask
