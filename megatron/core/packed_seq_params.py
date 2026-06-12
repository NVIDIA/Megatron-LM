# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

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
    pad_between_seqs: Optional[bool] = None

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


def resolve_cp_group(
    static_cp_group: dist.ProcessGroup, packed_seq_params: PackedSeqParams = None
) -> dist.ProcessGroup:
    """Return the dynamic CP group from packed_seq_params when available, else the static one.

    Dynamic CP assigns a per-microbatch CP group that may differ from the
    process-group stored at model construction time.  This helper centralises
    the resolution logic used by GPTModel, GatedDeltaNet, and MTP layers.
    """
    if packed_seq_params is not None and packed_seq_params.cp_group is not None:
        return packed_seq_params.cp_group
    return static_cp_group


def _pad_seq_tensor(t: Optional[Tensor], target_len: int) -> Optional[Tensor]:
    """Pad a [..., seq] tensor to ``target_len`` along the last dim with zeros.

    Asserts the actual length does not exceed ``target_len``: an oversize input
    would silently desync the captured graph from replay shapes.
    """
    if t is None:
        return None
    actual_len = t.shape[-1]
    assert actual_len <= target_len, (
        f"Sequence-length tensor (last dim = {actual_len}) exceeds target "
        f"({target_len}); refusing to silently truncate. Increase "
        f"--max-seqlen-per-dp-cp-rank or filter overlong samples upstream."
    )
    if actual_len == target_len:
        return t
    return F.pad(t, (0, target_len - actual_len), value=0)


def _pad_cu_seqlens(cu_seqlens: Optional[Tensor], target_entries: int) -> Optional[Tensor]:
    """Pad a cu_seqlens tensor to exactly ``target_entries`` entries.

    Asserts the actual entry count does not exceed ``target_entries``. An
    oversized pack cannot be represented by the configured static cu_seqlens
    buffer and would not match captured CUDA Graph replay shapes.
    """
    if cu_seqlens is None:
        return None
    actual_entries = cu_seqlens.shape[0]
    assert actual_entries <= target_entries, (
        f"Actual num_seqs ({actual_entries - 1}) exceeds thd_max_num_seqs "
        f"({target_entries - 1}). Increase --thd-max-num-seqs, decrease "
        f"--max-seqlen-per-dp-cp-rank, or filter shorter samples upstream so "
        f"the packing scheduler stops earlier."
    )
    if actual_entries == target_entries:
        return cu_seqlens
    pad_value = cu_seqlens[-1].item()
    padded = torch.full(
        (target_entries,), pad_value, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    padded[:actual_entries] = cu_seqlens
    return padded


def _append_dummy_seq(cu_seqlens: Optional[Tensor], dummy_end: int) -> Optional[Tensor]:
    """Append a dummy sequence boundary to a cu_seqlens tensor.

    ``dummy_end`` is the padded target length. Appending it to both
    ``cu_seqlens_*`` and ``cu_seqlens_*_padded`` represents the post-pack
    alignment tail as an ordinary dummy sequence. That keeps every token row
    covered by THD metadata without enabling TE's pad-between-sequences mode.
    """
    if cu_seqlens is None:
        return None

    dummy = torch.full((1,), int(dummy_end), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.cat((cu_seqlens, dummy), dim=0)


def _round_up_to_alignment(value: int, alignment: int) -> int:
    assert alignment > 0, f"Packed sequence padding alignment must be > 0, got {alignment}."
    return ((value + alignment - 1) // alignment) * alignment


def get_thd_padding_kwargs(
    pad_packed_seq_alignment: Union[int, Literal["max"]],
    max_seqlen_per_dp_cp_rank: Optional[int],
    thd_max_num_seqs: Optional[int],
    cuda_graph_static: bool,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Resolve ``pad_sequence_for_thd`` kwargs from the training config.

    ``--pad-packed-seq-alignment`` has two forms:

    - ``max`` pads token-like tensors to ``max_seqlen_per_dp_cp_rank``;
    - a positive value pads token-like tensors to a multiple of that value.

    Padding cu_seqlens to ``thd_max_num_seqs + 1`` is a CUDA Graph static-input
    requirement. Eager pad-to-max should preserve sequence metadata so kernels
    continue to see the real packed sequence boundaries.
    """
    if cuda_graph_static:
        return None, int(max_seqlen_per_dp_cp_rank), thd_max_num_seqs

    if pad_packed_seq_alignment == "max":
        return None, int(max_seqlen_per_dp_cp_rank), None

    return int(pad_packed_seq_alignment), None, None


def _resolve_thd_padding_lengths(
    tokens: Optional[Tensor],
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    position_ids: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    target_len: Optional[int],
    alignment: Optional[int],
) -> Tuple[int, int, int, int, torch.device]:
    """Resolve local/global THD padding lengths without changing tensors.

    Returns:
        local_actual_T: Current rank's token-like tensor length.
        global_actual_T: Global packed length represented by THD metadata.
        local_target_len: Current rank's padded token-like tensor length.
        global_target_len: Global padded endpoint represented by THD metadata.
        mask_device: Device used to build the returned padding mask.
    """

    # Resolve the CP geometry used to translate local lengths to global endpoints.
    from megatron.core import parallel_state

    cp_size = (
        packed_seq_params.local_cp_size
        if packed_seq_params.local_cp_size is not None
        else parallel_state.get_context_parallel_world_size()
    )
    cp_rank = parallel_state.get_context_parallel_rank() if cp_size > 1 else 0

    # Find the first token-like tensor that carries this rank's local length.
    local_tensor_T = None
    mask_device = None
    for candidate in (tokens, labels, loss_mask, position_ids):
        if candidate is not None:
            local_tensor_T = int(candidate.shape[-1])
            mask_device = candidate.device
            break

    # Prefer THD metadata for the global packed length when it is available.
    has_local_tensor = local_tensor_T is not None
    if packed_seq_params.cu_seqlens_q is not None:
        global_actual_T = int(packed_seq_params.cu_seqlens_q[-1].item())
        if mask_device is None:
            mask_device = packed_seq_params.cu_seqlens_q.device
    else:
        assert has_local_tensor, (
            "packed_seq_params.cu_seqlens_q must be available to derive padding_mask "
            "when tokens/labels/loss_mask/position_ids are all None."
        )
        global_actual_T = local_tensor_T * cp_size

    # Tensor path: use the already-sliced local shape and scale to the global endpoint.
    if has_local_tensor:
        local_actual_T = local_tensor_T
        local_target_len = (
            int(target_len)
            if target_len is not None
            else _round_up_to_alignment(local_actual_T, alignment)
        )
        global_target_len = local_target_len * cp_size
        return local_actual_T, global_actual_T, local_target_len, global_target_len, mask_device

    # Metadata-only path: resolve the global padded endpoint first.
    global_target_len = (
        int(target_len) * cp_size
        if target_len is not None
        else _round_up_to_alignment(global_actual_T, alignment)
    )

    # Under CP, ask TE which packed rows this rank would receive.
    if cp_size > 1:
        from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

        partition_cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        # The number of selected rows is this rank's local actual length.
        local_actual_T = int(
            get_thd_partitioned_indices(
                partition_cu_seqlens, global_actual_T, cp_size, cp_rank
            ).numel()
        )
        # Do the same for the padded endpoint; THD CP is not simple equal split.
        local_target_len = int(
            get_thd_partitioned_indices(
                partition_cu_seqlens, global_target_len, cp_size, cp_rank
            ).numel()
        )
    else:
        # Without CP, local and global metadata lengths are identical.
        local_actual_T = global_actual_T
        local_target_len = global_target_len

    return local_actual_T, global_actual_T, local_target_len, global_target_len, mask_device


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
) -> Tuple[
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    PackedSeqParams,
    Optional[Tensor],
]:
    """Pad packed THD tensors after packing.

    This appends padding tokens to token-like tensors and returns a padding mask
    for MoE auxiliary-loss/routing paths.

    Args:
        tokens: Packed token tensor with sequence length on the last dimension,
            or None on pipeline stages that do not own tokens.
        labels: Packed label tensor with sequence length on the last dimension,
            or None.
        loss_mask: Packed loss mask tensor with sequence length on the last
            dimension, or None.
        position_ids: Packed position id tensor with sequence length on the
            last dimension, or None.
        packed_seq_params: THD metadata for the packed batch.
        alignment: If set, round each CP-local token-like tensor length up to
            this multiple. Exactly one of ``alignment`` and ``target_len`` must
            be provided.
        target_len: If set, pad token-like tensors to this CP-local length.
            Exactly one of ``alignment`` and ``target_len`` must be provided.
        max_num_seqs: If set, pad cu_seqlens tensors to
            ``max_num_seqs + 1`` entries for static CUDA Graph inputs.
        pad_by_appending_dummy_seq: If true, represent the post-pack padding
            tail as an extra dummy sequence in cu_seqlens metadata.

    Notes:
        - THD CP slicing is defined by Transformer Engine. On metadata-only
          stages, Megatron asks TE which packed rows this CP rank would receive
          and uses that row count as the local length instead of assuming equal
          division by CP size.
        - When ``pad_by_appending_dummy_seq`` is true, the padding tail is also
          represented as an ordinary dummy sequence in cu_seqlens metadata.
        - ``max_num_seqs`` pads all four cu_seqlens tensors; this is required
          by CUDA Graph replay because those tensors are graph inputs.

    Returns:
        Padded (tokens, labels, loss_mask, position_ids, packed_seq_params, padding_mask)
        padding_mask: [1, target] bool tensor, True at padding positions.
    """
    assert (alignment is None) != (
        target_len is None
    ), "Exactly one of alignment or target_len must be provided for THD padding."

    local_actual_T, global_actual_T, local_target_len, global_target_len, mask_device = (
        _resolve_thd_padding_lengths(
            tokens,
            labels,
            loss_mask,
            position_ids,
            packed_seq_params,
            target_len=target_len,
            alignment=alignment,
        )
    )

    # Reject individual packed sequences that cannot fit the resolved target.
    if packed_seq_params.cu_seqlens_q is not None:
        _cu = packed_seq_params.cu_seqlens_q
        _individual_lens = _cu[1:] - _cu[:-1]
        _max_individual = int(_individual_lens.max().item()) if _individual_lens.numel() > 0 else 0
        assert _max_individual <= global_target_len, (
            f"Individual request length ({_max_individual}) exceeds the global max sequence length "
            f"({global_target_len}). Increase --max-seqlen-per-dp-cp-rank / alignment, "
            f"or filter out overlong requests."
        )

    # Pad token-like tensors to the CP-local target length.
    tokens = _pad_seq_tensor(tokens, local_target_len)
    labels = _pad_seq_tensor(labels, local_target_len)
    loss_mask = _pad_seq_tensor(loss_mask, local_target_len)
    position_ids = _pad_seq_tensor(position_ids, local_target_len)

    # Copy THD metadata before optionally appending/padding sequence boundaries.
    cu_seqlens_q = packed_seq_params.cu_seqlens_q
    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
    cu_seqlens_q_padded = packed_seq_params.cu_seqlens_q_padded
    cu_seqlens_kv_padded = packed_seq_params.cu_seqlens_kv_padded

    # Represent post-pack padding as a dummy sequence when requested.
    target_cu_entries = None if max_num_seqs is None else max_num_seqs + 1
    has_dummy_padding_seq = pad_by_appending_dummy_seq and global_target_len > global_actual_T
    dummy_seq_len = global_target_len - global_actual_T if has_dummy_padding_seq else 0

    if has_dummy_padding_seq:
        cu_seqlens_q = _append_dummy_seq(cu_seqlens_q, global_target_len)
        cu_seqlens_kv = _append_dummy_seq(cu_seqlens_kv, global_target_len)
        cu_seqlens_q_padded = _append_dummy_seq(cu_seqlens_q_padded, global_target_len)
        cu_seqlens_kv_padded = _append_dummy_seq(cu_seqlens_kv_padded, global_target_len)

    # Pad cu_seqlens entry counts for static CUDA Graph inputs.
    if target_cu_entries is not None:
        cu_seqlens_q = _pad_cu_seqlens(cu_seqlens_q, target_cu_entries)
        cu_seqlens_kv = _pad_cu_seqlens(cu_seqlens_kv, target_cu_entries)
        cu_seqlens_q_padded = _pad_cu_seqlens(cu_seqlens_q_padded, target_cu_entries)
        cu_seqlens_kv_padded = _pad_cu_seqlens(cu_seqlens_kv_padded, target_cu_entries)

    # Rebuild PackedSeqParams with the padded tensor and metadata shapes.
    padded_params = PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        max_seqlen_q=(
            global_target_len
            if target_cu_entries is not None
            else max(packed_seq_params.max_seqlen_q, dummy_seq_len)
        ),
        max_seqlen_kv=(
            global_target_len
            if target_cu_entries is not None
            else max(packed_seq_params.max_seqlen_kv, dummy_seq_len)
        ),
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
        total_tokens=local_target_len if target_cu_entries is None else None,
        pad_between_seqs=False if has_dummy_padding_seq else packed_seq_params.pad_between_seqs,
    )

    # True marks padded local token slots for routing/loss paths.
    padding_mask = torch.arange(local_target_len, device=mask_device).unsqueeze(0) >= local_actual_T

    return tokens, labels, loss_mask, position_ids, padded_params, padding_mask
