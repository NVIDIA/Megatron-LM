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


def _pad_padding_mask(mask: Tensor, target_len: int) -> Tensor:
    """Pad a [..., seq] bool padding mask to ``target_len`` with True."""
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
    """Pad a cu_seqlens tensor to exactly ``target_entries`` entries.

    Asserts the actual entry count does not exceed ``target_entries``. An
    oversized pack cannot be represented by the configured static cu_seqlens
    buffer and would not match captured CUDA Graph replay shapes.
    """
    if cu_seqlens is None:
        return None
    actual_entries = cu_seqlens.shape[0]
    assert actual_entries <= target_entries, (
        f"Actual num_seqs ({actual_entries - 1}) exceeds thd_max_packed_sequences "
        f"({target_entries - 1}). Increase --thd-max-packed-sequences, decrease "
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

    ``dummy_end`` is the endpoint in this tensor's coordinate space. Padded
    metadata uses the physical target; compact valid-token metadata uses its
    previous endpoint plus the dummy tail length when earlier gaps exist.
    """
    if cu_seqlens is None:
        return None

    dummy = torch.full((1,), int(dummy_end), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.cat((cu_seqlens, dummy), dim=0)


def _extend_last_padded_sequence(
    cu_seqlens_padded: Optional[Tensor], padded_end: int
) -> Optional[Tensor]:
    """Extend the final padded sequence to ``padded_end`` without adding a boundary."""
    if cu_seqlens_padded is None:
        return None

    assert (
        cu_seqlens_padded.numel() >= 2
    ), "Non-dummy THD tail padding requires at least one real sequence."
    current_end = int(cu_seqlens_padded[-1].item())
    assert (
        current_end <= padded_end
    ), f"THD padded endpoint ({current_end}) exceeds padding target ({padded_end})."
    if current_end == padded_end:
        return cu_seqlens_padded

    extended = cu_seqlens_padded.clone()
    extended[-1] = padded_end
    return extended


def _last_padded_sequence_length(cu_seqlens_padded: Optional[Tensor]) -> int:
    """Return the physical length of the final sequence, or zero when unavailable."""
    if cu_seqlens_padded is None or cu_seqlens_padded.numel() < 2:
        return 0
    return int((cu_seqlens_padded[-1] - cu_seqlens_padded[-2]).item())


def _round_up_to_alignment(value: int, alignment: int) -> int:
    assert alignment > 0, f"Packed sequence padding alignment must be > 0, got {alignment}."
    return ((value + alignment - 1) // alignment) * alignment


def get_thd_padding_kwargs(
    pad_packed_seq_alignment: Union[int, Literal["max"]],
    max_seqlen_per_dp_cp_rank: Optional[int],
    thd_max_packed_sequences: Optional[int],
    cuda_graph_static: bool,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Resolve ``pad_sequence_for_thd`` kwargs from the training config.

    ``--pad-packed-seq-alignment`` has two forms:

    - ``max`` pads token-like tensors to ``max_seqlen_per_dp_cp_rank``;
    - a positive value pads token-like tensors to a multiple of that value.

    Padding cu_seqlens to ``thd_max_packed_sequences + 1`` is a CUDA Graph static-input
    requirement. Eager pad-to-max should preserve sequence metadata so kernels
    continue to see the real packed sequence boundaries.
    """
    if cuda_graph_static:
        return None, int(max_seqlen_per_dp_cp_rank), thd_max_packed_sequences

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
    padding_mask: Optional[Tensor] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
    cp_size: Optional[int] = None,
    cp_rank: Optional[int] = None,
) -> Tuple[int, int, int, int, torch.device]:
    """Resolve local/global THD padding lengths without changing tensors.

    Returns:
        local_actual_T: Current rank's physical row count.
        global_actual_T: Global packed length represented by THD metadata.
        local_target_len: Current rank's padded token-like tensor length.
        global_target_len: Global padded endpoint represented by THD metadata.
        mask_device: Device used to build the returned padding mask.
    """

    cp_size, cp_rank = _resolve_thd_cp_geometry(
        packed_seq_params, cp_group=cp_group, cp_size=cp_size, cp_rank=cp_rank
    )

    # Find a tensor that already carries this rank's local physical row count.
    local_tensor_T = None
    mask_device = None
    for candidate in (tokens, labels, loss_mask, position_ids, padding_mask):
        if candidate is not None:
            local_tensor_T = int(candidate.shape[-1])
            mask_device = candidate.device
            break

    # The padded endpoint describes tensor storage. The unpadded endpoint is
    # only the compact valid-token count when gaps exist between sequences.
    has_local_tensor = local_tensor_T is not None
    if packed_seq_params.cu_seqlens_q_padded is not None:
        global_actual_T = int(packed_seq_params.cu_seqlens_q_padded[-1].item())
        if mask_device is None:
            mask_device = packed_seq_params.cu_seqlens_q_padded.device
    elif packed_seq_params.cu_seqlens_q is not None:
        global_actual_T = int(packed_seq_params.cu_seqlens_q[-1].item())
        if mask_device is None:
            mask_device = packed_seq_params.cu_seqlens_q.device
    else:
        assert has_local_tensor, (
            "packed_seq_params.cu_seqlens_q must be available to derive padding_mask "
            "when tokens/labels/loss_mask/position_ids/padding_mask are all None."
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


def _resolve_thd_cp_geometry(
    packed_seq_params: PackedSeqParams,
    cp_group: Optional[dist.ProcessGroup] = None,
    cp_size: Optional[int] = None,
    cp_rank: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve CP geometry for THD padding.

    Callers with a known CP group or explicit size/rank should pass it here.
    Falling back to ``parallel_state`` preserves legacy call sites.
    """
    if cp_group is not None:
        return int(dist.get_world_size(group=cp_group)), int(dist.get_rank(group=cp_group))

    if cp_size is not None:
        cp_size = int(cp_size)
        if cp_rank is not None:
            return cp_size, int(cp_rank)
        if cp_size == 1:
            return cp_size, 0

    if packed_seq_params.cp_group is not None:
        cp_group = packed_seq_params.cp_group
        return int(dist.get_world_size(group=cp_group)), int(dist.get_rank(group=cp_group))

    if cp_size is None and packed_seq_params.local_cp_size is not None:
        cp_size = int(packed_seq_params.local_cp_size)
        if cp_size == 1:
            return cp_size, 0

    # Last resort for compatibility with older callers that do not thread CP
    # geometry through PackedSeqParams.
    from megatron.core import parallel_state

    if cp_size is None:
        cp_size = int(parallel_state.get_context_parallel_world_size())
    if cp_rank is None:
        cp_rank = int(parallel_state.get_context_parallel_rank()) if cp_size > 1 else 0
    return int(cp_size), int(cp_rank)


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
            tail as an extra dummy sequence in cu_seqlens metadata. Otherwise,
            keep valid boundaries unchanged and extend the final padded
            sequence boundary to cover the tail.
        padding_mask: Existing bool padding mask for already-packed tokens,
            with True marking padding positions.
        cp_group: Context-parallel process group for resolving local/global
            THD padding lengths. If omitted, ``packed_seq_params.cp_group`` is
            used when available.
        cp_size: Explicit context-parallel world size used when no CP group is
            available.
        cp_rank: Explicit context-parallel rank used with ``cp_size``.

    Notes:
        - On stages without token-like tensors, an already CP-sliced padding
          mask supplies the local physical row count. Legacy callers without a
          padding mask ask TE which packed rows this CP rank would receive.
        - When ``pad_by_appending_dummy_seq`` is true, the padding tail is also
          represented as an ordinary dummy sequence in cu_seqlens metadata.
          Existing valid/physical gaps remain intact.
        - When false, valid cu_seqlens remain unchanged while the final padded
          endpoint is extended, treating the tail as padding of the last sequence.
          With CP, callers must do this on the global tensors before slicing.
        - With zigzag context parallelism, a dummy tail must be divisible by
          twice the effective CP size, matching TE's THD partitioning requirement.
        - ``max_num_seqs`` pads all four cu_seqlens tensors; this is required
          by CUDA Graph replay because those tensors are graph inputs.
          It also fixes ``pad_between_seqs`` to ``True`` so both CUDA Graph
          implementations see one stable, conservative Python signature.

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
            padding_mask=padding_mask,
            cp_group=cp_group,
            cp_size=cp_size,
            cp_rank=cp_rank,
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

    # Represent post-pack padding either as a dummy sequence or as physical
    # padding attached to the final real sequence.
    target_cu_entries = None if max_num_seqs is None else max_num_seqs + 1
    has_dummy_padding_seq = pad_by_appending_dummy_seq and global_target_len > global_actual_T
    has_non_dummy_padding_tail = (
        not pad_by_appending_dummy_seq and global_target_len > global_actual_T
    )
    dummy_seq_len = global_target_len - global_actual_T if has_dummy_padding_seq else 0
    last_padded_q_len = 0
    last_padded_kv_len = 0

    # Existing valid/padded differences describe physical gaps between real
    # sequences. Preserve that contract while adding an ordinary dummy sequence:
    # its valid and physical lengths are both exactly the new tail length.
    has_inter_sequence_padding = packed_seq_params.pad_between_seqs is True
    if has_dummy_padding_seq and packed_seq_params.pad_between_seqs is None:
        has_inter_sequence_padding = any(
            valid is not None and padded is not None and not torch.equal(valid, padded)
            for valid, padded in (
                (cu_seqlens_q, cu_seqlens_q_padded),
                (cu_seqlens_kv, cu_seqlens_kv_padded),
            )
        )

    if has_dummy_padding_seq:
        cp_size, _ = _resolve_thd_cp_geometry(
            packed_seq_params, cp_group=cp_group, cp_size=cp_size, cp_rank=cp_rank
        )
        if cp_size > 1 and packed_seq_params.cp_partition_mode == "zigzag":
            assert dummy_seq_len % (2 * cp_size) == 0, (
                f"THD dummy padding length ({dummy_seq_len}) must be divisible by "
                f"2 * context_parallel_size ({2 * cp_size}) for zigzag partitioning. "
                "Choose an even CP-local padding target/alignment."
            )

        if has_inter_sequence_padding:
            if cu_seqlens_q is not None:
                cu_seqlens_q = _append_dummy_seq(
                    cu_seqlens_q, int(cu_seqlens_q[-1].item()) + dummy_seq_len
                )
            if cu_seqlens_kv is not None:
                cu_seqlens_kv = _append_dummy_seq(
                    cu_seqlens_kv, int(cu_seqlens_kv[-1].item()) + dummy_seq_len
                )
        else:
            # Legacy equal-boundary case: retain the original dummy behavior.
            cu_seqlens_q = _append_dummy_seq(cu_seqlens_q, global_target_len)
            cu_seqlens_kv = _append_dummy_seq(cu_seqlens_kv, global_target_len)
        cu_seqlens_q_padded = _append_dummy_seq(cu_seqlens_q_padded, global_target_len)
        cu_seqlens_kv_padded = _append_dummy_seq(cu_seqlens_kv_padded, global_target_len)
    elif has_non_dummy_padding_tail:
        # Keep compact valid-token coordinates unchanged. Only physical
        # metadata grows, so the tail belongs to the final real sequence.
        non_dummy_cp_size, _ = _resolve_thd_cp_geometry(
            packed_seq_params, cp_group=cp_group, cp_size=cp_size, cp_rank=cp_rank
        )
        assert non_dummy_cp_size == 1, (
            "Non-dummy THD tail padding with context parallelism must be applied "
            "to global tensors before CP slicing."
        )
        if cu_seqlens_q_padded is None:
            cu_seqlens_q_padded = cu_seqlens_q
        if cu_seqlens_kv_padded is None:
            cu_seqlens_kv_padded = cu_seqlens_kv
        assert (
            cu_seqlens_q_padded is not None or cu_seqlens_kv_padded is not None
        ), "Non-dummy THD tail padding requires metadata for at least one real sequence."
        cu_seqlens_q_padded = _extend_last_padded_sequence(cu_seqlens_q_padded, global_target_len)
        cu_seqlens_kv_padded = _extend_last_padded_sequence(cu_seqlens_kv_padded, global_target_len)
        last_padded_q_len = _last_padded_sequence_length(cu_seqlens_q_padded)
        last_padded_kv_len = _last_padded_sequence_length(cu_seqlens_kv_padded)

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
            else max(
                packed_seq_params.max_seqlen_q,
                dummy_seq_len,
                last_padded_q_len if has_non_dummy_padding_tail else 0,
            )
        ),
        max_seqlen_kv=(
            global_target_len
            if target_cu_entries is not None
            else max(
                packed_seq_params.max_seqlen_kv,
                dummy_seq_len,
                last_padded_kv_len if has_non_dummy_padding_tail else 0,
            )
        ),
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
        cp_partition_mode=packed_seq_params.cp_partition_mode,
        total_tokens=local_target_len if target_cu_entries is None else None,
        pad_between_seqs=(
            True
            if target_cu_entries is not None
            else (
                has_inter_sequence_padding
                if has_dummy_padding_seq
                else (True if has_non_dummy_padding_tail else packed_seq_params.pad_between_seqs)
            )
        ),
    )

    # True marks padded local token slots for routing/loss paths.
    tail_padding_mask = (
        torch.arange(local_target_len, device=mask_device).unsqueeze(0) >= local_actual_T
    )
    if padding_mask is None:
        padding_mask = tail_padding_mask
    else:
        padding_mask = _pad_padding_mask(padding_mask, local_target_len) | tail_padding_mask

    return tokens, labels, loss_mask, position_ids, padded_params, padding_mask
