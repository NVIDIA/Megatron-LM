# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native packed-THD indexing shared by DSv4 and CSA2.

Common helpers define segment indices, compression groups and static capacity.
CSA2 layouts add validity masks, rotary positions and metadata snapshots
for shared state. Logical cumulative lengths describe valid sequence prefixes;
padded cumulative lengths describe physical addresses. As in DSv4, a dummy
sequence registered in both sets of prefixes is processed as an ordinary sequence.
"""

import os
from dataclasses import dataclass

import torch
from torch import Tensor

from megatron.core.env_flags import env_flag
from megatron.core.packed_seq_params import PackedSeqParams


def batch_of_row(cu_seqlens_q: Tensor, total_q: int | None = None) -> Tensor:
    """For a THD-packed query of length ``total_q``, return a ``(total_q,)``
    int64 tensor where entry ``i`` is the index of the segment that owns
    query row ``i`` (i.e. the unique ``b`` with
    ``cu_seqlens_q[b] <= i < cu_seqlens_q[b+1]``).

    When ``total_q`` exceeds ``cu_seqlens_q[-1]`` (e.g. after
    ``pad_thd_for_cuda_graph`` pads token tensors to a static capacity),
    orphan rows are clamped to the last segment so the returned indices
    are always in ``[0, B-1]`` and never cause OOB on per-segment arrays.

    Used by every helper that needs to translate between per-row indices
    and per-segment cumulative tensors.

    Args:
        cu_seqlens_q: ``(B+1,)`` int — cumulative Q lengths.
        total_q: optional row count override; defaults to
            ``int(cu_seqlens_q[-1].item())`` (forces a GPU→CPU sync).

    Returns:
        ``(total_q,)`` int64.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    num_sequences = cu_seqlens_q.shape[0] - 1
    row_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp(
        max=max(num_sequences - 1, 0)
    )


def get_thd_compressed_capacity(
    total_tokens: int, max_seqlen: int, num_sequences: int, ratio: int
) -> int:
    """Return DSv4's host-known upper bound on packed compressed rows.

    The bound uses both physical token capacity and sequence metadata capacity,
    without reading cumulative lengths from the device. The exact compressed
    endpoint remains in the device-side compressed cumulative lengths.
    """
    return min(int(total_tokens) // ratio, num_sequences * (int(max_seqlen) // ratio))


def get_thd_compressed_cu_seqlens(cu_seqlens: Tensor, ratio: int) -> Tensor:
    """Return physical compressed prefixes, dropping each segment's incomplete tail."""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seg_compressed_lens = seq_lens // ratio
    return torch.cat(
        [
            torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device),
            seg_compressed_lens.cumsum(0).to(cu_seqlens.dtype),
        ]
    )


def get_thd_compressed_group_indices(
    cu_seqlens: Tensor, cu_seqlens_compressed: Tensor, ratio: int, total_comp: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Map compressed capacity rows to DSv4's per-segment token groups.

    Args:
        cu_seqlens: Physical token prefixes, shape [num_sequences + 1].
        cu_seqlens_compressed: Physical compressed prefixes of matching shape.
        ratio: Number of source tokens per compressed group.
        total_comp: Host-known compressed capacity, possibly beyond the true
            compressed endpoint.

    Returns:
        ``(gather_idx, local_pos, batch_ids, valid_comp)``. Gather indices have
        shape [total_comp, ratio]; the other tensors have shape [total_comp].
        ``local_pos`` is a sequence-local group ID, before multiplication by the
        ratio for rotary positions. Unused capacity rows retain DSv4's gather
        indices [0, ..., ratio - 1], local position zero, and valid_comp=False.
        Validity here describes physical groups, not logical token padding.
    """
    device = cu_seqlens.device
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_compressed.dtype)
    batch_ids = batch_of_row(cu_seqlens_compressed, total_q=total_comp)
    valid_comp = row_idx < cu_seqlens_compressed[-1]
    local_pos = row_idx - cu_seqlens_compressed[batch_ids]
    local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
    base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
    base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
    offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
    gather_idx = base + offsets
    return gather_idx, local_pos, batch_ids, valid_comp


# THD padding rows must reach the attention kernels and the following projection as zeros.
# With MCORE_CSA2_SPARSE_PAD_ZERO on (the default), CSA2 writes zeros into only those rows,
# in place, instead of building a masked copy of each whole tensor; =0 restores masked_fill.
# Both write the same +0 values, so results are bit-identical.
SPARSE_PAD_ZERO = env_flag("MCORE_CSA2_SPARSE_PAD_ZERO", True)


def _pad_row_capacity() -> int:
    """Parse MCORE_CSA2_PAD_ROW_CAPACITY, a positive integer (default 512)."""
    value = os.environ.get("MCORE_CSA2_PAD_ROW_CAPACITY")
    if value is None:
        return 512
    try:
        capacity = int(value)
    except ValueError:
        capacity = 0
    if capacity < 1:
        raise ValueError(
            f"Environment variable MCORE_CSA2_PAD_ROW_CAPACITY={value!r} is not a positive integer"
        )
    return capacity


# Static number of padding rows one zeroing call handles (the row list has a fixed shape, so
# it can be captured in CUDA graphs). More padding rows in one call trip a device assert.
PAD_ROW_CAPACITY = _pad_row_capacity()


def padding_rows(invalid: Tensor) -> tuple[Tensor, Tensor]:
    """List the rows flagged in ``invalid`` (one flag per row) for :func:`zero_rows_`.

    The list has a static shape and is built without host syncs, so it can run inside CUDA
    graphs. Returns ``rows`` of length ``PAD_ROW_CAPACITY`` and a per-entry ``keep`` flag.
    Unused entries name the first valid row with ``keep`` set, so writing that row back
    leaves it unchanged; when every row is padding, they name row 0 with ``keep`` unset.
    """
    invalid = invalid.reshape(-1)
    count = invalid.numel()
    capacity = PAD_ROW_CAPACITY
    position = invalid.cumsum(0) - 1
    if count:
        torch._assert_async(
            position[-1] < capacity,
            "CSA2 THD padding rows exceed MCORE_CSA2_PAD_ROW_CAPACITY; raise it or set "
            "MCORE_CSA2_SPARSE_PAD_ZERO=0",
        )
    slot = torch.where(invalid & (position < capacity), position, capacity)
    rows = torch.full((capacity + 1,), -1, dtype=torch.long, device=invalid.device)
    rows = rows.scatter(0, slot, torch.arange(count, device=invalid.device))[:capacity]
    used = rows >= 0
    if count:
        # index_select, not invalid[anchor]: indexing with a 0-dim tensor syncs with the host.
        anchor = (~invalid).int().argmax().view(1)
        anchor_valid = ~invalid.index_select(0, anchor)
    else:
        anchor = torch.zeros(1, dtype=torch.long, device=invalid.device)
        anchor_valid = torch.zeros(1, dtype=torch.bool, device=invalid.device)
    keep = ~used & anchor_valid
    return torch.where(used, rows, anchor), keep


def zero_rows_(tensor: Tensor, rows: Tensor, keep: Tensor) -> Tensor:
    """Zero ``tensor[rows]`` where ``keep`` is unset, in place along dim 0, outside autograd.

    Writes go through ``.data``, so tensors that other autograd nodes saved stay usable;
    callers keep the gradients of padding rows at zero themselves (see
    :func:`zero_padding_rows_`). Returns ``tensor``.
    """
    if tensor.numel() == 0:
        # Empty THD batches: view(0, -1) is ambiguous, and padding_rows names row 0.
        return tensor
    flat = tensor.data.view(tensor.shape[0], -1)
    # where, not a 0/1 product: padding rows may hold NaN or inf, and NaN * 0 is NaN.
    flat.index_copy_(0, rows, torch.where(keep[:, None], flat.index_select(0, rows), 0))
    return tensor


def save_rows(tensor: Tensor, rows: Tensor) -> Tensor | None:
    """Copy ``tensor[rows]`` along dim 0, outside autograd, for :func:`restore_rows_`."""
    if tensor.numel() == 0:
        return None
    return tensor.data.view(tensor.shape[0], -1).index_select(0, rows)


def restore_rows_(tensor: Tensor, rows: Tensor, saved: Tensor | None) -> Tensor:
    """Write rows copied by :func:`save_rows` back in place, undoing a :func:`zero_rows_`.

    Duplicate row entries carry identical values, so their write order does not matter.
    """
    if saved is not None:
        tensor.data.view(tensor.shape[0], -1).index_copy_(0, rows, saved)
    return tensor


def zero_padding_rows_(invalid: Tensor, *tensors: Tensor) -> tuple[Tensor, ...]:
    """In-place counterpart of ``masked_fill(invalid, 0)`` over dim 0 for contiguous tensors.

    Zeroes the padding rows of each tensor and, through a gradient hook, the same rows of
    each tensor's gradient, as masked_fill's backward does.
    """
    rows, keep = padding_rows(invalid)

    def zero_grad_rows(grad):
        """Zero the padding rows of an incoming gradient."""
        return zero_rows_(grad.contiguous(), rows, keep)

    for tensor in tensors:
        zero_rows_(tensor, rows, keep)
        if tensor.requires_grad:
            tensor.register_hook(zero_grad_rows)
    return tensors


# Device-side validation of packed THD metadata: every check is a compare/reduce plus a
# torch._assert_async launch, about 17 per layout build, captured CUDA graphs included. The
# checks never change a tensor, and a failing one aborts the CUDA context without a Python
# stack, so they are a debugging aid and off by default; MCORE_CSA2_DEVICE_CHECKS=1 turns
# them on (recommended for new data pipelines). Metadata held on the CPU is always validated.
DEVICE_CHECKS = env_flag("MCORE_CSA2_DEVICE_CHECKS", False)


def metadata_checks_enabled(device: torch.device) -> bool:
    """Whether THD metadata on ``device`` is validated: always on the host, on CUDA only
    with ``MCORE_CSA2_DEVICE_CHECKS=1``."""
    return DEVICE_CHECKS or device.type != "cuda"


def _check_values(condition_fn, message: str, device: torch.device) -> None:
    """Validate metadata on ``device`` without transferring CUDA scalars to the host.

    ``condition_fn`` builds the condition tensor. It is not called when the check is
    disabled, so a disabled device check launches no kernel.
    """
    if not metadata_checks_enabled(device):
        return
    condition = condition_fn()
    if condition.is_cuda:
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def get_thd_token_metadata(
    cu_seqlens: Tensor, cu_seqlens_padded: Tensor, global_rows: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Map global packed rows, including left halos, to sequence IDs, positions and validity."""
    num_sequences = cu_seqlens.numel() - 1
    if num_sequences == 0:
        return (
            torch.full_like(global_rows, -1),
            torch.zeros_like(global_rows),
            torch.zeros_like(global_rows, dtype=torch.bool),
        )
    sequence_ids = torch.bucketize(global_rows, cu_seqlens_padded[1:], right=True).clamp_max(
        num_sequences - 1
    )
    positions = global_rows - cu_seqlens_padded[sequence_ids].to(torch.int64)
    assigned = (global_rows >= 0) & (global_rows < cu_seqlens_padded[-1])
    valid = assigned & (positions < cu_seqlens.diff()[sequence_ids])
    return sequence_ids.masked_fill(~assigned, -1), positions.masked_fill(~valid, 0), valid


@dataclass(frozen=True)
class CSA2THDCompressionLayout:
    """Physical compression slots and their sequence-local rotary positions.

    ``cu_seqlens_padded`` reserves ``floor(physical_length / ratio)`` slots per
    sequence, following DSv4. ``capacity`` can additionally include unused tail
    slots. ``cu_seqlens`` counts groups within each logical sequence length;
    ``cu_seqlens_padded`` defines their physical addresses. Consumers must retain
    ``valid_groups`` and must not renumber the sequence-local groups.
    Invalid slots have safe source indices and zero rotary positions.
    Local CP compact layouts set ``cp_rank/cp_size`` and retain original
    sequence positions despite using local compressed prefixes.
    """

    ratio: int
    total_tokens: int
    capacity: int
    max_seqlen: int
    cu_seqlens: torch.Tensor
    cu_seqlens_padded: torch.Tensor
    sequence_ids: torch.Tensor
    position_ids: torch.Tensor
    source_indices: torch.Tensor
    valid_groups: torch.Tensor
    cp_rank: int = 0
    cp_size: int = 1


@dataclass(frozen=True)
class CSA2THDLayout:
    """One forward's immutable snapshot of the packed self-attention layout.

    Tensor fields must be treated as read-only. Sequence IDs remain physical
    sequence IDs for padding within a sequence; unassigned capacity has ID -1.
    Invalid token rotary positions are zero. ``valid_tokens`` follows DSv4's
    logical lengths within each physical segment.
    """

    total_tokens: int
    max_seqlen: int
    cu_seqlens: torch.Tensor
    cu_seqlens_padded: torch.Tensor
    sequence_ids: torch.Tensor
    position_ids: torch.Tensor
    valid_tokens: torch.Tensor
    cp_rank: int = 0
    cp_size: int = 1

    @property
    def global_start(self) -> int:
        """First global physical token row owned by this contiguous CP partition."""
        return self.cp_rank * self.total_tokens

    @property
    def global_total_tokens(self) -> int:
        """Physical capacity across all equal-sized CP partitions."""
        return self.cp_size * self.total_tokens

    def for_compression(self, ratio: int) -> CSA2THDCompressionLayout:
        """Build canonical global r1/r2 groups without crossing sequence boundaries.

        Under CP, source indices address the global packed buffer. Local
        projection uses the separate CP compact layout and its halo indices.
        """
        if type(ratio) is not int or ratio not in (1, 2):
            raise ValueError("CSA2 THD compression ratio must be 1 or 2")
        num_sequences = self.cu_seqlens.numel() - 1
        capacity = get_thd_compressed_capacity(
            self.global_total_tokens, self.max_seqlen, num_sequences, ratio
        )
        cu_physical = get_thd_compressed_cu_seqlens(self.cu_seqlens_padded, ratio)
        sources, local_groups, sequence_ids, assigned = get_thd_compressed_group_indices(
            self.cu_seqlens_padded, cu_physical, ratio, capacity
        )
        # Keep DSv4's physical addresses while excluding groups past the logical
        # sequence length. Dummy sequences remain ordinary logical segments.
        sources = sources.to(torch.int64).masked_fill(~assigned[:, None], 0)
        if self.cp_size == 1:
            source_valid = self.valid_tokens[sources]
        else:
            _, _, source_valid = get_thd_token_metadata(
                self.cu_seqlens, self.cu_seqlens_padded, sources
            )
        valid_groups = assigned & source_valid.all(dim=-1)
        positions = (local_groups.to(torch.int64) * ratio).masked_fill(~valid_groups, 0)
        sequence_ids = sequence_ids.masked_fill(~assigned, -1)

        return CSA2THDCompressionLayout(
            ratio=ratio,
            total_tokens=self.global_total_tokens,
            capacity=capacity,
            max_seqlen=self.max_seqlen // ratio,
            cu_seqlens=get_thd_compressed_cu_seqlens(self.cu_seqlens, ratio),
            cu_seqlens_padded=cu_physical,
            sequence_ids=sequence_ids,
            position_ids=positions,
            source_indices=sources,
            valid_groups=valid_groups,
        )

    def validate_compatible(
        self,
        packed_seq_params: PackedSeqParams,
        total_tokens: int,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        """Reject reusing a forward state with a different packed layout.

        Snapshotted prefixes also detect in-place mutation of the caller's
        original metadata. CUDA value comparisons remain on the device.
        """
        other = build_csa2_thd_layout(packed_seq_params, total_tokens, cp_group=cp_group)
        self.validate_layout(other)

    def validate_layout(self, other: "CSA2THDLayout") -> None:
        """Validate another snapshot before sharing compressed state with it."""
        if self is other:
            return
        if (
            self.total_tokens != other.total_tokens
            or self.cp_rank != other.cp_rank
            or self.cp_size != other.cp_size
            or self.max_seqlen != other.max_seqlen
            or self.cu_seqlens.shape != other.cu_seqlens.shape
            or self.cu_seqlens.dtype != other.cu_seqlens.dtype
            or self.cu_seqlens.device != other.cu_seqlens.device
        ):
            raise ValueError("CSA2 state cannot be reused with a different THD layout")
        for name in ("cu_seqlens", "cu_seqlens_padded", "valid_tokens"):
            _check_values(
                lambda: (getattr(self, name) == getattr(other, name)).all(),
                "CSA2 state cannot be reused with a different THD layout",
                self.cu_seqlens.device,
            )


def build_csa2_thd_layout(
    packed_seq_params: PackedSeqParams,
    total_tokens: int,
    *,
    cp_group: torch.distributed.ProcessGroup | None = None,
) -> CSA2THDLayout:
    """Resolve local token validity and positions from global packed prefixes.

    Args:
        packed_seq_params: Self-attention metadata in THD format. Query and KV
            layouts must agree. Padded prefixes fall back to logical prefixes.
        total_tokens: Local physical token capacity, including padding. With CP,
            ranks own equal contiguous intervals of the global packed buffer.
        cp_group: Static CP group when it is not specified by the packed metadata.

    Returns:
        An independent metadata snapshot with token validity and positions.
    """
    if packed_seq_params.qkv_format != "thd":
        raise ValueError("CSA2 THD layout requires qkv_format='thd'")
    if type(total_tokens) is not int or total_tokens < 0:
        raise ValueError("CSA2 THD total_tokens must be a non-negative integer")
    group = packed_seq_params.cp_group if packed_seq_params.cp_group is not None else cp_group
    cp_size, cp_rank = (group.size(), group.rank()) if group is not None else (1, 0)
    if packed_seq_params.local_cp_size not in (None, cp_size):
        raise ValueError("CSA2 THD local_cp_size requires a matching explicit CP group")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("CSA2 THD requires valid CP group membership")
    # Main stores the layout on TransformerConfig, not PackedSeqParams. The
    # CSA2 config/attention wrapper validates contiguous CP before this helper.
    global_total = total_tokens * cp_size

    cu_q = packed_seq_params.cu_seqlens_q
    cu_kv = packed_seq_params.cu_seqlens_kv
    physical_q = packed_seq_params.cu_seqlens_q_padded
    physical_kv = packed_seq_params.cu_seqlens_kv_padded
    physical_q = cu_q if physical_q is None else physical_q
    physical_kv = cu_kv if physical_kv is None else physical_kv
    prefixes = (cu_q, cu_kv, physical_q, physical_kv)
    for prefix in prefixes:
        if not isinstance(prefix, torch.Tensor) or prefix.ndim != 1 or prefix.numel() == 0:
            raise ValueError("CSA2 THD cumulative lengths must be non-empty 1D tensors")
        if prefix.dtype not in (torch.int32, torch.int64):
            raise ValueError("CSA2 THD cumulative lengths must have int32 or int64 dtype")
        if global_total > torch.iinfo(prefix.dtype).max:
            raise ValueError("CSA2 THD token capacity exceeds cumulative-length dtype limits")
        if prefix.shape != cu_q.shape or prefix.dtype != cu_q.dtype or prefix.device != cu_q.device:
            raise ValueError(
                "CSA2 THD query/KV metadata must have matching shape, dtype and device"
            )
        _check_values(
            lambda: prefix[0] == 0, "CSA2 THD cumulative lengths must start at zero", prefix.device
        )
        _check_values(
            lambda: (prefix.diff() >= 0).all(),
            "CSA2 THD cumulative lengths must be monotonic",
            prefix.device,
        )
        _check_values(
            lambda: prefix[-1] <= global_total,
            "CSA2 THD metadata exceeds physical token capacity",
            prefix.device,
        )
    device = cu_q.device
    _check_values(
        lambda: (cu_q == cu_kv).all(),
        "CSA2 THD self-attention requires equal query/KV lengths",
        device,
    )
    _check_values(
        lambda: (physical_q == physical_kv).all(),
        "CSA2 THD self-attention requires equal query/KV physical lengths",
        device,
    )
    _check_values(
        lambda: (cu_q.diff() <= physical_q.diff()).all(),
        "CSA2 THD valid sequence length exceeds its physical segment",
        device,
    )
    max_seqlen = packed_seq_params.max_seqlen_q
    max_seqlen_kv = packed_seq_params.max_seqlen_kv
    for limit in (max_seqlen, max_seqlen_kv):
        if type(limit) is not int or limit < 0:
            raise ValueError("CSA2 THD max_seqlen_q/kv must be non-negative host integers")
        _check_values(
            lambda: (physical_q.diff() <= limit).all(),
            "CSA2 THD physical sequence length exceeds max_seqlen_q/kv",
            device,
        )
    if max_seqlen != max_seqlen_kv:
        raise ValueError("CSA2 THD self-attention requires equal max_seqlen_q/kv")

    # Snapshot metadata so a shared state cannot silently observe a subsequent
    # microbatch's in-place replacement of static cumulative-length buffers.
    cu_q = cu_q.clone()
    physical_q = physical_q.clone()
    rows = (
        torch.arange(total_tokens, device=cu_q.device, dtype=torch.int64) + cp_rank * total_tokens
    )
    sequence_ids, positions, valid_tokens = get_thd_token_metadata(cu_q, physical_q, rows)

    return CSA2THDLayout(
        total_tokens=total_tokens,
        max_seqlen=max_seqlen,
        cu_seqlens=cu_q,
        cu_seqlens_padded=physical_q,
        sequence_ids=sequence_ids,
        position_ids=positions,
        valid_tokens=valid_tokens,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )


def get_window_topk_idxs_thd(window_size: int, cu_seqlens: Tensor, *, total_q: int) -> Tensor:
    """Return document-relative windows for a packed native reference."""
    rows = torch.arange(total_q, device=cu_seqlens.device)
    sequence = batch_of_row(cu_seqlens, total_q)
    position = rows - cu_seqlens[sequence]
    keys = (position - window_size + 1).clamp_min(0)[:, None] + torch.arange(
        window_size, device=rows.device
    )
    valid = (rows < cu_seqlens[-1])[:, None] & (keys <= position[:, None])
    return keys.masked_fill(~valid, -1).int()
