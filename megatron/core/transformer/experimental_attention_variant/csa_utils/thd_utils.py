# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native packed-THD indexing shared by DSv4 and CSA2.

Common helpers define segment indices, compression groups and static capacity.
CSA2's CP=1 layouts add validity masks, rotary positions and metadata snapshots
for shared state. Logical cumulative lengths describe valid sequence prefixes;
padded cumulative lengths describe physical addresses. As in DSv4, a dummy
sequence registered in both sets of prefixes is processed as an ordinary sequence.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

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


def _check_values(condition: torch.Tensor, message: str) -> None:
    """Validate device metadata without transferring CUDA scalars to the host."""
    if condition.is_cuda:
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class CSA2THDCompressionLayout:
    """Physical compression slots and their sequence-local rotary positions.

    ``cu_seqlens_padded`` reserves ``floor(physical_length / ratio)`` slots per
    sequence, following DSv4. ``capacity`` can additionally include unused tail
    slots. ``cu_seqlens`` counts groups within each logical sequence length;
    ``cu_seqlens_padded`` defines their physical addresses. Consumers must retain
    ``valid_groups`` and must not renumber the sequence-local groups.
    Invalid slots have safe source indices and zero rotary positions.
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

    def for_compression(self, ratio: int) -> CSA2THDCompressionLayout:
        """Build r1/r2 groups without crossing sequence or padding boundaries."""
        if type(ratio) is not int or ratio not in (1, 2):
            raise ValueError("CSA2 THD compression ratio must be 1 or 2")
        num_sequences = self.cu_seqlens.numel() - 1
        capacity = get_thd_compressed_capacity(
            self.total_tokens, self.max_seqlen, num_sequences, ratio
        )
        cu_physical = get_thd_compressed_cu_seqlens(self.cu_seqlens_padded, ratio)
        sources, local_groups, sequence_ids, assigned = get_thd_compressed_group_indices(
            self.cu_seqlens_padded, cu_physical, ratio, capacity
        )
        # Keep DSv4's physical addresses while excluding groups past the logical
        # sequence length. Dummy sequences remain ordinary logical segments.
        sources = sources.to(torch.int64).masked_fill(~assigned[:, None], 0)
        valid_groups = assigned & self.valid_tokens[sources].all(dim=-1)
        positions = (local_groups.to(torch.int64) * ratio).masked_fill(~valid_groups, 0)
        sequence_ids = sequence_ids.masked_fill(~assigned, -1)

        return CSA2THDCompressionLayout(
            ratio=ratio,
            total_tokens=self.total_tokens,
            capacity=capacity,
            max_seqlen=self.max_seqlen // ratio,
            cu_seqlens=get_thd_compressed_cu_seqlens(self.cu_seqlens, ratio),
            cu_seqlens_padded=cu_physical,
            sequence_ids=sequence_ids,
            position_ids=positions,
            source_indices=sources,
            valid_groups=valid_groups,
        )

    def validate_compatible(self, packed_seq_params: PackedSeqParams, total_tokens: int) -> None:
        """Reject reusing a forward state with a different packed layout.

        Snapshotted prefixes also detect in-place mutation of the caller's
        original metadata. CUDA value comparisons remain on the device.
        """
        other = build_csa2_thd_layout(packed_seq_params, total_tokens)
        self.validate_layout(other)

    def validate_layout(self, other: "CSA2THDLayout") -> None:
        """Validate another snapshot before sharing compressed state with it."""
        if self is other:
            return
        if (
            self.total_tokens != other.total_tokens
            or self.max_seqlen != other.max_seqlen
            or self.cu_seqlens.shape != other.cu_seqlens.shape
            or self.cu_seqlens.dtype != other.cu_seqlens.dtype
            or self.cu_seqlens.device != other.cu_seqlens.device
        ):
            raise ValueError("CSA2 state cannot be reused with a different THD layout")
        for name in ("cu_seqlens", "cu_seqlens_padded", "valid_tokens"):
            _check_values(
                (getattr(self, name) == getattr(other, name)).all(),
                "CSA2 state cannot be reused with a different THD layout",
            )


def build_csa2_thd_layout(packed_seq_params: PackedSeqParams, total_tokens: int) -> CSA2THDLayout:
    """Resolve logical validity and physical addresses for CP=1 packed tokens.

    Args:
        packed_seq_params: Self-attention metadata in THD format. Query and KV
            layouts must agree. Padded prefixes fall back to logical prefixes.
        total_tokens: Actual physical token tensor length, including all tail
            capacity introduced by pad_packed_seq_alignment.

    Returns:
        An independent metadata snapshot with token validity and positions.
    """
    if packed_seq_params.qkv_format != "thd":
        raise ValueError("CSA2 THD layout requires qkv_format='thd'")
    if type(total_tokens) is not int or total_tokens < 0:
        raise ValueError("CSA2 THD total_tokens must be a non-negative integer")
    if packed_seq_params.local_cp_size not in (None, 1):
        raise ValueError("CSA2 THD layout currently requires CP=1")
    if packed_seq_params.cp_group is not None and packed_seq_params.cp_group.size() != 1:
        raise ValueError("CSA2 THD layout currently requires CP=1")

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
        if total_tokens > torch.iinfo(prefix.dtype).max:
            raise ValueError("CSA2 THD token capacity exceeds cumulative-length dtype limits")
        if prefix.shape != cu_q.shape or prefix.dtype != cu_q.dtype or prefix.device != cu_q.device:
            raise ValueError(
                "CSA2 THD query/KV metadata must have matching shape, dtype and device"
            )
        _check_values(prefix[0] == 0, "CSA2 THD cumulative lengths must start at zero")
        _check_values((prefix.diff() >= 0).all(), "CSA2 THD cumulative lengths must be monotonic")
        _check_values(
            prefix[-1] <= total_tokens, "CSA2 THD metadata exceeds physical token capacity"
        )
    _check_values((cu_q == cu_kv).all(), "CSA2 THD self-attention requires equal query/KV lengths")
    _check_values(
        (physical_q == physical_kv).all(),
        "CSA2 THD self-attention requires equal query/KV physical lengths",
    )
    _check_values(
        (cu_q.diff() <= physical_q.diff()).all(),
        "CSA2 THD valid sequence length exceeds its physical segment",
    )
    max_seqlen = packed_seq_params.max_seqlen_q
    max_seqlen_kv = packed_seq_params.max_seqlen_kv
    for limit in (max_seqlen, max_seqlen_kv):
        if type(limit) is not int or limit < 0:
            raise ValueError("CSA2 THD max_seqlen_q/kv must be non-negative host integers")
        _check_values(
            (physical_q.diff() <= limit).all(),
            "CSA2 THD physical sequence length exceeds max_seqlen_q/kv",
        )
    if max_seqlen != max_seqlen_kv:
        raise ValueError("CSA2 THD self-attention requires equal max_seqlen_q/kv")

    # Snapshot metadata so a shared state cannot silently observe a subsequent
    # microbatch's in-place replacement of static cumulative-length buffers.
    cu_q = cu_q.clone()
    physical_q = physical_q.clone()
    rows = torch.arange(total_tokens, device=cu_q.device, dtype=torch.int64)
    num_sequences = cu_q.numel() - 1
    if num_sequences == 0:
        sequence_ids = torch.full_like(rows, -1)
        positions = torch.zeros_like(rows)
        valid_tokens = torch.zeros(total_tokens, device=cu_q.device, dtype=torch.bool)
    else:
        sequence_ids = batch_of_row(physical_q, total_q=total_tokens)
        positions = rows - physical_q[sequence_ids].to(torch.int64)
        assigned = rows < physical_q[-1]
        valid_tokens = assigned & (positions < cu_q.diff()[sequence_ids])
        sequence_ids = torch.where(assigned, sequence_ids, torch.full_like(sequence_ids, -1))
        positions = torch.where(valid_tokens, positions, torch.zeros_like(positions))

    return CSA2THDLayout(
        total_tokens=total_tokens,
        max_seqlen=max_seqlen,
        cu_seqlens=cu_q,
        cu_seqlens_padded=physical_q,
        sequence_ids=sequence_ids,
        position_ids=positions,
        valid_tokens=valid_tokens,
    )
