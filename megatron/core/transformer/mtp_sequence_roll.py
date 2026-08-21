# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group

_MTP_SEQUENCE_FIELD_FILL_VALUES = {
    "input_ids": 0,
    "position_ids": 0,
    "labels": 0,
    "loss_mask": 0,
    "padding_mask": True,
}


# Shared field state and addressing.


@dataclass(frozen=True)
class MTPSequenceRollField:
    """One immutable field descriptor for absolute sequence-roll access.

    ``sequence_dim`` and ``batch_dim`` describe the logical row layout. All
    remaining dimensions are payload dimensions and are preserved when a shifted
    row is materialized. The context never modifies the source tensor; external
    mutation is detected through the tensor version when available.
    """

    key: str
    tensor: Tensor
    sequence_dim: int
    batch_dim: int
    fill_value: Union[bool, int, float] = 0

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("An MTP sequence-roll field key must be a non-empty string.")
        if not isinstance(self.tensor, Tensor):
            raise TypeError("An MTP sequence-roll field payload must be a tensor.")
        if self.tensor.dim() < 2:
            raise ValueError("An MTP sequence-roll field needs sequence and batch dimensions.")
        sequence_dim = _normalize_mtp_sequence_roll_dim(
            self.sequence_dim, self.tensor.dim(), name=f"{self.key}.sequence_dim"
        )
        batch_dim = _normalize_mtp_sequence_roll_dim(
            self.batch_dim, self.tensor.dim(), name=f"{self.key}.batch_dim"
        )
        if sequence_dim == batch_dim:
            raise ValueError(
                f"MTP sequence-roll field {self.key!r} uses one dimension for sequence and batch."
            )


@dataclass(frozen=True)
class MTPSequenceRollAddress:
    """Canonical source-plus-halo address for one absolute roll offset.

    Roll offset zero denotes the source row, and roll offset ``k`` denotes its
    ``k``-th successor. A legacy cumulative ``roll_depth=d`` therefore consumes
    absolute roll offset ``d + 1``.
    """

    source: Tensor
    halo: Optional[Tensor]
    row_indices: Tensor
    valid_rows: Tensor


@dataclass(frozen=True)
class _PreparedMTPSequenceRollField:
    """Canonical ``[sequence, batch, ...]`` field and layout restoration state."""

    key: str
    source: Tensor
    source_id: int
    source_version: Optional[int]
    inverse_permutation: Tuple[int, ...]
    halo: Optional[Tensor]
    fill_value: Union[bool, int, float]
    fill_tensor: Tensor

    def restore(self, canonical: Tensor) -> Tensor:
        """Restore a canonical sequence-roll tensor to its declared layout."""
        return canonical.permute(self.inverse_permutation)


def _normalize_mtp_sequence_roll_dim(dim: int, ndim: int, *, name: str) -> int:
    """Normalize a field dimension while rejecting out-of-range indexing."""
    if not isinstance(dim, int) or not -ndim <= dim < ndim:
        raise ValueError(f"{name}={dim!r} is invalid for a {ndim}-dimensional tensor.")
    return dim % ndim


def _get_mtp_sequence_roll_source_version(tensor: Tensor) -> Optional[int]:
    """Return the mutation version, or ``None`` for inference tensors."""
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors intentionally do not carry version counters. They
        # may still be prepared, but a context cannot safely prove reuse.
        return None


def _canonicalize_mtp_sequence_roll_field(
    field: MTPSequenceRollField, *, sequence_length: int, batch_size: int, device: torch.device
) -> _PreparedMTPSequenceRollField:
    """Validate one field and expose it as ``[sequence, batch, ...]``."""
    sequence_dim = _normalize_mtp_sequence_roll_dim(
        field.sequence_dim, field.tensor.dim(), name=f"{field.key}.sequence_dim"
    )
    batch_dim = _normalize_mtp_sequence_roll_dim(
        field.batch_dim, field.tensor.dim(), name=f"{field.key}.batch_dim"
    )
    if sequence_dim == batch_dim:
        raise ValueError(
            f"MTP sequence-roll field {field.key!r} uses one dimension for sequence and batch."
        )
    permutation = (sequence_dim, batch_dim) + tuple(
        dim for dim in range(field.tensor.dim()) if dim not in (sequence_dim, batch_dim)
    )
    inverse_permutation = tuple(permutation.index(dim) for dim in range(field.tensor.dim()))
    source = field.tensor.permute(permutation)
    if source.size(0) != sequence_length:
        raise ValueError(
            f"MTP sequence-roll field {field.key!r} has sequence length {source.size(0)}, "
            f"expected {sequence_length}."
        )
    if source.size(1) != batch_size:
        raise ValueError(
            f"MTP sequence-roll field {field.key!r} has batch size {source.size(1)}, "
            f"expected {batch_size}."
        )
    if source.device != device:
        raise ValueError("All fields sharing an MTP sequence-roll context must use its device.")
    return _PreparedMTPSequenceRollField(
        key=field.key,
        source=source,
        source_id=id(field.tensor),
        source_version=_get_mtp_sequence_roll_source_version(field.tensor),
        inverse_permutation=inverse_permutation,
        halo=None,
        fill_value=field.fill_value,
        fill_tensor=field.tensor.new_full((), field.fill_value),
    )


def _validate_mtp_sequence_roll_fields(
    fields: Sequence[MTPSequenceRollField],
    *,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[_PreparedMTPSequenceRollField, ...]:
    """Validate a complete replacement field group before any communication."""
    fields = tuple(fields)
    keys = [field.key for field in fields]
    if len(set(keys)) != len(keys):
        raise ValueError("Each MTP sequence-roll field key may be prepared only once.")
    return tuple(
        _canonicalize_mtp_sequence_roll_field(
            field, sequence_length=sequence_length, batch_size=batch_size, device=device
        )
        for field in fields
    )


def _get_mtp_sequence_roll_field(
    prepared_fields: Tuple[_PreparedMTPSequenceRollField, ...], key: str
) -> _PreparedMTPSequenceRollField:
    """Return one prepared field or raise a precise lookup error."""
    for prepared in prepared_fields:
        if prepared.key == key:
            return prepared
    raise KeyError(f"MTP sequence-roll field {key!r} has not been prepared.")


def _mtp_sequence_roll_field_matches_source(
    prepared: _PreparedMTPSequenceRollField, field: MTPSequenceRollField
) -> bool:
    """Return whether a prepared field still describes this exact tensor source."""
    source_version = _get_mtp_sequence_roll_source_version(field.tensor)
    if (
        prepared.key != field.key
        or prepared.source_id != id(field.tensor)
        or prepared.source_version is None
        or source_version is None
        or prepared.source_version != source_version
        or prepared.fill_value != field.fill_value
    ):
        return False
    try:
        candidate = _canonicalize_mtp_sequence_roll_field(
            field,
            sequence_length=prepared.source.size(0),
            batch_size=prepared.source.size(1),
            device=prepared.source.device,
        )
    except (TypeError, ValueError, RuntimeError):
        return False
    try:
        return (
            candidate.inverse_permutation == prepared.inverse_permutation
            and candidate.source.shape == prepared.source.shape
            and candidate.source.stride() == prepared.source.stride()
            and candidate.source.storage_offset() == prepared.source.storage_offset()
            and candidate.source.data_ptr() == prepared.source.data_ptr()
            and candidate.source.dtype == prepared.source.dtype
            and candidate.source.layout == prepared.source.layout
        )
    except RuntimeError:
        return False


def _materialize_mtp_sequence_roll_rows(
    prepared: _PreparedMTPSequenceRollField, row_indices: Tensor, valid_rows: Tensor
) -> Tensor:
    """Materialize canonical rows through safe indices, then apply boundary fills."""
    source = prepared.source
    sequence_length, batch_size = source.shape[:2]
    payload_shape = source.shape[2:]
    output_shape = tuple(row_indices.shape) + tuple(payload_shape)
    if row_indices.numel() == 0:
        return source.new_empty(output_shape)

    flat_source = source.reshape(sequence_length * batch_size, *payload_shape)
    if prepared.halo is not None:
        halo = prepared.halo
        flat_halo = halo.reshape(-1, *payload_shape)
        flat_source = torch.cat((flat_source, flat_halo), dim=0)
    gathered = flat_source.index_select(0, row_indices.reshape(-1)).reshape(output_shape)
    expanded_valid = valid_rows.reshape(tuple(valid_rows.shape) + (1,) * len(payload_shape))
    return torch.where(expanded_valid, gathered, prepared.fill_tensor)


# Layout-neutral context API.


class MTPSequenceRollContext:
    """Base type for layout-specific state shared by MTP sequence rolls.

    The public MTP call chain passes one context without knowing the physical
    sequence layout. Concrete subtypes own their row geometry, halo transport,
    and prepared field state while exposing the same roll and addressing API.
    """

    @property
    def keys(self) -> Tuple[str, ...]:
        """Return prepared field keys in their declaration order."""
        prepared_fields = getattr(self, "_prepared_fields", ())
        return tuple(prepared.key for prepared in prepared_fields)

    @property
    def max_offset(self) -> int:
        """Return the largest prepared roll offset; offset zero denotes the source row."""
        return getattr(self, "_prepared_max_offset", 0)

    def is_prepared_for_fields(self, fields: Sequence[MTPSequenceRollField]) -> bool:
        """Return whether prepared payloads still correspond to the current sources."""
        fields = tuple(fields)
        if not fields or len({field.key for field in fields}) != len(fields):
            return False
        prepared_fields = getattr(self, "_prepared_fields", ())
        for field in fields:
            try:
                prepared = _get_mtp_sequence_roll_field(prepared_fields, field.key)
            except KeyError:
                return False
            if not _mtp_sequence_roll_field_matches_source(prepared, field):
                return False
        return True

    def prepare_fields(
        self, fields: Sequence[MTPSequenceRollField], *, max_offset: int
    ) -> MTPSequenceRollContext:
        """Prepare absolute shifted rows when this layout supports them.

        The base implementation is the atomic fallback: unsupported layout
        contexts remain unchanged and retain their established roll behavior.
        """
        if not isinstance(max_offset, int) or max_offset <= 0:
            raise ValueError("MTP sequence-roll max_offset must be a positive integer.")
        return self

    def _validate_prepared_offset(self, offset: int) -> None:
        if not isinstance(offset, int):
            raise TypeError("MTP sequence-roll offset must be an integer.")
        if not 0 <= offset <= self.max_offset:
            raise ValueError(
                f"MTP sequence-roll offset {offset} is outside [0, {self.max_offset}]."
            )

    def address(self, key: str, offset: int) -> MTPSequenceRollAddress:
        """Return canonical direct-address metadata for one absolute offset."""
        self._validate_prepared_offset(offset)
        prepared = _get_mtp_sequence_roll_field(getattr(self, "_prepared_fields", ()), key)
        row_indices = getattr(self, "_roll_row_indices", None)
        valid_rows = getattr(self, "_roll_valid_rows", None)
        if row_indices is None or valid_rows is None:
            raise RuntimeError("This MTP sequence-roll context has no prepared shifted rows.")
        return MTPSequenceRollAddress(
            source=prepared.source,
            halo=prepared.halo,
            row_indices=row_indices[offset],
            valid_rows=valid_rows[offset],
        )

    def materialize(self, key: str, offset: int) -> Tensor:
        """Materialize one absolute roll offset in the field's declared layout."""
        self._validate_prepared_offset(offset)
        prepared = _get_mtp_sequence_roll_field(getattr(self, "_prepared_fields", ()), key)
        if offset == 0:
            return prepared.restore(prepared.source)
        address = self.address(key, offset)
        canonical = _materialize_mtp_sequence_roll_rows(
            prepared, address.row_indices, address.valid_rows
        )
        return prepared.restore(canonical)

    def materialize_all(self, key: str) -> Tuple[Tensor, ...]:
        """Materialize absolute offsets ``1..max_offset`` with one indexed gather."""
        prepared = _get_mtp_sequence_roll_field(getattr(self, "_prepared_fields", ()), key)
        if self.max_offset <= 0:
            raise RuntimeError("This MTP sequence-roll context has no prepared shifted rows.")
        row_indices = getattr(self, "_roll_row_indices", None)
        valid_rows = getattr(self, "_roll_valid_rows", None)
        if row_indices is None or valid_rows is None:
            raise RuntimeError("This MTP sequence-roll context has no prepared shifted rows.")
        canonical = _materialize_mtp_sequence_roll_rows(prepared, row_indices[1:], valid_rows[1:])
        return tuple(prepared.restore(canonical[offset]) for offset in range(self.max_offset))

    def prefetch_halos(
        self,
        width: int,
        *,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> MTPSequenceRollContext:
        """Return a context with successor rows prepared for repeated MTP rolls.

        Layout-specific contexts implement the communication and boundary rules.
        Keeping this operation on the context lets model entry points remain layout
        neutral while each concrete subtype uses its own transport strategy.
        Optional fields that are not consumed on this pipeline stage can be omitted.

        Args:
            width: Number of successor rows required by repeated MTP rolls.
            input_ids: Local token IDs used by MTP embedding or RL label derivation.
            position_ids: Local learned-absolute position IDs, when rolled by MTP.
            labels: Local SFT labels consumed by MTP loss.
            loss_mask: Local MTP loss mask.
            padding_mask: Local padding flags rolled by MTP embedding.

        Returns:
            A context containing the prepared layout-specific halo payload.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support halo prefetch.")


# Local CP1 layout.


@dataclass(frozen=True, eq=False)
class LocalRollContext(MTPSequenceRollContext):
    """Absolute sequence-roll geometry for two-dimensional CP1 sequence fields.

    This subtype adds no communication. A bare context preserves the existing CP1
    fallback; consumers that prepare fields use absolute roll offsets. Packed
    boundaries are shared by every batch row; an omitted physical tail is treated
    as one final sequence, matching the established roll helper.
    """

    sequence_length: int
    batch_size: int
    device: torch.device
    packed_cu_seqlens: Optional[Tensor] = None
    _prepared_fields: Tuple[_PreparedMTPSequenceRollField, ...] = ()
    _prepared_max_offset: int = 0
    _roll_row_indices: Optional[Tensor] = None
    _roll_valid_rows: Optional[Tensor] = None

    def prefetch_halos(
        self,
        width: int,
        *,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> MTPSequenceRollContext:
        """Return this local context; CP1 has no remote successor halos."""
        if width <= 0:
            raise ValueError("Local MTP sequence-roll width must be positive.")
        return self

    def prepare_fields(
        self, fields: Sequence[MTPSequenceRollField], *, max_offset: int
    ) -> MTPSequenceRollContext:
        """Return an immutable sibling containing local absolute shifted rows."""
        if not isinstance(max_offset, int) or max_offset <= 0:
            raise ValueError("MTP sequence-roll max_offset must be a positive integer.")
        prepared_fields = _validate_mtp_sequence_roll_fields(
            fields,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            device=self.device,
        )
        if not prepared_fields:
            return LocalRollContext(
                sequence_length=self.sequence_length,
                batch_size=self.batch_size,
                device=self.device,
                packed_cu_seqlens=self.packed_cu_seqlens,
            )
        reuse_geometry = (
            self._prepared_max_offset >= max_offset
            and self._roll_row_indices is not None
            and self._roll_valid_rows is not None
        )
        if reuse_geometry:
            prepared_max_offset = self._prepared_max_offset
            row_indices = self._roll_row_indices
            valid_rows = self._roll_valid_rows
        else:
            prepared_max_offset = max_offset
            row_indices, valid_rows = _build_local_roll_geometry(
                sequence_length=self.sequence_length,
                batch_size=self.batch_size,
                max_offset=prepared_max_offset,
                device=self.device,
                packed_cu_seqlens=self.packed_cu_seqlens,
            )
        return LocalRollContext(
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            device=self.device,
            packed_cu_seqlens=self.packed_cu_seqlens,
            _prepared_fields=prepared_fields,
            _prepared_max_offset=prepared_max_offset,
            _roll_row_indices=row_indices,
            _roll_valid_rows=valid_rows,
        )


def _build_local_roll_geometry(
    *,
    sequence_length: int,
    batch_size: int,
    max_offset: int,
    device: torch.device,
    packed_cu_seqlens: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Build compact int32 row maps for a local CP1 microbatch."""
    rows = torch.arange(sequence_length, device=device, dtype=torch.long)
    offsets = torch.arange(max_offset + 1, device=device, dtype=torch.long).unsqueeze(1)
    targets = rows.unsqueeze(0) + offsets
    valid = targets < sequence_length
    if packed_cu_seqlens is not None and sequence_length > 0:
        cu = packed_cu_seqlens.to(device=device, dtype=torch.long)
        # Appending the physical capacity also models the legal implicit tail;
        # a duplicate final endpoint is harmless with right-biased bucketization.
        boundaries = torch.cat((cu[1:], cu.new_full((1,), sequence_length)))
        sequence_slots = torch.bucketize(rows, boundaries, right=True).clamp(
            max=max(boundaries.numel() - 1, 0)
        )
        sequence_ends = boundaries.index_select(0, sequence_slots)
        valid &= targets < sequence_ends.unsqueeze(0)

    safe_targets = targets.clamp(min=0, max=max(sequence_length - 1, 0))
    batches = torch.arange(batch_size, device=device, dtype=torch.long)
    row_indices = safe_targets.unsqueeze(-1) * batch_size + batches
    row_indices = row_indices.to(torch.int32).contiguous()
    valid_rows = valid.unsqueeze(-1).expand(-1, -1, batch_size).contiguous()
    return row_indices, valid_rows


# Contiguous packed-CP layout.


@dataclass(frozen=True)
class ContiguousPackedSeqRollPlan:
    """Per-microbatch metadata for one-token contiguous-CP packed-sequence rolls.

    A one-token left roll is local except at the end of a CP shard, where the
    replacement value may be the first element owned by the next CP rank. Packed
    sequences add another constraint: positions at a physical packed-sequence
    boundary must receive a field-specific fill value instead of a value from the
    following sequence.

    The CP neighbors and boundary mask depend only on the physical packed layout,
    not on tensor dtype or payload. One context can therefore reuse this plan for
    input IDs, learned-absolute position IDs, padding masks, labels, and loss masks
    throughout a microbatch. When prefetched halos are present, the neighbor ranks
    remain recorded for validation and fallback but no rolling P2P is issued.

    Reuse is valid only for tensors on the recorded device, with the recorded local
    sequence length, using the recorded CP group. Do not cache a plan across
    microbatches unless those layout invariants are guaranteed to remain unchanged.

    Attributes:
        invalid_next: One-dimensional boolean mask over the local sequence axis.
            True means that the corresponding global position has no immediate
            physical successor in the same packed sequence. Repeated local rolls
            propagate fills at internal boundaries; prefetched tail halos are
            separately sanitized for every prediction depth.
        sequence_length: Length of the local contiguous CP shard.
        device: Device on which the boundary mask and compatible payload tensors
            reside.
        cp_group: Effective CP process group for this microbatch. This may be the
            dynamic group injected into PackedSeqParams rather than the model's
            statically configured CP group.
        recv_rank: Global rank of the next contiguous CP shard, used by the P2P
            fallback when no prefetched halo is supplied. None on the last CP rank.
        send_rank: Global rank of the previous contiguous CP shard, used by the P2P
            fallback. None on the first CP rank.
        has_sequences: Whether de-duplicated cumulative sequence lengths describe at
            least one physical packed sequence.
        right_halo_valid_count: Number of successor rows after the local final row
            that remain in the same physical packed sequence. This device scalar
            sanitizes an arbitrary small halo width without rebuilding packed
            metadata.
    """

    invalid_next: Tensor
    sequence_length: int
    device: torch.device
    cp_group: torch.distributed.ProcessGroup
    recv_rank: Optional[int]
    send_rank: Optional[int]
    has_sequences: bool
    right_halo_valid_count: Tensor


def _get_packed_roll_cu_seqlens(packed_seq_params: PackedSeqParams) -> Tensor:
    """Return the physical packed-sequence boundaries used by MTP rolling."""
    cu_seqlens = (
        packed_seq_params.cu_seqlens_q_padded
        if getattr(packed_seq_params, 'cu_seqlens_q_padded', None) is not None
        else packed_seq_params.cu_seqlens_q
    )
    assert cu_seqlens is not None, "Packed sequence parameters must provide cu_seqlens_q."
    return cu_seqlens


def _get_packed_seq_end_indices(
    cu_seqlens: Tensor, device: torch.device, sequence_length: int
) -> Tensor:
    """Return the ends of explicit packed sequences and any implicit tail.

    PackedSeqParams permits the physical tensor to be longer than the final
    cumulative sequence length. In that case, the remaining buffer is an
    implicit tail sequence whose final element must also be filled after the
    full-buffer roll. Duplicate end indices are safe because index_fill_ is
    idempotent.
    """
    sequence_end_indices = cu_seqlens[1:].to(device=device, dtype=torch.long) - 1
    if sequence_length == 0:
        return sequence_end_indices.new_empty((0,))
    implicit_tail_end = sequence_end_indices.new_full((1,), sequence_length - 1)
    return torch.cat((sequence_end_indices, implicit_tail_end))


def _build_contiguous_packed_seq_roll_plan(
    tensor: Tensor, dims: int, cu_seqlens: Tensor, cp_group: torch.distributed.ProcessGroup
) -> ContiguousPackedSeqRollPlan:
    """Build reusable boundary and neighbor metadata for a contiguous-CP shard."""
    assert (
        dims == -1 or dims == tensor.dim() - 1
    ), "Packed sequence roll only supports the last dimension."

    local_seq_len = tensor.size(dims)
    cp_size = cp_group.size()
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)

    cu = cu_seqlens.to(device=tensor.device, dtype=torch.long)
    if cu.numel() > 1:
        # Static packed metadata can repeat its final boundary to pad the number
        # of cu_seqlens entries. Remove duplicates before assigning positions to
        # packed intervals so every retained interval has a nonzero length.
        nonduplicate_boundaries = torch.ones(cu.numel(), device=cu.device, dtype=torch.bool)
        nonduplicate_boundaries[1:] = cu[1:] != cu[:-1]
        cu = cu[nonduplicate_boundaries]

    has_sequences = cu.numel() > 1
    if local_seq_len == 0 or not has_sequences:
        invalid_next = torch.ones(local_seq_len, device=tensor.device, dtype=torch.bool)
        right_halo_valid_count = torch.zeros((), device=tensor.device, dtype=cu.dtype)
    else:
        global_start = local_rank * local_seq_len
        global_positions = global_start + torch.arange(
            local_seq_len, device=tensor.device, dtype=cu.dtype
        )
        seq_idx = torch.bucketize(global_positions, cu[1:], right=True).clamp(max=cu.numel() - 2)
        seq_ends = cu[1:][seq_idx]
        # This deliberately stays true at the local shard's final position when
        # the same packed sequence continues on the next CP rank. A prefetched
        # halo or grouped P2P supplies that successor; only physical ends are masked.
        valid_next = (global_positions < cu[-1]) & (global_positions + 1 < seq_ends)
        invalid_next = ~valid_next

        # Successor rows are valid only until the physical sequence containing
        # this shard's final row ends. Keeping the count as a device scalar avoids
        # a host synchronization and lets halo preparation build any small width.
        local_tail = global_positions[-1]
        right_halo_valid_count = torch.where(
            local_tail < cu[-1],
            (seq_ends[-1] - local_tail - 1).clamp_min(0),
            torch.zeros((), device=tensor.device, dtype=cu.dtype),
        )

    # A left roll receives from the next contiguous shard and sends the first
    # local element to the previous shard. Store global ranks because PyTorch's
    # P2P API interprets peer ranks globally even when a process group is passed.
    return ContiguousPackedSeqRollPlan(
        invalid_next=invalid_next,
        sequence_length=local_seq_len,
        device=tensor.device,
        cp_group=cp_group,
        recv_rank=global_ranks[local_rank + 1] if local_rank < cp_size - 1 else None,
        send_rank=global_ranks[local_rank - 1] if local_rank > 0 else None,
        has_sequences=has_sequences,
        right_halo_valid_count=right_halo_valid_count,
    )


class MTPSequenceRollHalos:
    """Base type for layout-specific successor rows prepared before MTP.

    Halo storage is owned by the corresponding roll context and never travels through
    the model's public forward signature. Concrete layouts can choose how to acquire
    and represent their successor rows without changing GPT, Hybrid, or MTP layers.
    """


@dataclass(frozen=True)
class ContiguousPackedCPRollHalos(MTPSequenceRollHalos):
    """Compact successor rows prefetched across contiguous CP ranks.

    Each tensor stores only a small right halo, never a view of the full packed
    microbatch. Halo index zero is the immediate successor of this CP rank's local
    final row; halo index d is used by the d-th repeated left roll. Values that cross
    a physical packed-sequence boundary are replaced with the field's normal
    boundary fill value once, before MTP starts.

    The explicit optional fields keep this dataclass friendly to CUDA-graph input
    traversal and make the supported payload contract visible. A pipeline stage
    may omit fields it does not own.

    Attributes:
        input_ids: Successor token IDs (the data batch calls this field 'tokens').
        position_ids: Successor learned-absolute position IDs.
        labels: Successor SFT labels.
        loss_mask: Successor loss-mask values.
        padding_mask: Successor padding flags, with True marking padding.
    """

    input_ids: Optional[Tensor] = None
    position_ids: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    loss_mask: Optional[Tensor] = None
    padding_mask: Optional[Tensor] = None

    def __post_init__(self):
        present_halos = [
            halo
            for halo in (
                self.input_ids,
                self.position_ids,
                self.labels,
                self.loss_mask,
                self.padding_mask,
            )
            if halo is not None
        ]
        if not present_halos:
            raise ValueError("A contiguous packed-CP halo payload must contain at least one field.")
        widths = {halo.size(-1) for halo in present_halos}
        if len(widths) != 1:
            raise ValueError("All contiguous packed-CP halo fields must have the same width.")

    @property
    def width(self) -> int:
        """Return the number of prefetched successor rows."""
        for halo in (
            self.input_ids,
            self.position_ids,
            self.labels,
            self.loss_mask,
            self.padding_mask,
        ):
            if halo is not None:
                return halo.size(-1)
        raise AssertionError("ContiguousPackedCPRollHalos requires at least one field.")

    def get(self, sequence_field: str) -> Optional[Tensor]:
        """Return the halo for a canonical MTP sequence field."""
        if sequence_field not in {
            "input_ids",
            "position_ids",
            "labels",
            "loss_mask",
            "padding_mask",
        }:
            raise ValueError(f"Unsupported MTP sequence halo field: {sequence_field}.")
        return getattr(self, sequence_field)


@dataclass(frozen=True, eq=False)
class ContiguousPackedCPRollContext(MTPSequenceRollContext):
    """State reused by all contiguous packed-CP rolls in one microbatch.

    Attributes:
        plan: Boundary and CP-neighbor metadata shared by every field and depth.
        halos: Optional compact successor rows prefetched immediately before MTP.
            None retains grouped P2P as a correctness fallback for direct callers.
    """

    plan: ContiguousPackedSeqRollPlan
    halos: Optional[ContiguousPackedCPRollHalos] = None
    batch_size: int = 1
    _prepared_fields: Tuple[_PreparedMTPSequenceRollField, ...] = ()
    _prepared_max_offset: int = 0
    _roll_row_indices: Optional[Tensor] = None
    _roll_valid_rows: Optional[Tensor] = None

    def prepare_fields(
        self, fields: Sequence[MTPSequenceRollField], *, max_offset: int
    ) -> MTPSequenceRollContext:
        """Prepare one field group with a single one-hop grouped P2P exchange.

        A width larger than the local shard would need more than one neighbor hop;
        that legal but unsupported case returns this context unchanged so callers
        can atomically retain the established roll fallback.
        """
        if not isinstance(max_offset, int) or max_offset <= 0:
            raise ValueError("MTP sequence-roll max_offset must be a positive integer.")
        prepared_fields = _validate_mtp_sequence_roll_fields(
            fields,
            sequence_length=self.plan.sequence_length,
            batch_size=self.batch_size,
            device=self.plan.device,
        )
        if not prepared_fields:
            return ContiguousPackedCPRollContext(
                plan=self.plan, halos=self.halos, batch_size=self.batch_size
            )
        if any(prepared.source.requires_grad for prepared in prepared_fields):
            raise ValueError(
                "Contiguous-CP sequence-roll preparation does not support fields that "
                "require gradients."
            )
        if max_offset > self.plan.sequence_length:
            return self

        reuse_geometry = (
            self._prepared_max_offset >= max_offset
            and self._roll_row_indices is not None
            and self._roll_valid_rows is not None
        )
        if reuse_geometry:
            prepared_max_offset = self._prepared_max_offset
            row_indices = self._roll_row_indices
            valid_rows = self._roll_valid_rows
        else:
            prepared_max_offset = max_offset
            row_indices, valid_rows = _build_contiguous_packed_cp_roll_geometry(
                self.plan, batch_size=self.batch_size, max_offset=prepared_max_offset
            )
        halos = _exchange_contiguous_packed_cp_roll_field_halos(
            prepared_fields, max_offset=prepared_max_offset, plan=self.plan
        )
        prepared_with_halos = tuple(
            _PreparedMTPSequenceRollField(
                key=prepared.key,
                source=prepared.source,
                source_id=prepared.source_id,
                source_version=prepared.source_version,
                inverse_permutation=prepared.inverse_permutation,
                halo=halos[prepared.key],
                fill_value=prepared.fill_value,
                fill_tensor=prepared.fill_tensor,
            )
            for prepared in prepared_fields
        )
        return ContiguousPackedCPRollContext(
            plan=self.plan,
            halos=self.halos,
            batch_size=self.batch_size,
            _prepared_fields=prepared_with_halos,
            _prepared_max_offset=prepared_max_offset,
            _roll_row_indices=row_indices,
            _roll_valid_rows=valid_rows,
        )

    def prefetch_halos(
        self,
        width: int,
        *,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> MTPSequenceRollContext:
        """Prefetch contiguous-CP successor rows in one grouped P2P operation.

        The returned context is immutable and shares this context's roll plan. The
        payload contains only width rows per present field, so it does not retain or
        copy a full packed microbatch. Missing optional fields keep their later
        grouped roll calls on the communication fallback.

        Args:
            width: Number of successor rows required by repeated MTP rolls.
            input_ids: Local token IDs used by MTP embedding or RL label derivation.
            position_ids: Local learned-absolute position IDs, when rolled by MTP.
            labels: Local SFT labels consumed by MTP loss.
            loss_mask: Local MTP loss mask.
            padding_mask: Local padding flags rolled by MTP embedding.

        Returns:
            A new context with compact halos, or this context when prefetch is not
            applicable to the local shard or no fields are present.
        """
        if self.halos is not None:
            raise ValueError("Contiguous packed-CP halos have already been prefetched.")
        if width <= 0:
            raise ValueError("Contiguous packed-CP halo width must be positive.")
        if width > self.plan.sequence_length:
            # One neighbor exchange cannot supply rows spanning multiple CP shards.
            # Keep the established per-roll grouped P2P path for these tiny shards.
            return self

        tensors_by_field = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "padding_mask": padding_mask,
        }
        sequence_fields = []
        present_tensors: List[Tensor] = []
        for field, tensor in tensors_by_field.items():
            if tensor is not None:
                sequence_fields.append(field)
                present_tensors.append(tensor)
        if not sequence_fields:
            return self

        return ContiguousPackedCPRollContext(
            plan=self.plan,
            halos=_prefetch_contiguous_packed_cp_roll_halos(
                tensors=present_tensors,
                sequence_fields=sequence_fields,
                fill_values=[_MTP_SEQUENCE_FIELD_FILL_VALUES[field] for field in sequence_fields],
                width=width,
                plan=self.plan,
            ),
            batch_size=self.batch_size,
            _prepared_fields=self._prepared_fields,
            _prepared_max_offset=self._prepared_max_offset,
            _roll_row_indices=self._roll_row_indices,
            _roll_valid_rows=self._roll_valid_rows,
        )


def _build_contiguous_packed_cp_roll_geometry(
    plan: ContiguousPackedSeqRollPlan, *, batch_size: int, max_offset: int
) -> Tuple[Tensor, Tensor]:
    """Build one-hop absolute row maps from the existing contiguous roll plan."""
    sequence_length = plan.sequence_length
    rows = torch.arange(sequence_length, device=plan.device, dtype=torch.long)
    offsets = torch.arange(max_offset + 1, device=plan.device, dtype=torch.long).unsqueeze(1)
    targets = rows.unsqueeze(0) + offsets

    # A target is valid only if every traversed local edge is valid. The compact
    # right-halo count covers the remaining edges after the local final row.
    invalid_prefix = torch.cat(
        (
            torch.zeros(1, device=plan.device, dtype=torch.int32),
            plan.invalid_next.to(torch.int32).cumsum(0),
        )
    )
    local_edge_end = targets.clamp(max=sequence_length)
    crossed_invalid = invalid_prefix.index_select(0, local_edge_end.reshape(-1)).reshape(
        targets.shape
    ) - invalid_prefix[:-1].unsqueeze(0)
    halo_rows = (targets - sequence_length).clamp_min(0)
    halo_valid = (targets < sequence_length) | (halo_rows < plan.right_halo_valid_count)
    valid = (crossed_invalid == 0) & halo_valid

    addressed_rows = torch.where(targets < sequence_length, targets, sequence_length + halo_rows)
    addressed_rows = addressed_rows.clamp(min=0, max=max(sequence_length + max_offset - 1, 0))
    batches = torch.arange(batch_size, device=plan.device, dtype=torch.long)
    row_indices = addressed_rows.unsqueeze(-1) * batch_size + batches
    return (
        row_indices.to(torch.int32).contiguous(),
        valid.unsqueeze(-1).expand(-1, -1, batch_size).contiguous(),
    )


def _exchange_contiguous_packed_cp_roll_field_halos(
    prepared_fields: Tuple[_PreparedMTPSequenceRollField, ...],
    *,
    max_offset: int,
    plan: ContiguousPackedSeqRollPlan,
) -> dict[str, Tensor]:
    """Acquire all generic field halos in one deterministic grouped P2P."""
    # Sort only the transport order. The returned context preserves declaration
    # order, while peers that supplied the same keys in a different order still
    # match send and receive operations deterministically.
    transport_fields = sorted(prepared_fields, key=lambda prepared: prepared.key)
    halos = {
        prepared.key: prepared.source.new_full(
            (max_offset,) + tuple(prepared.source.shape[1:]), prepared.fill_value
        )
        for prepared in transport_fields
    }
    send_buffers: List[Tensor] = []
    p2p_ops = []
    if plan.has_sequences and plan.recv_rank is not None:
        for prepared in transport_fields:
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    halos[prepared.key],
                    plan.recv_rank,
                    group=plan.cp_group,
                )
            )
    if plan.has_sequences and plan.send_rank is not None:
        for prepared in transport_fields:
            send_buffer = prepared.source.narrow(0, 0, max_offset).contiguous()
            send_buffers.append(send_buffer)
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend, send_buffer, plan.send_rank, group=plan.cp_group
                )
            )

    works = torch.distributed.batch_isend_irecv(p2p_ops) if p2p_ops else []
    for work in works:
        work.wait()
    return halos


def _prefetch_contiguous_packed_cp_roll_halos(
    tensors: List[Tensor],
    sequence_fields: List[str],
    fill_values: List[Union[bool, int, float]],
    width: int,
    plan: ContiguousPackedSeqRollPlan,
) -> ContiguousPackedCPRollHalos:
    """Fetch compact right halos for contiguous packed CP in one grouped P2P.

    Each rank sends the first width rows of every requested field to its
    predecessor and receives the corresponding rows from its successor. Received
    rows are sanitized once using the physical packed boundary that contains this
    rank's local final row. Repeated MTP rolls can then consume halo offsets without
    further communication.

    Args:
        tensors: Local MTP fields sharing the plan's sequence axis.
        sequence_fields: Canonical names corresponding to tensors.
        fill_values: Per-field physical-boundary fill values.
        width: Number of successor rows required by all MTP prediction depths.
        plan: Reusable contiguous packed-CP layout and neighbor metadata.

    Returns:
        Compact, independently allocated successor rows keyed by MTP field.

    Raises:
        ValueError: If the field metadata is inconsistent with the roll plan.
    """
    if width <= 0:
        raise ValueError("Contiguous packed-CP halo width must be positive.")
    if len(tensors) != len(sequence_fields) or len(tensors) != len(fill_values):
        raise ValueError("Each halo tensor must have a canonical field and fill value.")
    if len(set(sequence_fields)) != len(sequence_fields):
        raise ValueError("Each contiguous packed-CP halo field may be prefetched only once.")
    if width > plan.sequence_length:
        raise ValueError(
            f"Contiguous packed-CP halo width {width} exceeds local sequence length "
            f"{plan.sequence_length}."
        )

    halos: List[Tensor] = []
    for tensor, sequence_field, fill_value in zip(tensors, sequence_fields, fill_values):
        if sequence_field not in _MTP_SEQUENCE_FIELD_FILL_VALUES:
            raise ValueError(f"Unsupported MTP sequence halo field: {sequence_field}.")
        expected_fill_value = _MTP_SEQUENCE_FIELD_FILL_VALUES[sequence_field]
        if fill_value != expected_fill_value:
            raise ValueError(
                f"Halo field {sequence_field} requires boundary fill value "
                f"{expected_fill_value!r}, got {fill_value!r}."
            )
        if tensor.device != plan.device:
            raise ValueError("All halo tensors sharing a roll plan must be on the same device.")
        if tensor.size(-1) != plan.sequence_length:
            raise ValueError(
                "All halo tensors sharing a roll plan must have the same sequence length."
            )

        halo_shape = list(tensor.shape)
        halo_shape[-1] = width
        halos.append(tensor.new_full(halo_shape, fill_value))

    # Retain contiguous send slices until every grouped work handle completes.
    send_buffers: List[Tensor] = []
    p2p_ops = []
    if plan.has_sequences and plan.recv_rank is not None:
        for halo in halos:
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv, halo, plan.recv_rank, group=plan.cp_group
                )
            )
    if plan.has_sequences and plan.send_rank is not None:
        for tensor in tensors:
            send_buffer = tensor.narrow(-1, 0, width).contiguous()
            send_buffers.append(send_buffer)
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend, send_buffer, plan.send_rank, group=plan.cp_group
                )
            )

    works = torch.distributed.batch_isend_irecv(p2p_ops) if p2p_ops else []
    for work in works:
        work.wait()

    # Offset d is valid only when the local tail and its (d + 1)-th successor
    # belong to the same physical packed sequence. Broadcasting this small mask
    # sanitizes every field without retaining the full packed metadata.
    invalid_halo = (
        torch.arange(width, device=plan.device, dtype=plan.right_halo_valid_count.dtype)
        >= plan.right_halo_valid_count
    )
    for halo, fill_value in zip(halos, fill_values):
        halo.masked_fill_(invalid_halo, fill_value)

    halo_by_field = dict(zip(sequence_fields, halos))
    return ContiguousPackedCPRollHalos(
        input_ids=halo_by_field.get("input_ids"),
        position_ids=halo_by_field.get("position_ids"),
        labels=halo_by_field.get("labels"),
        loss_mask=halo_by_field.get("loss_mask"),
        padding_mask=halo_by_field.get("padding_mask"),
    )


# Zigzag packed-CP layout.


@dataclass(frozen=True)
class _ZigzagPackedCPRollTransport:
    """Compact prefix rows exchanged with the two physical zigzag neighbors."""

    front_prefix_rows: Tensor
    back_prefix_rows: Tensor
    prefix_valid: Tensor


@dataclass(frozen=True, eq=False)
class ZigzagPackedCPRollContext(MTPSequenceRollContext):
    """One-hop absolute rows for scheduler-certified zigzag packed CP.

    This subtype extends the base sequence-roll interface while preserving the
    established zigzag dispatcher as the fallback. A positive scheduler certificate
    proves that every requested roll offset crosses at most one physical half-chunk.
    If a consumer asks for a wider offset, field preparation leaves this context
    bare and that consumer atomically uses the original cumulative-roll path.
    """

    sequence_length: int
    batch_size: int
    device: torch.device
    cp_group: torch.distributed.ProcessGroup
    packed_cu_seqlens: Tensor
    min_chunk_size: int
    _prepared_fields: Tuple[_PreparedMTPSequenceRollField, ...] = ()
    _prepared_max_offset: int = 0
    _roll_row_indices: Optional[Tensor] = None
    _roll_valid_rows: Optional[Tensor] = None
    _roll_transport: Optional[_ZigzagPackedCPRollTransport] = None

    def prepare_fields(
        self, fields: Sequence[MTPSequenceRollField], *, max_offset: int
    ) -> MTPSequenceRollContext:
        """Prepare one field group after validating its one-hop certificate."""
        if not isinstance(max_offset, int) or max_offset <= 0:
            raise ValueError("MTP sequence-roll max_offset must be a positive integer.")
        prepared_fields = _validate_mtp_sequence_roll_fields(
            fields,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            device=self.device,
        )
        if not prepared_fields:
            return ZigzagPackedCPRollContext(
                sequence_length=self.sequence_length,
                batch_size=self.batch_size,
                device=self.device,
                cp_group=self.cp_group,
                packed_cu_seqlens=self.packed_cu_seqlens,
                min_chunk_size=self.min_chunk_size,
            )
        if any(prepared.source.requires_grad for prepared in prepared_fields):
            raise ValueError(
                "One-hop zigzag packed-CP sequence-roll preparation does not support "
                "fields that require gradients."
            )
        if max_offset > self.min_chunk_size:
            return self

        reuse_geometry = (
            self._prepared_max_offset >= max_offset
            and self._roll_row_indices is not None
            and self._roll_valid_rows is not None
            and self._roll_transport is not None
        )
        prepared_max_offset = self._prepared_max_offset if reuse_geometry else max_offset
        if reuse_geometry:
            row_indices = self._roll_row_indices
            valid_rows = self._roll_valid_rows
            transport = self._roll_transport
            assert transport is not None
        else:
            row_indices, valid_rows, transport = _build_zigzag_packed_cp_roll_geometry(
                sequence_length=self.sequence_length,
                batch_size=self.batch_size,
                max_offset=prepared_max_offset,
                device=self.device,
                cp_group=self.cp_group,
                packed_cu_seqlens=self.packed_cu_seqlens,
            )
        halos = _exchange_zigzag_packed_cp_roll_field_halos(
            prepared_fields, cp_group=self.cp_group, transport=transport
        )
        prepared_with_halos = tuple(
            _PreparedMTPSequenceRollField(
                key=prepared.key,
                source=prepared.source,
                source_id=prepared.source_id,
                source_version=prepared.source_version,
                inverse_permutation=prepared.inverse_permutation,
                halo=halos[prepared.key],
                fill_value=prepared.fill_value,
                fill_tensor=prepared.fill_tensor,
            )
            for prepared in prepared_fields
        )
        return ZigzagPackedCPRollContext(
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            device=self.device,
            cp_group=self.cp_group,
            packed_cu_seqlens=self.packed_cu_seqlens,
            min_chunk_size=self.min_chunk_size,
            _prepared_fields=prepared_with_halos,
            _prepared_max_offset=prepared_max_offset,
            _roll_row_indices=row_indices,
            _roll_valid_rows=valid_rows,
            _roll_transport=transport,
        )


def _build_zigzag_packed_cp_roll_geometry(
    *,
    sequence_length: int,
    batch_size: int,
    max_offset: int,
    device: torch.device,
    cp_group: torch.distributed.ProcessGroup,
    packed_cu_seqlens: Tensor,
) -> Tuple[Tensor, Tensor, _ZigzagPackedCPRollTransport]:
    """Build exact one-hop row maps and transport for a certified zigzag packed-CP microbatch."""
    cp_size = cp_group.size()
    cu = packed_cu_seqlens.to(device=device, dtype=torch.long)
    if cu.ndim != 1 or cu.numel() < 2:
        raise ValueError(
            "Zigzag packed-CP MTP metadata requires one-dimensional cu_seqlens with at least "
            "two entries."
        )
    torch._assert_async(cu[0] == 0, "Zigzag packed-CP MTP cumulative lengths must start at zero.")
    torch._assert_async(
        torch.all(cu[1:] >= cu[:-1]),
        "Zigzag packed-CP MTP cumulative lengths must be nondecreasing.",
    )
    torch._assert_async(
        cu[-1] == sequence_length * cp_size,
        "Zigzag packed-CP MTP metadata must cover the physical buffer exactly.",
    )

    physical_lengths = cu[1:] - cu[:-1]
    divisor = 2 * cp_size
    torch._assert_async(
        torch.all((physical_lengths == 0) | (physical_lengths.remainder(divisor) == 0)),
        "Zigzag packed-CP MTP certificate contradicts physical chunk divisibility.",
    )
    local_cu = torch.div(cu, cp_size, rounding_mode="floor")
    chunk_lengths = torch.div(physical_lengths, divisor, rounding_mode="floor")
    torch._assert_async(
        torch.all((physical_lengths == 0) | (chunk_lengths >= max_offset)),
        "Zigzag packed-CP MTP certificate overstates the minimum physical chunk size.",
    )

    num_sequence_slots = chunk_lengths.numel()
    prefix_offsets = torch.arange(max_offset, device=device, dtype=torch.long).unsqueeze(0)
    prefix_valid = prefix_offsets < chunk_lengths.unsqueeze(1)
    front_prefix_rows = (local_cu[:-1].unsqueeze(1) + prefix_offsets).clamp(
        min=0, max=max(sequence_length - 1, 0)
    )
    back_prefix_rows = (
        local_cu[:-1].unsqueeze(1) + chunk_lengths.unsqueeze(1) + prefix_offsets
    ).clamp(min=0, max=max(sequence_length - 1, 0))

    local_rows = torch.arange(sequence_length, device=device, dtype=torch.long)
    sequence_slots = torch.bucketize(local_rows, local_cu[1:], right=True).clamp(
        max=max(num_sequence_slots - 1, 0)
    )
    sequence_starts = local_cu[:-1].index_select(0, sequence_slots)
    row_chunk_lengths = chunk_lengths.index_select(0, sequence_slots)
    offsets_in_local_sequence = local_rows - sequence_starts
    is_front_chunk = offsets_in_local_sequence < row_chunk_lengths
    offsets_in_chunk = torch.where(
        is_front_chunk, offsets_in_local_sequence, offsets_in_local_sequence - row_chunk_lengths
    )

    offsets = torch.arange(1, max_offset + 1, device=device, dtype=torch.long).unsqueeze(1)
    advanced_offsets = offsets_in_chunk.unsqueeze(0) + offsets
    crosses_chunk = advanced_offsets >= row_chunk_lengths.unsqueeze(0)
    halo_offsets = advanced_offsets - row_chunk_lengths.unsqueeze(0)
    local_targets = local_rows.unsqueeze(0) + offsets

    local_rank = torch.distributed.get_rank(group=cp_group)
    if local_rank == cp_size - 1:
        front_cross_targets = (
            sequence_starts.unsqueeze(0) + row_chunk_lengths.unsqueeze(0) + halo_offsets
        )
    else:
        front_cross_targets = (
            sequence_length + sequence_slots.unsqueeze(0) * max_offset + halo_offsets
        )
    if local_rank == 0:
        back_cross_targets = torch.zeros_like(halo_offsets)
    else:
        back_cross_targets = (
            sequence_length
            + num_sequence_slots * max_offset
            + sequence_slots.unsqueeze(0) * max_offset
            + halo_offsets
        )
    cross_targets = torch.where(
        is_front_chunk.unsqueeze(0), front_cross_targets, back_cross_targets
    )
    addressed_rows = torch.where(crosses_chunk, cross_targets, local_targets)
    valid = ~(crosses_chunk & ~is_front_chunk.unsqueeze(0) & (local_rank == 0))

    batches = torch.arange(batch_size, device=device, dtype=torch.long)
    shifted_indices = addressed_rows.unsqueeze(-1) * batch_size + batches
    identity = local_rows.unsqueeze(-1) * batch_size + batches
    row_indices = torch.cat((identity.unsqueeze(0), shifted_indices), dim=0)
    valid_rows = torch.cat(
        (
            torch.ones((1, sequence_length, batch_size), device=device, dtype=torch.bool),
            valid.unsqueeze(-1).expand(-1, -1, batch_size),
        ),
        dim=0,
    )
    return (
        row_indices.to(torch.int32).contiguous(),
        valid_rows.contiguous(),
        _ZigzagPackedCPRollTransport(
            front_prefix_rows=front_prefix_rows.flatten().contiguous(),
            back_prefix_rows=back_prefix_rows.flatten().contiguous(),
            prefix_valid=prefix_valid.flatten().contiguous(),
        ),
    )


def _exchange_zigzag_packed_cp_roll_field_halos(
    prepared_fields: Tuple[_PreparedMTPSequenceRollField, ...],
    *,
    cp_group: torch.distributed.ProcessGroup,
    transport: _ZigzagPackedCPRollTransport,
) -> dict[str, Tensor]:
    """Exchange both zigzag prefix tapes for every field in one P2P batch."""
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    cp_size = cp_group.size()
    transport_fields = sorted(prepared_fields, key=lambda prepared: prepared.key)
    next_rank = global_ranks[local_rank + 1] if local_rank < cp_size - 1 else None
    previous_rank = global_ranks[local_rank - 1] if local_rank > 0 else None
    # Halo segment zero is the next rank's front prefix; segment one is
    # the previous rank's back prefix. Keep send and receive peers explicit so
    # each route has one unambiguous semantic even when every P2P uses tag zero.
    route_specs = (
        (transport.front_prefix_rows, previous_rank, next_rank),
        (transport.back_prefix_rows, next_rank, previous_rank),
    )
    segments = {prepared.key: [] for prepared in transport_fields}
    send_buffers = []
    p2p_ops = []
    for send_rows, send_rank, recv_rank in route_specs:
        for prepared in transport_fields:
            valid_shape = (-1,) + (1,) * (prepared.source.dim() - 1)
            send = prepared.source.index_select(0, send_rows)
            send = send.masked_fill(
                ~transport.prefix_valid.view(valid_shape), prepared.fill_value
            ).contiguous()
            send_buffers.append(send)
            recv = prepared.source.new_full(send.shape, prepared.fill_value)
            segments[prepared.key].append(recv)
            if recv_rank is not None:
                p2p_ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.irecv, recv, recv_rank, group=cp_group
                    )
                )
            if send_rank is not None:
                p2p_ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.isend, send, send_rank, group=cp_group
                    )
                )

    works = torch.distributed.batch_isend_irecv(p2p_ops) if p2p_ops else []
    for work in works:
        work.wait()
    return {
        key: torch.cat(field_segments, dim=0).contiguous()
        for key, field_segments in segments.items()
    }


# Public preparation entry points.


def prepare_mtp_sequence_roll_context(
    tensor: Tensor | None,
    cp_group: torch.distributed.ProcessGroup | None,
    packed_seq_params: PackedSeqParams | None,
    dims: int = -1,
) -> MTPSequenceRollContext | None:
    """Prepare layout-specific state shared by MTP rolls in one microbatch.

    The public boundary is layout neutral. Two-dimensional CP1 inputs receive a
    local context capable of optional absolute addressing while retaining the legacy
    roll helper as fallback. Contiguous packed CP retains its existing plan and halo
    behavior while also supporting explicit one-hop sequence-roll preparation. A
    scheduler-certified zigzag packed-CP microbatch receives the same absolute-row
    interface with two compact neighbor tapes. Uncertified and unsupported layouts
    return None and keep their established paths.

    Args:
        tensor: Reference payload that establishes local sequence length and device.
        cp_group: Effective context-parallel process group for the microbatch.
        packed_seq_params: Physical packed-sequence layout metadata.
        dims: Sequence dimension; packed rolling supports only the final dimension.

    Returns:
        A layout-specific roll context, or None when no prepared state is needed.
    """
    if tensor is None:
        return None

    cp_group = resolve_cp_group(cp_group, packed_seq_params)
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size <= 1:
        if tensor.dim() != 2:
            return None
        sequence_dim = _normalize_mtp_sequence_roll_dim(dims, tensor.dim(), name="dims")
        # Packed rolling supports only the final dimension. Do not create a
        # context that the established dispatcher would reject.
        if packed_seq_params is not None and sequence_dim != tensor.dim() - 1:
            return None
        batch_dim = 1 - sequence_dim
        return LocalRollContext(
            sequence_length=tensor.size(sequence_dim),
            batch_size=tensor.size(batch_dim),
            device=tensor.device,
            packed_cu_seqlens=(
                _get_packed_roll_cu_seqlens(packed_seq_params)
                if packed_seq_params is not None
                else None
            ),
        )

    if packed_seq_params is None or cp_group is None:
        return None

    cp_partition_mode = getattr(packed_seq_params, 'cp_partition_mode', 'zigzag')
    if cp_partition_mode == "zigzag":
        min_chunk_size = packed_seq_params.zigzag_cp_min_chunk_size
        if min_chunk_size is None or min_chunk_size == 0:
            # Zero is a scheduler-produced "not certifiable" result, not
            # malformed packed metadata. Preserve the established zigzag roll
            # path exactly as for an absent certificate.
            return None
        if min_chunk_size < 0:
            raise ValueError("Zigzag packed-CP MTP scheduler certificate cannot be negative.")
        if (
            packed_seq_params.local_cp_size is not None
            and int(packed_seq_params.local_cp_size) != cp_size
        ):
            raise ValueError(
                "Zigzag packed-CP MTP metadata local_cp_size must match "
                "the effective CP group size."
            )
        if (
            packed_seq_params.qkv_format != "thd"
            or tensor.dim() != 2
            or tensor.size(0) != 1
            or (
                _normalize_mtp_sequence_roll_dim(dims, tensor.dim(), name="dims")
                != tensor.dim() - 1
            )
        ):
            return None
        return ZigzagPackedCPRollContext(
            sequence_length=tensor.size(-1),
            batch_size=tensor.size(0),
            device=tensor.device,
            cp_group=cp_group,
            packed_cu_seqlens=_get_packed_roll_cu_seqlens(packed_seq_params),
            min_chunk_size=min_chunk_size,
        )
    if cp_partition_mode != 'contiguous':
        return None

    return ContiguousPackedCPRollContext(
        plan=_build_contiguous_packed_seq_roll_plan(
            tensor, dims, _get_packed_roll_cu_seqlens(packed_seq_params), cp_group
        ),
        batch_size=tensor.size(0) if tensor.dim() == 2 else 1,
    )


def prepare_mtp_sequence_roll_fields(
    sequence_roll_context: Optional[MTPSequenceRollContext],
    fields: Sequence[MTPSequenceRollField],
    *,
    max_offset: int,
) -> Optional[MTPSequenceRollContext]:
    """Return one atomically prepared consumer field group when supported.

    A context already containing the complete group for these exact source
    tensors is reused. Otherwise the fields are late-bound as one replacement
    group. Unsupported layouts return ``None`` so the caller takes its complete
    legacy cumulative-roll path; a consumer never mixes addressed and rolled fields.
    """
    if sequence_roll_context is None:
        return None
    if (
        sequence_roll_context.max_offset >= max_offset
        and sequence_roll_context.is_prepared_for_fields(fields)
    ):
        return sequence_roll_context
    prepared_context = sequence_roll_context.prepare_fields(fields, max_offset=max_offset)
    if prepared_context.max_offset >= max_offset and prepared_context.is_prepared_for_fields(
        fields
    ):
        return prepared_context
    return None


# Cumulative-roll compatibility path.


def roll_tensor(
    tensors: List[Tensor],
    shifts: int = -1,
    dims: int = -1,
    cp_group: torch.distributed.ProcessGroup | None = None,
    packed_seq_params: PackedSeqParams | None = None,
    fill_values: List[Union[bool, int, float]] | None = None,
    roll_context: MTPSequenceRollContext | None = None,
    sequence_fields: List[str] | None = None,
    roll_depth: int = 0,
) -> List[Tensor]:
    """Roll one or more MTP tensor fields along the sequence dimension.

    All tensors in one call share the same physical sequence layout. Grouping them
    allows contiguous packed CP to share metadata and use one P2P batch. When a
    contiguous context owns prefetched halos, the same dispatcher replaces each
    local tail from the requested field/depth and issues no rolling P2P.

    Args:
        tensors: Tensor fields to roll together.
        shifts: Shift along the sequence dimension.
        dims: Sequence dimension.
        cp_group: Effective context-parallel process group.
        packed_seq_params: Packed-sequence layout metadata, when applicable.
        fill_values: Per-field values written at physical sequence boundaries.
        roll_context: Layout-specific state prepared for this microbatch. None
            retains the regular dispatcher and communication fallback.
        sequence_fields: Canonical source fields corresponding to tensors. These
            identify prefetched halo payloads and are required only when the
            context contains halos.
        roll_depth: Zero-based repeated-roll depth. Depth zero consumes the
            immediate successor; depth d consumes halo offset d.

    Returns:
        Rolled tensors in the same order as tensors.

    Raises:
        ValueError: If field counts, depth, or roll-context arguments are inconsistent.
    """
    if not tensors:
        return []
    if roll_depth < 0:
        raise ValueError("roll_depth must be non-negative.")
    if fill_values is None:
        fill_values = [0] * len(tensors)
    if len(tensors) != len(fill_values):
        raise ValueError("Each tensor must have a corresponding roll fill value.")
    if sequence_fields is not None and len(tensors) != len(sequence_fields):
        raise ValueError("Each tensor must have a corresponding canonical sequence field.")

    if packed_seq_params is None:
        if roll_context is not None and not isinstance(roll_context, LocalRollContext):
            raise ValueError("A prepared sequence-roll context requires packed parameters.")
        return _roll_tensors_unpacked(tensors, shifts, dims, cp_group, fill_values)

    return _roll_tensors_packed_seq(
        tensors,
        shifts,
        dims,
        packed_seq_params,
        cp_group,
        fill_values,
        roll_context,
        sequence_fields,
        roll_depth,
    )


def _roll_tensors_unpacked(
    tensors: List[Tensor],
    shifts: int,
    dims: int,
    cp_group: Optional[torch.distributed.ProcessGroup],
    fill_values: List[Union[bool, int, float]],
) -> List[Tensor]:
    """Roll unpacked tensors for CP1 or the standard zigzag CP layout."""
    if cp_group is None or cp_group.size() == 1:
        rolled_tensors = [torch.roll(tensor, shifts=shifts, dims=dims) for tensor in tensors]
        for rolled_tensor, fill_value in zip(rolled_tensors, fill_values):
            rolled_tensor.select(dims, shifts).fill_(fill_value)
        return rolled_tensors

    return [
        _roll_tensor_unpacked_zigzag_cp(tensor, shifts, dims, cp_group, fill_value=fill_value)
        for tensor, fill_value in zip(tensors, fill_values)
    ]


def _roll_tensor_unpacked_zigzag_cp(tensor, shifts, dims, cp_group, fill_value=0):
    """Roll one unpacked tensor in the standard two-chunk zigzag CP layout."""
    # This matches the batch splitting logic in get_batch_on_this_cp_rank().
    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

    # Prepare tensors for communication between CP ranks
    # Each CP rank needs to send boundary elements to adjacent ranks
    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
        empty_tensor = torch.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=torch.cuda.current_device(),
        )
        tensor_recv_list.append(empty_tensor)

    # Get the global rank of next and prev process in the cp group
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    # Start send and recv ops
    ops = []
    if local_rank != 0:
        req_send_first_part = torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank)
        ops.append(req_send_first_part)
        req_recv_second_part = torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank)
        ops.append(req_recv_second_part)
    else:
        tensor_recv_list[1] = fill_value
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank)
        ops.append(req_recv_first_part)
        req_send_second_part = torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank)
        ops.append(req_send_second_part)
    else:
        # For the last CP rank, the removed elements of second part go into the first part
        tensor_recv_list[0] = tensor_send_list[1]

    # Wait for all communication operations to complete
    for op in ops:
        op.wait()

    # Splicing: Replace boundary elements with received elements from adjacent ranks
    # This ensures proper sequence continuity across CP boundaries
    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    # Concatenate the processed chunks back into a single tensor
    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)

    return rolled_tensor


def _roll_tensors_packed_seq(
    tensors: List[Tensor],
    shifts: int,
    dims: int,
    packed_seq_params: PackedSeqParams,
    cp_group: Optional[torch.distributed.ProcessGroup],
    fill_values: List[Union[bool, int, float]],
    roll_context: Optional[MTPSequenceRollContext],
    sequence_fields: Optional[List[str]],
    roll_depth: int,
) -> List[Tensor]:
    """Dispatch packed tensors to CP1, zigzag CP, or contiguous CP rolling."""
    for tensor in tensors:
        assert (
            dims == -1 or dims == tensor.dim() - 1
        ), "Packed sequence roll only supports the last dimension."
    assert shifts == -1, "Packed sequence roll only supports a single-token left shift."

    # Prefer padded cumulative seqlens because CP's local THD layout uses the
    # padded physical boundaries. Unpadded boundaries index the wrong local
    # chunks when sequence lengths are not already divisible by 2 * cp_size.
    cu_seqlens = _get_packed_roll_cu_seqlens(packed_seq_params)

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        if roll_context is not None and not isinstance(roll_context, LocalRollContext):
            raise ValueError("A prepared sequence-roll context cannot be used for packed CP1.")
        reference_tensor = tensors[0]
        sequence_end_indices = _get_packed_seq_end_indices(
            cu_seqlens, reference_tensor.device, reference_tensor.size(dims)
        )
        for tensor in tensors:
            if tensor.device != reference_tensor.device:
                raise ValueError("All packed CP1 tensors must be on the same device.")
            if tensor.size(dims) != reference_tensor.size(dims):
                raise ValueError("All packed CP1 tensors must have the same sequence length.")
        return [
            _roll_tensor_packed_seq_cp1(
                tensor, shifts, dims, sequence_end_indices, fill_value=fill_value
            )
            for tensor, fill_value in zip(tensors, fill_values)
        ]

    cp_partition_mode = getattr(packed_seq_params, 'cp_partition_mode', 'zigzag')
    if cp_partition_mode == 'zigzag':
        if roll_context is not None and not isinstance(roll_context, ZigzagPackedCPRollContext):
            raise ValueError(
                "A prepared sequence-roll context is not supported for zigzag packed CP."
            )
        return [
            _roll_tensor_packed_seq_zigzag_cp(
                tensor, shifts, dims, cu_seqlens, cp_group, fill_value=fill_value
            )
            for tensor, fill_value in zip(tensors, fill_values)
        ]
    if cp_partition_mode == 'contiguous':
        contiguous_roll_halos = None
        if roll_context is None:
            contiguous_roll_plan = _build_contiguous_packed_seq_roll_plan(
                tensors[0], dims, cu_seqlens, cp_group
            )
        elif isinstance(roll_context, ContiguousPackedCPRollContext):
            contiguous_roll_plan = roll_context.plan
            contiguous_roll_halos = roll_context.halos
        else:
            raise ValueError(
                "The prepared sequence-roll context does not support contiguous packed CP."
            )
        return _roll_tensors_packed_seq_contiguous_cp(
            tensors,
            dims,
            fill_values,
            contiguous_roll_plan,
            contiguous_roll_halos,
            sequence_fields,
            roll_depth,
        )
    raise ValueError(f"Unsupported packed sequence CP partition mode: {cp_partition_mode}")


def _roll_tensor_packed_seq_cp1(tensor, shifts, dims, sequence_end_indices, fill_value=0):
    """Roll one CP1 packed tensor and fill every physical sequence end."""
    # A full-buffer left roll is equivalent to rolling each packed sequence
    # independently once the values that crossed sequence boundaries are filled.
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.index_fill_(dims, sequence_end_indices, fill_value)
    return rolled_tensor


def _roll_tensor_packed_seq_zigzag_cp(tensor, shifts, dims, cu_seqlens, cp_group, fill_value=0):
    """Roll a zigzag-CP THD shard without crossing packed sequence boundaries."""
    cp_size = cp_group.size()
    rolled_tensor = tensor.clone()

    # CP enabled: each rank owns two chunks per sequence (front and mirrored tail).
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    # Iterate over each sequence individually
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]

        # the idx has been multiplied by cp_size, need to divide it by cp_size to get the local idx
        local_start_idx = start_idx // cp_size
        local_end_idx = end_idx // cp_size

        # Skip empty sequences - this can happen when a sequence is very short and
        # after dividing by cp_size, the local slice has zero length
        local_seq_len = local_end_idx - local_start_idx
        if local_seq_len == 0:
            continue

        tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()

        # The following code is very similar as the code in roll_tensor function
        local_chunks = tensor_slice.chunk(2, dim=dims)
        rolled_chunks = [torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks]

        tensor_send_list = []
        tensor_recv_list = []
        for chunk in rolled_chunks:
            # Skip empty chunks that can occur when the sequence slice is very small
            if chunk.size(dims) == 0:
                tensor_send_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                tensor_recv_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                continue
            boundary = chunk.select(dims, shifts).contiguous().clone()
            tensor_send_list.append(boundary)
            tensor_recv_list.append(torch.empty_like(boundary))

        ops = []
        if local_rank != 0:
            ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank))
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank))
        else:
            tensor_recv_list[1].fill_(fill_value)

        if local_rank != cp_size - 1:
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank))
            ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank))
        else:
            tensor_recv_list[0].copy_(tensor_send_list[1])

        for op in ops:
            op.wait()

        index = [slice(None)] * rolled_chunks[0].dim()
        index[dims] = shifts
        for chunk, recv in zip(rolled_chunks, tensor_recv_list):
            # Skip empty chunks
            if chunk.size(dims) == 0:
                continue
            chunk[tuple(index)] = recv

        seq_result = torch.cat(rolled_chunks, dim=dims)

        # update the rolled tensor
        rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

    return rolled_tensor


def _roll_tensors_packed_seq_contiguous_cp(
    tensors: List[Tensor],
    dims: int,
    fill_values: List[Union[bool, int, float]],
    contiguous_roll_plan: ContiguousPackedSeqRollPlan,
    contiguous_roll_halos: Optional[ContiguousPackedCPRollHalos] = None,
    sequence_fields: Optional[List[str]] = None,
    roll_depth: int = 0,
) -> List[Tensor]:
    """Roll contiguous packed-CP tensors from halos or one grouped P2P exchange.

    A prefetched halo is used only when every field in this grouped call is present.
    Otherwise the entire call takes the grouped P2P fallback, keeping all CP ranks
    on the same communication branch while supporting optional model inputs.
    """
    assert len(tensors) == len(fill_values)
    if not tensors:
        return []

    for tensor in tensors:
        assert (
            dims == -1 or dims == tensor.dim() - 1
        ), "Packed sequence roll only supports the last dimension."
        if tensor.size(dims) != contiguous_roll_plan.sequence_length:
            raise ValueError(
                "All tensors sharing a packed-sequence roll plan must have the same "
                "sequence length."
            )
        if tensor.device != contiguous_roll_plan.device:
            raise ValueError(
                "All tensors sharing a packed-sequence roll plan must be on the same device."
            )

    if contiguous_roll_plan.sequence_length == 0:
        return [torch.roll(tensor, shifts=-1, dims=dims) for tensor in tensors]

    if not contiguous_roll_plan.has_sequences:
        rolled_tensors = [torch.roll(tensor, shifts=-1, dims=dims) for tensor in tensors]
        for rolled_tensor, fill_value in zip(rolled_tensors, fill_values):
            rolled_tensor.fill_(fill_value)
        return rolled_tensors

    halo_tail_values = None
    if contiguous_roll_halos is not None and sequence_fields is not None:
        if len(sequence_fields) != len(tensors):
            raise ValueError("Each rolled tensor must have a canonical sequence field.")

        requested_halos = [contiguous_roll_halos.get(field) for field in sequence_fields]
        if all(halo is not None for halo in requested_halos):
            if roll_depth >= contiguous_roll_halos.width:
                raise ValueError(
                    f"roll_depth={roll_depth} exceeds the prefetched halo width "
                    f"{contiguous_roll_halos.width}."
                )

            halo_tail_values = []
            for tensor, sequence_field, fill_value, halo in zip(
                tensors, sequence_fields, fill_values, requested_halos
            ):
                assert halo is not None
                expected_fill_value = _MTP_SEQUENCE_FIELD_FILL_VALUES[sequence_field]
                if fill_value != expected_fill_value:
                    raise ValueError(
                        f"Halo field {sequence_field} requires boundary fill value "
                        f"{expected_fill_value!r}, got {fill_value!r}."
                    )
                if halo.device != tensor.device:
                    raise ValueError(
                        f"Halo field {sequence_field} and its tensor must be on the same device."
                    )
                if halo.dtype != tensor.dtype:
                    raise ValueError(
                        f"Halo field {sequence_field} and its tensor must have the same dtype."
                    )
                if halo.dim() != tensor.dim() or halo.shape[:-1] != tensor.shape[:-1]:
                    raise ValueError(
                        f"Halo field {sequence_field} must match its tensor's leading dimensions."
                    )
                halo_tail_values.append(halo.select(dims, roll_depth))

    if halo_tail_values is not None:
        rolled_tensors = [torch.roll(tensor, shifts=-1, dims=dims) for tensor in tensors]
        for rolled_tensor, halo_tail, fill_value in zip(
            rolled_tensors, halo_tail_values, fill_values
        ):
            rolled_tensor.select(dims, -1).copy_(halo_tail)
            # Internal physical sequence ends are handled by the shared immediate
            # boundary mask. Tail values for deeper rolls were sanitized before
            # slicing because their validity depends on the requested depth.
            rolled_tensor[..., contiguous_roll_plan.invalid_next] = fill_value
        return rolled_tensors

    recv_buffers: List[Optional[Tensor]] = [None] * len(tensors)
    # Keep contiguous send buffers alive until every grouped work handle completes.
    send_buffers: List[Tensor] = []
    p2p_ops = []

    if contiguous_roll_plan.recv_rank is not None:
        # After a left roll, each local tail consumes the first element from the
        # next contiguous CP shard.
        for index, tensor in enumerate(tensors):
            recv_buffer = torch.empty_like(tensor.select(dims, 0))
            recv_buffers[index] = recv_buffer
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    recv_buffer,
                    contiguous_roll_plan.recv_rank,
                    group=contiguous_roll_plan.cp_group,
                )
            )
    if contiguous_roll_plan.send_rank is not None:
        # This rank's first element becomes the previous shard's local tail.
        for tensor in tensors:
            send_buffer = tensor.select(dims, 0).contiguous()
            send_buffers.append(send_buffer)
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    send_buffer,
                    contiguous_roll_plan.send_rank,
                    group=contiguous_roll_plan.cp_group,
                )
            )

    works = torch.distributed.batch_isend_irecv(p2p_ops) if p2p_ops else []
    rolled_tensors = [torch.roll(tensor, shifts=-1, dims=dims) for tensor in tensors]
    for work in works:
        work.wait()

    for rolled_tensor, recv_buffer, fill_value in zip(rolled_tensors, recv_buffers, fill_values):
        if recv_buffer is not None:
            rolled_tensor.select(dims, -1).copy_(recv_buffer)
        # Apply the shared boundary mask after installing the adjacent value so a
        # physical packed-sequence end always wins over a cross-rank successor.
        rolled_tensor[..., contiguous_roll_plan.invalid_next] = fill_value

    return rolled_tensors
