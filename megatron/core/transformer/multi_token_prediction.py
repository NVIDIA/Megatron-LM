# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping, replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_mtp_prefix import mtp_e2e_prefix_objective
from megatron.core.fusions.fused_mtp_tv import vocab_parallel_tv_distance
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group
from megatron.core.pipeline_parallel.utils import is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.inference_layers import (
    inference_all_gather_from_tensor_model_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.hyper_connection import learned_output_contract
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    get_pg_rank,
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
    make_viewless_tensor,
)

if TYPE_CHECKING:
    from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]


_MTP_SEQUENCE_FIELD_FILL_VALUES = {
    "input_ids": 0,
    "position_ids": 0,
    "labels": 0,
    "loss_mask": 0,
    "padding_mask": True,
}


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
            or _normalize_mtp_sequence_roll_dim(dims, tensor.dim(), name="dims") != tensor.dim() - 1
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


if HAVE_TE:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
else:
    TESpecProvider = None

from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout


def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict,
    word_emb_weight: Tensor,
    word_emb_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the embedding of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        word_emb_weight (Tensor): weight of the word embedding.
        word_emb_weight_key (str): key of the word embedding in the sharded state dict.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel group
        dp_cp_group (torch.distributed.ProcessGroup): The dp-cp comm group

    Returns: None, acts in-place
    """
    mtp_word_emb_replica_id = (
        1,  # copy of embedding in pre processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert word_emb_weight_key in sharded_state_dict
    del sharded_state_dict[word_emb_weight_key]
    sharded_state_dict[word_emb_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=word_emb_weight,
        key=word_emb_weight_key,
        replica_id=mtp_word_emb_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def tie_output_layer_state_dict(
    sharded_state_dict: ShardedStateDict,
    output_layer_weight: Tensor,
    output_layer_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the output layer of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        output_layer_weight (Tensor): weight of the output layer.
        output_layer_weight_key (str): key of the output layer in the sharded state dict.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel group
        dp_cp_group (torch.distributed.ProcessGroup): The dp-cp comm group

    Returns: None, acts in-place
    """
    mtp_output_layer_replica_id = (
        1,  # copy of output layer in post processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert output_layer_weight_key in sharded_state_dict
    del sharded_state_dict[output_layer_weight_key]
    sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=output_layer_weight,
        key=output_layer_weight_key,
        replica_id=mtp_output_layer_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


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


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses and acceptance rates."""

    tracker = {}

    @staticmethod
    def configure_acceptance_collection(enabled: bool):
        """Configure whether every microbatch collects acceptance statistics.

        Training enables collection only when TensorBoard or W&B consumes the
        metric. Standalone callers that do not configure a policy retain the
        legacy collect-every-call behavior.
        """
        tracker = MTPLossLoggingHelper.tracker
        tracker["acceptance_collection_enabled"] = enabled
        if not enabled:
            MTPLossLoggingHelper._clean_acceptance_in_tracker()

    @staticmethod
    def should_collect_acceptance() -> bool:
        """Return whether this microbatch should collect acceptance statistics."""
        return MTPLossLoggingHelper.tracker.get("acceptance_collection_enabled", True)

    @staticmethod
    def _save_acceptance_counts(
        correct: torch.Tensor,
        total: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: Optional[torch.distributed.ProcessGroup],
        avg_group: Optional[torch.distributed.ProcessGroup],
    ):
        """Accumulate packed correct/total counters for one later collective."""
        tracker = MTPLossLoggingHelper.tracker
        expected_shape = (2, num_layers)
        if (
            "acceptance_counts" not in tracker
            or tracker["acceptance_counts"].shape != expected_shape
        ):
            tracker["acceptance_counts"] = torch.zeros(
                expected_shape, device=torch.cuda.current_device()
            )
            # Keep the historical tracker names as views for compatibility; both
            # counters are nevertheless reduced together as one packed tensor.
            tracker["correct_values"] = tracker["acceptance_counts"][0]
            tracker["total_values"] = tracker["acceptance_counts"][1]

        tracker["acceptance_counts"][0, layer_number] += correct.detach()
        tracker["acceptance_counts"][1, layer_number] += total.detach()
        tracker["acceptance_reduce_group"] = reduce_group
        tracker["acceptance_avg_group"] = avg_group

    @staticmethod
    def save_metrics_to_tracker(
        loss: torch.Tensor,
        correct: torch.Tensor,
        total: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """Save normalized MTP loss and acceptance counts for logging.

        This compatibility path is used by tests and callers that already
        computed a normalized per-layer loss. Dynamic-CP code uses
        ``save_loss_to_tracker`` to normalize each local contribution safely.
        """
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker:
            tracker["loss_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["loss_values"][layer_number] += loss.detach()
        MTPLossLoggingHelper._save_acceptance_counts(
            correct, total, layer_number, num_layers, reduce_group, avg_group
        )
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def save_loss_to_tracker(
        loss_sum: torch.Tensor,
        num_tokens: torch.Tensor,
        layer_number: int,
        num_layers: int,
        correct: Optional[torch.Tensor] = None,
        total: Optional[torch.Tensor] = None,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
        calculate_per_token_loss: bool = False,
    ):
        """Accumulate MTP loss for logging.

        With per-token loss normalization, store raw loss sums and token
        counts so logging reports ``sum(loss) / sum(tokens)``. Otherwise,
        preserve the legacy microbatch-normalized logging contract.

        Args:
            loss_sum (torch.Tensor): Sum of per-element losses on this rank.
            num_tokens (torch.Tensor): Number of valid tokens on this rank.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            correct (Optional[torch.Tensor]): Number of correct MTP predictions.
            total (Optional[torch.Tensor]): Total number of MTP predictions.
            reduce_group (torch.distributed.ProcessGroup): Group for summing losses.
            avg_group (torch.distributed.ProcessGroup): Group for averaging losses.
            calculate_per_token_loss (bool): Whether the main training path uses
                per-token loss normalization.
        """
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "loss_sums" not in tracker:
            tracker["loss_sums"] = torch.zeros(num_layers, device=torch.cuda.current_device())
            tracker["calculate_per_token_loss"] = calculate_per_token_loss
            if calculate_per_token_loss:
                tracker["num_tokens"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        else:
            assert (
                tracker.get("calculate_per_token_loss") == calculate_per_token_loss
            ), "MTP loss tracker cannot mix per-token and microbatch-normalized logging modes."

        if calculate_per_token_loss:
            tracker["loss_sums"][layer_number] += loss_sum.detach()
            tracker["num_tokens"][layer_number] += num_tokens.detach()
        else:
            loss_sum = (loss_sum * (num_tokens > 0).to(loss_sum.dtype)) / num_tokens.clamp(min=1)
            tracker["loss_sums"][layer_number] += loss_sum.detach()
        if correct is not None and total is not None:
            MTPLossLoggingHelper._save_acceptance_counts(
                correct, total, layer_number, num_layers, reduce_group, avg_group
            )
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def _clean_acceptance_in_tracker():
        """Clear per-step acceptance counters while retaining replay metadata."""
        tracker = MTPLossLoggingHelper.tracker
        if "acceptance_counts" in tracker:
            tracker["acceptance_counts"].zero_()

    @staticmethod
    def clean_metrics_in_tracker():
        """Clear the mtp metrics."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" in tracker:
            tracker["loss_values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None
        MTPLossLoggingHelper._clean_acceptance_in_tracker()

    @staticmethod
    def reduce_metrics_in_tracker():
        """Collect and reduce the MTP metrics across ranks."""
        tracker = MTPLossLoggingHelper.tracker

        if "loss_values" in tracker:
            loss_values = tracker["loss_values"]
            if tracker.get("reduce_group") is not None:
                torch.distributed.all_reduce(loss_values, group=tracker.get("reduce_group"))
            if tracker.get("avg_group") is not None:
                torch.distributed.all_reduce(
                    loss_values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG
                )

        if (
            not tracker.get("acceptance_collection_enabled", True)
            or "acceptance_counts" not in tracker
        ):
            return

        acceptance_counts = tracker["acceptance_counts"]
        if tracker.get("acceptance_reduce_group") is not None:
            torch.distributed.all_reduce(
                acceptance_counts, group=tracker["acceptance_reduce_group"]
            )
        if tracker.get("acceptance_avg_group") is not None:
            torch.distributed.all_reduce(
                acceptance_counts,
                group=tracker["acceptance_avg_group"],
                op=torch.distributed.ReduceOp.SUM,
            )

    @staticmethod
    def clean_loss_in_tracker():
        """Clear per-step MTP loss and acceptance counters."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_sums" in tracker:
            tracker["loss_sums"].zero_()
        if "num_tokens" in tracker:
            tracker["num_tokens"].zero_()
        if "values" in tracker:
            tracker["values"].zero_()
        if "loss_values" in tracker:
            tracker["loss_values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None
        MTPLossLoggingHelper._clean_acceptance_in_tracker()

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the mtp losses across ranks.

        Per-token mode reduces raw numerators and denominators before dividing.
        Legacy mode reduces already-normalized microbatch losses.
        """
        tracker = MTPLossLoggingHelper.tracker
        if "loss_sums" not in tracker:
            return
        if tracker.get("calculate_per_token_loss", False):
            packed = torch.cat([tracker["loss_sums"], tracker["num_tokens"]])
            for group_key in ('reduce_group', 'avg_group'):
                group = tracker.get(group_key)
                if group is not None:
                    torch.distributed.all_reduce(packed, group=group)
            loss_sums, num_tokens = packed.chunk(2)
            tracker["values"] = loss_sums / num_tokens.clamp(min=1)
            return

        values = tracker["loss_sums"]
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker['reduce_group'])
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        tracker["values"] = values

    @staticmethod
    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track per-step MTP loss and acceptance metrics."""
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        MTPLossLoggingHelper.reduce_metrics_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "loss_sums" in tracker and "values" in tracker:
            mtp_losses = tracker["values"] * loss_scale
        elif "loss_values" in tracker:
            mtp_losses = tracker["loss_values"] * loss_scale
        else:
            return

        has_acceptance = (
            tracker.get("acceptance_collection_enabled", True) and "acceptance_counts" in tracker
        )
        if has_acceptance:
            mtp_corrects, mtp_totals = tracker["acceptance_counts"]

            # Process-local logging state; cumulative rates intentionally
            # reset after restart/resume.
            if (
                "cumulative_acceptance_counts" not in tracker
                or tracker["cumulative_acceptance_counts"].shape
                != tracker["acceptance_counts"].shape
            ):
                tracker["cumulative_acceptance_counts"] = torch.zeros_like(
                    tracker["acceptance_counts"]
                )

            tracker["cumulative_acceptance_counts"] += tracker["acceptance_counts"]
            mtp_cumulative_corrects, mtp_cumulative_totals = tracker["cumulative_acceptance_counts"]

        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            loss_name = f"mtp_{i+1} loss"
            loss = mtp_losses[i]

            if total_loss_dict is not None:
                total_loss_dict[loss_name] = (
                    total_loss_dict.get(loss_name, torch.zeros_like(loss)) + loss
                )

            if writer is not None:
                writer.add_scalar(loss_name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{loss_name}": loss}, iteration)

            if has_acceptance:
                step_acc_name = f"mtp_{i+1}_acceptance_rate"
                cum_acc_name = f"mtp_{i+1}_cumulative_acceptance_rate"
                # Empty masks can leave no valid MTP positions, so clamp denominators.
                step_rate = (mtp_corrects[i] / torch.clamp(mtp_totals[i], min=1)) * 100.0
                cum_rate = (
                    mtp_cumulative_corrects[i] / torch.clamp(mtp_cumulative_totals[i], min=1)
                ) * 100.0

                if writer is not None:
                    writer.add_scalar(step_acc_name, step_rate, iteration)
                    writer.add_scalar(cum_acc_name, cum_rate, iteration)
                if wandb_writer is not None:
                    wandb_writer.log({f"{step_acc_name}": step_rate}, iteration)
                    wandb_writer.log({f"{cum_acc_name}": cum_rate}, iteration)

        MTPLossLoggingHelper.clean_loss_in_tracker()


def _mtp_logits_are_vocab_sharded(
    output_layer: Callable, runtime_gather_output: Optional[bool]
) -> bool:
    """Return whether MTP logits are still vocab-sharded across tensor-parallel ranks."""
    if runtime_gather_output is not None:
        return not runtime_gather_output
    return not getattr(output_layer, "gather_output", False)


def _vocab_parallel_argmax(
    vocab_parallel_logits: Tensor, tp_group: torch.distributed.ProcessGroup, tp_size: int
) -> Tensor:
    """Return global argmax ids from logits sharded across the vocab dimension."""
    vocab_shard_size = vocab_parallel_logits.size(-1)
    local_max_vals, local_argmax = vocab_parallel_logits.max(dim=-1)  # [s, b], [s, b]

    gathered_max_vals = [torch.empty_like(local_max_vals) for _ in range(tp_size)]
    gathered_argmax = [torch.empty_like(local_argmax) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered_max_vals, local_max_vals, group=tp_group)
    torch.distributed.all_gather(gathered_argmax, local_argmax, group=tp_group)

    stacked_max_vals = torch.stack(gathered_max_vals, dim=0)
    stacked_argmax = torch.stack(gathered_argmax, dim=0)
    winning_rank = stacked_max_vals.argmax(dim=0)  # [s, b]
    winning_local_argmax = torch.gather(stacked_argmax, 0, winning_rank.unsqueeze(0)).squeeze(
        0
    )  # [s, b]
    return winning_rank * vocab_shard_size + winning_local_argmax  # [s, b]


def _compute_mtp_acceptance_counts(
    mtp_logits: Tensor,
    mtp_labels: Tensor,
    loss_mask: Tensor,
    output_layer: Callable,
    runtime_gather_output: Optional[bool],
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[Tensor, Tensor]:
    """Compute MTP acceptance correct/total counts."""
    with torch.no_grad():
        logits_are_vocab_sharded = _mtp_logits_are_vocab_sharded(
            output_layer, runtime_gather_output
        )
        if (
            tp_group is None
            and logits_are_vocab_sharded
            and parallel_state.is_initialized()
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        ):
            raise ValueError(
                "tp_group must be provided when computing MTP acceptance counts "
                "from vocab-sharded logits under tensor model parallelism."
            )
        tp_size = torch.distributed.get_world_size(group=tp_group) if tp_group is not None else 1

        # Apply TP rank offsets only when logits are vocab-sharded; gathered logits already
        # contain global vocab ids in their last dimension.
        if tp_group is not None and tp_size > 1 and logits_are_vocab_sharded:
            preds = _vocab_parallel_argmax(mtp_logits, tp_group, tp_size)
        else:
            preds = torch.argmax(mtp_logits, dim=-1)  # [s, b]

        labels_match = mtp_labels.transpose(0, 1).contiguous()  # [b, s] => [s, b]
        mask_match = loss_mask.transpose(0, 1).contiguous()  # [b, s] => [s, b]
        valid_positions = mask_match.bool()
        correct = ((preds == labels_match) & valid_positions).sum().float()
        total = valid_positions.sum().float()

    return correct, total


@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        hnorm: Specification or instance of the hidden states normalization to be applied.
        enorm: Specification or instance of the embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        mtp_model_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer or mamba block to be applied.
    """

    enorm: LayerNormBuilder
    hnorm: LayerNormBuilder
    # TODO(nschank): Move this back below transformer_layer once eh_proj and transformer_layer have
    # their defaults removed.
    layer_norm: LayerNormBuilder

    eh_proj: Union[ModuleSpec, type] = None
    e_proj: Union[ModuleSpec, type] = None
    h_proj: Union[ModuleSpec, type] = None
    mtp_model_layer: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(
    mtp_model_layer_spec: ModuleSpec,
    use_transformer_engine: bool,
    enable_hyper_connections: bool = False,
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    return get_mtp_layer_spec_for_backend(
        mtp_model_layer_spec,
        backend=TESpecProvider() if use_transformer_engine else LocalSpecProvider(),
        enable_hyper_connections=enable_hyper_connections,
    )


def get_mtp_layer_spec_for_backend(
    mtp_model_layer_spec: ModuleSpec,
    backend: BackendSpecProvider,
    enable_hyper_connections: bool = False,
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with modules from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl = backend.layer_norm()

    submodules_kwargs = dict(
        enorm=layer_norm_impl,
        hnorm=layer_norm_impl,
        mtp_model_layer=mtp_model_layer_spec,
        layer_norm=layer_norm_impl,
    )
    if enable_hyper_connections:
        submodules_kwargs["e_proj"] = column_parallel_linear_impl
        submodules_kwargs["h_proj"] = column_parallel_linear_impl
    else:
        submodules_kwargs["eh_proj"] = column_parallel_linear_impl

    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(**submodules_kwargs),
    )
    return mtp_layer_spec


def mtp_on_this_rank(
    layout: PipelineParallelLayerLayout = None,
    mtp_num_layers: Optional[int] = None,
    ignore_virtual: Optional[bool] = True,
    vp_stage: Optional[int] = None,
) -> bool:
    """
    Check if there is MTP on the current rank.

    Behavior:
        - If a custom pipeline model parallel layout is provided:
            - If virtual pipeline parallelism is enabled (and `ignore_virtual` is False), checks
              whether any MTP layers are present on this (pp_rank, vp_stage) pair.
            - Otherwise, checks all virtual pipeline ranks of the current pipeline rank. Returns
              True if any virtual sub-rank includes at least one MTP layer.
        - If no custom layout is provided, assumes all MTP layers (if any) are placed on the last
          pipeline stage. The function returns True only on the last pipeline stage.
    """
    if layout is not None and hasattr(layout, "pipeline_model_parallel_layout"):
        # Backward-compat: some callers pass a TransformerConfig as the first
        # positional argument instead of (layout, mtp_num_layers). Unpack it.
        _config = layout
        layout = _config.pipeline_model_parallel_layout
        mtp_num_layers = _config.mtp_num_layers
    mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if layout is not None:
        # with custom PP layout, we support put MTP layers on any pipeline stage
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout.layout[pp_rank][vp_stage].count(LayerType.mtp)
            mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout.layout[pp_rank])):
                num_layers_to_build = layout.layout[pp_rank][vpp_rank].count(LayerType.mtp)
                if num_layers_to_build > 0:
                    mtp_on_this_rank = True
                    break
    else:
        # without custom PP layout, we only support put all of MTP layers on the last pipeline stage
        if mtp_num_layers is not None:
            mtp_on_this_rank = parallel_state.is_pipeline_last_stage(
                ignore_virtual=ignore_virtual, vp_stage=vp_stage
            )
        else:
            mtp_on_this_rank = False
    return mtp_on_this_rank


def get_mtp_ranks(pp_ranks: List[int], config: TransformerConfig) -> List[int]:
    """Get the ranks of the MTP layers."""
    mtp_ranks = set()
    if config.mtp_num_layers is None:
        return []
    if config.pipeline_model_parallel_layout is None:
        return [pp_ranks[-1]]
    layout = config.pipeline_model_parallel_layout.layout
    for pp_rank in range(len(layout)):
        for vpp_rank in range(len(layout[pp_rank])):
            num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
            if num_layers_to_build:
                mtp_ranks.add(pp_ranks[pp_rank])
    return list(mtp_ranks)


def get_mtp_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Get the offset of the MTP layer."""
    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.mtp, vp_stage=vp_stage
            )
        else:
            offset = 0
    else:
        offset = 0
    return offset


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the number of MTP layers to build."""
    if config.pipeline_model_parallel_layout is not None:
        # If we have a custom PP layout, get the number of mtp layers in the layout array.
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support put all of MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
            num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            num_layers_to_build = 0
    return num_layers_to_build


class MTPLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            mtp_loss (torch.Tensor): The mtp loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for mtp loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        (mtp_loss,) = ctx.saved_tensors
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = torch.ones_like(mtp_loss) * mtp_loss_backward_scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MTPLossAutoScaler.main_loss_backward_scale = scale


def _prepare_mtp_sequence_roll_fields(
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


def _process_mtp_e2e_tv_loss(
    hidden_states_list: tuple[Tensor, ...],
    loss_mask: Tensor,
    output_layer: Callable,
    output_weight: Tensor,
    runtime_gather_output: Optional[bool],
    is_training: bool,
    config: TransformerConfig,
    cp_group: Optional[torch.distributed.ProcessGroup],
    tp_group: Optional[torch.distributed.ProcessGroup],
    dp_cp_group: Optional[torch.distributed.ProcessGroup],
    packed_seq_params: Optional[PackedSeqParams],
    scale_logits_fn: Optional[Callable[[Tensor], Tensor]],
    sequence_roll_context: Optional[MTPSequenceRollContext],
    derived_labels_from_input_ids: bool,
) -> Tensor:
    """Apply the end-to-end TV objective to MTP draft hidden states.

    The frozen backbone is projected once. For depth ``i``, its target logits are
    shifted by ``i + 1`` positions to align with the MTP draft distribution for
    the same future token. Only start positions with a complete chain across all
    configured MTP depths contribute to the objective.
    """
    hidden_states = hidden_states_list[0]
    logits_are_vocab_sharded = _mtp_logits_are_vocab_sharded(output_layer, runtime_gather_output)

    # Project the frozen backbone once, then align each target distribution by
    # rolling logits. Re-projecting one shifted hidden tensor per MTP depth would
    # add a full vocabulary GEMM for every draft step.
    with torch.no_grad():
        target_logits, _ = output_layer(
            hidden_states.detach(),
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        if scale_logits_fn is not None:
            target_logits = scale_logits_fn(target_logits)
    num_depths = config.mtp_num_layers
    max_offset = num_depths + int(derived_labels_from_input_ids)
    tv_roll_context = _prepare_mtp_sequence_roll_fields(
        sequence_roll_context,
        (
            MTPSequenceRollField("target_logits", target_logits, 0, 1, 0),
            MTPSequenceRollField("loss_mask", loss_mask, -1, 0, 0),
        ),
        max_offset=max_offset,
    )
    use_prepared_roll_rows = tv_roll_context is not None

    # The legacy path keeps vocabulary before sequence because roll_tensor
    # operates on the final dimension. The addressed path leaves target logits
    # in their native [sequence, batch, vocab] layout and reads them directly.
    if use_prepared_roll_rows:
        assert tv_roll_context is not None
        aligned_loss_masks = tv_roll_context.materialize_all("loss_mask")
        current_loss_mask = loss_mask
        original_loss_mask = aligned_loss_masks[0] if derived_labels_from_input_ids else loss_mask
    else:
        target_logits = target_logits.permute(1, 2, 0)
        current_loss_mask = loss_mask
        if derived_labels_from_input_ids:
            current_loss_mask = roll_tensor(
                [current_loss_mask],
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                roll_context=sequence_roll_context,
                sequence_fields=["loss_mask"],
                roll_depth=0,
            )[0]
        original_loss_mask = current_loss_mask

    original_num_tokens = original_loss_mask.sum() if config.calculate_per_token_loss else None
    chain_valid = torch.ones_like(loss_mask, dtype=torch.bool)
    tv_distances = []

    for mtp_layer_number in range(num_depths):
        if use_prepared_roll_rows:
            assert tv_roll_context is not None
            target_address = tv_roll_context.address("target_logits", mtp_layer_number + 1)
            current_loss_mask = aligned_loss_masks[
                mtp_layer_number + int(derived_labels_from_input_ids)
            ]
        else:
            target_logits = roll_tensor(
                [target_logits],
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                fill_values=[0],
            )[0]
            current_loss_mask = roll_tensor(
                [current_loss_mask],
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                roll_context=sequence_roll_context,
                sequence_fields=["loss_mask"],
                roll_depth=mtp_layer_number + int(derived_labels_from_input_ids),
            )[0]
        chain_valid &= current_loss_mask.bool()

        draft_logits, _ = output_layer(
            hidden_states_list[mtp_layer_number + 1],
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        if scale_logits_fn is not None:
            draft_logits = scale_logits_fn(draft_logits)
        if use_prepared_roll_rows:
            tv_distances.append(
                vocab_parallel_tv_distance(
                    draft_logits,
                    target_address.source,
                    tp_group=tp_group,
                    logits_are_vocab_sharded=logits_are_vocab_sharded,
                    target_row_indices=target_address.row_indices,
                    target_valid_rows=target_address.valid_rows,
                    target_halo_logits=target_address.halo,
                )
            )
        else:
            aligned_target_logits = target_logits.permute(2, 0, 1)
            tv_distances.append(
                vocab_parallel_tv_distance(
                    draft_logits,
                    aligned_target_logits,
                    tp_group=tp_group,
                    logits_are_vocab_sharded=logits_are_vocab_sharded,
                )
            )

    tv_distances_tensor = torch.stack(tv_distances, dim=0)
    per_step_acceptance = 1.0 - tv_distances_tensor
    e2e_tv_loss, prefix_losses = mtp_e2e_prefix_objective(per_step_acceptance)

    chain_mask = chain_valid.transpose(0, 1).to(e2e_tv_loss.dtype)
    num_tokens = chain_mask.sum()
    masked_e2e_tv_loss = e2e_tv_loss * chain_mask

    if is_training:
        collect_acceptance = MTPLossLoggingHelper.should_collect_acceptance()
        avg_group = dp_cp_group
        for mtp_layer_number in range(config.mtp_num_layers):
            correct = None
            total = None
            if collect_acceptance:
                correct = torch.sum(per_step_acceptance[mtp_layer_number] * chain_mask)
                total = num_tokens
            MTPLossLoggingHelper.save_loss_to_tracker(
                torch.sum(prefix_losses[mtp_layer_number] * chain_mask),
                num_tokens,
                mtp_layer_number,
                config.mtp_num_layers,
                correct=correct,
                total=total,
                avg_group=avg_group,
                calculate_per_token_loss=config.calculate_per_token_loss,
            )

    assert config.mtp_loss_scaling_factor is not None
    if config.calculate_per_token_loss:
        assert original_num_tokens is not None
        normalized_loss = (
            config.mtp_loss_scaling_factor
            * masked_e2e_tv_loss
            * (original_num_tokens / num_tokens.clamp(min=1))
        )
    else:
        normalized_loss = (
            config.mtp_loss_scaling_factor * masked_e2e_tv_loss / num_tokens.clamp(min=1)
        )
    return MTPLossAutoScaler.apply(hidden_states, normalized_loss)


def process_mtp_loss(
    hidden_states: Tensor,
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    output_layer: Callable,
    output_weight: Optional[Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable,
    config: TransformerConfig,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    scale_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
    input_ids: Optional[Tensor] = None,
    sequence_roll_context: Optional[MTPSequenceRollContext] = None,
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """Process Multi-Token Prediction (MTP) loss computation.

    This is a standalone function that handles MTP loss computation. It's used on the
    post_process rank to split concatenated hidden states and compute MTP losses.

    Args:
        hidden_states (Tensor): Hidden states tensor (concatenated with MTP outputs).
        labels (Tensor): Ground truth labels.
        loss_mask (Optional[Tensor]): Mask for loss computation. If None, uses all ones.
        output_layer (Callable): Output layer method to compute logits.
        output_weight (Optional[Tensor]): Optional output weight for shared embeddings.
        runtime_gather_output (Optional[bool]): Whether to gather output at runtime.
        is_training (bool): Whether the model is in training mode.
        compute_language_model_loss (Callable): Method to compute language model loss.
        config (TransformerConfig): Model configuration containing mtp_num_layers etc.
        cp_group (Optional[ProcessGroup]): Context parallelism process group.
        tp_group (Optional[ProcessGroup]): Tensor parallelism process group.
        packed_seq_params (Optional[PackedSeqParams]): Packed sequence parameters.
        scale_logits_fn (Optional[Callable[[Tensor], Tensor]]): Optional function to
            scale logits before loss computation (e.g., MuP output scaling).
        input_ids (Optional[Tensor]): Input token IDs. Used to derive labels when
            ``labels`` is None (e.g. RL training), by rolling left to match the SFT
            label convention (``label[i] = input_id[i + 1]``). Ignored when ``labels``
            is provided.
        sequence_roll_context (Optional[MTPSequenceRollContext]): Layout-specific
            metadata shared by MTP rolls in this microbatch.
        dp_cp_group (Optional[ProcessGroup]): Data and context parallelism process group.

    Returns:
        Tensor: Updated hidden states after MTP loss processing (first chunk only).
    """
    hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
    hidden_states = hidden_states_list[0]

    derived_labels_from_input_ids = labels is None
    if derived_labels_from_input_ids:
        if input_ids is None:
            return hidden_states
        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids)
    elif loss_mask is None:
        loss_mask = torch.ones_like(labels)

    assert loss_mask is not None

    if config.mtp_detach_heads:
        if output_weight is not None:
            output_weight = output_weight.detach()
        else:
            output_weight = output_layer.weight.detach()

    if config.mtp_loss_type == "e2e_tv":
        assert output_weight is not None
        return _process_mtp_e2e_tv_loss(
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            output_layer=output_layer,
            output_weight=output_weight,
            runtime_gather_output=runtime_gather_output,
            is_training=is_training,
            config=config,
            cp_group=cp_group,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
            packed_seq_params=packed_seq_params,
            scale_logits_fn=scale_logits_fn,
            sequence_roll_context=sequence_roll_context,
            derived_labels_from_input_ids=derived_labels_from_input_ids,
        )

    label_source = input_ids if derived_labels_from_input_ids else labels
    assert label_source is not None
    label_key = "input_ids" if derived_labels_from_input_ids else "labels"
    max_offset = config.mtp_num_layers + int(derived_labels_from_input_ids)
    loss_roll_context = _prepare_mtp_sequence_roll_fields(
        sequence_roll_context,
        (
            MTPSequenceRollField(label_key, label_source, -1, 0, 0),
            MTPSequenceRollField("loss_mask", loss_mask, -1, 0, 0),
        ),
        max_offset=max_offset,
    )
    use_prepared_roll_rows = loss_roll_context is not None
    if use_prepared_roll_rows:
        assert loss_roll_context is not None
        aligned_labels = loss_roll_context.materialize_all(label_key)
        aligned_loss_masks = loss_roll_context.materialize_all("loss_mask")
        original_loss_mask = aligned_loss_masks[0] if derived_labels_from_input_ids else loss_mask
        mtp_labels = label_source
    else:
        # Preserve the original cumulative-roll path exactly. RL first derives
        # SFT-format labels/mask at offset one; every MTP depth then rolls once.
        mtp_labels = label_source
        if derived_labels_from_input_ids:
            mtp_labels, loss_mask = roll_tensor(
                [mtp_labels, loss_mask],
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                roll_context=sequence_roll_context,
                sequence_fields=["input_ids", "loss_mask"],
                roll_depth=0,
            )
        original_loss_mask = loss_mask

    original_num_tokens = original_loss_mask.sum() if config.calculate_per_token_loss else None

    fuse_linear_cross_entropy = (
        config.cross_entropy_loss_fusion and config.cross_entropy_fusion_impl == "linear"
    )
    for mtp_layer_number in range(config.mtp_num_layers):
        if use_prepared_roll_rows:
            offset_index = mtp_layer_number + int(derived_labels_from_input_ids)
            mtp_labels = aligned_labels[offset_index]
            loss_mask = aligned_loss_masks[offset_index]
        else:
            mtp_labels, loss_mask = roll_tensor(
                [mtp_labels, loss_mask],
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                roll_context=sequence_roll_context,
                sequence_fields=[label_key, "loss_mask"],
                roll_depth=mtp_layer_number + int(derived_labels_from_input_ids),
            )
        num_tokens = loss_mask.sum()
        if fuse_linear_cross_entropy:
            mtp_loss = output_layer(
                hidden_states_list[mtp_layer_number + 1],
                weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                output_cross_entropy_loss=True,
                labels=mtp_labels,
            )
            # Fused linear cross entropy does not materialize logits, so MTP
            # acceptance counts cannot be computed for this layer.
            mtp_logits = None
        else:
            mtp_logits, _ = output_layer(
                hidden_states_list[mtp_layer_number + 1],
                weight=output_weight,
                runtime_gather_output=runtime_gather_output,
            )
            if scale_logits_fn is not None:
                mtp_logits = scale_logits_fn(mtp_logits)
            mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
        mtp_loss = loss_mask * mtp_loss

        if is_training:
            correct = None
            total = None
            if mtp_logits is not None and MTPLossLoggingHelper.should_collect_acceptance():
                correct, total = _compute_mtp_acceptance_counts(
                    mtp_logits, mtp_labels, loss_mask, output_layer, runtime_gather_output, tp_group
                )

            MTPLossLoggingHelper.save_loss_to_tracker(
                torch.sum(mtp_loss),
                num_tokens,
                mtp_layer_number,
                config.mtp_num_layers,
                correct=correct,
                total=total,
                avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                calculate_per_token_loss=config.calculate_per_token_loss,
            )
        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if config.calculate_per_token_loss:
            # When calculate_per_token_loss is enabled, finalize_model_grads will
            # divide all gradients by total_num_tokens (from main loss).
            # However, MTP has fewer valid tokens due to rolling. To ensure correct
            # per-token gradient weighting, we normalize by the rolled token count
            # and re-scale by the original token count.
            # Avoid division by zero
            assert original_num_tokens is not None
            num_tokens_safe = torch.clamp(num_tokens, min=1)
            mtp_loss_normalized = (
                mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe)
            )
            hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
        else:
            safe_num_tokens = num_tokens.clamp(min=1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens
            )

    return hidden_states


class MultiTokenPredictionLayer(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    For more information, refer to DeepSeek-V3 Technical Report
    https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        # For hybrid path - pattern and submodules to build inner layers directly
        mtp_layer_pattern: Optional[str] = None,
        hybrid_submodules: Optional[HybridStackSubmodules] = None,
        mamba_submodules: Optional[HybridStackSubmodules] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config=config)
        if mamba_submodules is not None:
            if hybrid_submodules is not None:
                raise ValueError(
                    "Cannot specify both hybrid_submodules and mamba_submodules. "
                    "mamba_submodules has been deprecated; use hybrid_submodules instead."
                )
            warnings.warn(
                "mamba_submodules has been deprecated. Use hybrid_submodules instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hybrid_submodules = mamba_submodules
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number + get_mtp_layer_offset(self.config, vp_stage)
        self.vp_stage = vp_stage
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp if pg_collection is not None else None
        self.mtp_layer_pattern = mtp_layer_pattern

        # Validate attention mask type if using transformer-based inner layers
        if self.submodules.mtp_model_layer is not None and hasattr(
            self.submodules.mtp_model_layer, 'submodules'
        ):
            from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
            from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

            layer_submodules = None
            if isinstance(self.submodules.mtp_model_layer.submodules, HybridStackSubmodules):
                attention_layer_spec = self.submodules.mtp_model_layer.submodules.attention_layer
                if hasattr(attention_layer_spec, 'submodules'):
                    assert isinstance(attention_layer_spec.submodules, TransformerLayerSubmodules)
                    layer_submodules = attention_layer_spec.submodules
            elif isinstance(self.submodules.mtp_model_layer.submodules, TransformerLayerSubmodules):
                layer_submodules = self.submodules.mtp_model_layer.submodules
            else:
                raise ValueError(
                    "Unsupported mtp_model_layer submodules type for attention mask validation."
                )
            if layer_submodules:
                self_attention_spec = layer_submodules.self_attention
                attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
                assert attn_mask_type in SUPPORTED_ATTN_MASK, (
                    f"Multi-Token Prediction (MTP) is not yet supported with "
                    f"{attn_mask_type} attention mask type. "
                    f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
                )

        self.mhc_enabled = self.config.enable_hyper_connections

        self.enorm = self.submodules.enorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = self.submodules.hnorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        if self.mhc_enabled:
            # mHC mode: separate e_proj and h_proj, operating per-stream.
            # e_proj: [h] -> [h], applied to embedding then broadcast across streams.
            # h_proj: [h] -> [h], applied per-stream on hidden states.
            self.e_proj = build_module(
                self.submodules.e_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="mtp_e_proj",
                tp_group=pg_collection.tp if pg_collection is not None else None,
                name=(name + ".e_proj") if name is not None else None,
            )
            self.h_proj = build_module(
                self.submodules.h_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="mtp_h_proj",
                tp_group=pg_collection.tp if pg_collection is not None else None,
                name=(name + ".h_proj") if name is not None else None,
            )
            self.eh_proj = None
        else:
            # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
            # of the i-th token's hidden states and the (i + K)-th token's decoder input,
            # so the input's shape is [s, b, 2*h].
            # The output will be send to the following transformer layer,
            # so the output's shape should be [s, b, h].
            self.eh_proj = build_module(
                self.submodules.eh_proj,
                self.config.hidden_size * 2,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="mtp_eh_proj",
                tp_group=pg_collection.tp if pg_collection is not None else None,
                name=(name + ".eh_proj") if name is not None else None,
            )
            self.e_proj = None
            self.h_proj = None

        # Build inner layers: two possible paths
        # 1. Hybrid path: use HybridStack for hybrid pattern support
        # 2. GPT path: single TransformerLayer
        if mtp_layer_pattern is not None and hybrid_submodules is not None:
            from megatron.core.models.hybrid.hybrid_block import HybridStack
            from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers

            self.mtp_model_layer = HybridStack(
                config=self.config,
                submodules=hybrid_submodules,
                layer_type_list=validate_segment_layers(mtp_layer_pattern),
                pp_layer_offset=0,
                pre_process=True,  # Always receives input from eh_proj
                post_layer_norm=False,  # MTP has its own final_layernorm
                post_process=True,  # MTP layer is self-contained
                pg_collection=pg_collection,
                is_mtp_layer=True,
                mtp_layer_number=self.layer_number,
                name=(name + ".mtp_model_layer") if name is not None else None,
            )
        elif self.config.mtp_num_layers is not None:
            # GPT path: Uses the transformer block spec for MTP layer
            # MTP inner layers use their own layer numbering (self.layer_number = 1, 2, etc.)
            # rather than continuing from decoder layer numbers. This is consistent with the
            # Mamba path and ensures proper aux loss tracking in router.py.
            self.mtp_model_layer = build_module(
                self.submodules.mtp_model_layer,
                config=self.config,
                vp_stage=self.vp_stage,
                layer_number=self.layer_number,
                is_mtp_layer=True,
                pg_collection=pg_collection,
                name=(name + ".mtp_model_layer") if name is not None else None,
            )

        self.final_layernorm = self.submodules.layer_norm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        if self.mhc_enabled:
            hc_mult = self.config.num_residual_streams
            hc_dim = self.config.hidden_size * hc_mult
            self.hc_head_fn = mark_keep_in_fp32(nn.Parameter(torch.randn(hc_mult, hc_dim)))
            self.hc_head_base = mark_keep_in_fp32(nn.Parameter(torch.zeros(hc_mult)))
            self.hc_head_scale = mark_keep_in_fp32(nn.Parameter(torch.ones(1)))
            nn.init.xavier_uniform_(self.hc_head_fn)
            if self.config.sequence_parallel:
                setattr(self.hc_head_fn, 'sequence_parallel', True)
                setattr(self.hc_head_base, 'sequence_parallel', True)
                setattr(self.hc_head_scale, 'sequence_parallel', True)

        self.offload_context = nullcontext()

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[torch.Tensor] = None,
        sequence_roll_context: Optional[MTPSequenceRollContext] = None,
        roll_depth: int = 0,
        _inputs_pre_aligned: bool = False,
    ):
        """Roll MTP inputs once and compute the next-depth token embeddings.

        Args:
            input_ids: Current-depth token IDs.
            position_ids: Position IDs corresponding to input_ids.
            embedding: Parent model's embedding module.
            hidden_states: Current-depth hidden states in [s, b, h] layout.
            packed_seq_params: Packed sequence layout metadata.
            padding_mask: Optional padding mask rolled with input IDs.
            sequence_roll_context: Layout-specific state shared by MTP rolls
                in this microbatch.
            roll_depth: Zero-based prediction depth selecting the prefetched
                successor row for this repeated roll.
            _inputs_pre_aligned: Internal block-to-layer contract indicating that
                input, learned position, and padding fields already represent the
                absolute shifted row for this depth.
        """
        if not _inputs_pre_aligned:
            cp_group = resolve_cp_group(self.cp_group, packed_seq_params)

            tensors_to_roll = [input_ids]
            fill_values = [0]
            sequence_fields = ["input_ids"]
            roll_position_ids = getattr(embedding, 'add_position_embedding', True)
            if roll_position_ids:
                tensors_to_roll.append(position_ids)
                fill_values.append(0)
                sequence_fields.append("position_ids")
            if padding_mask is not None:
                tensors_to_roll.append(padding_mask)
                fill_values.append(True)
                sequence_fields.append("padding_mask")

            rolled_tensors = roll_tensor(
                tensors_to_roll,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                fill_values=fill_values,
                roll_context=sequence_roll_context,
                sequence_fields=sequence_fields,
                roll_depth=roll_depth,
            )
            input_ids = rolled_tensors[0]
            next_rolled_tensor = 1
            if roll_position_ids:
                position_ids = rolled_tensors[next_rolled_tensor]
                next_rolled_tensor += 1
            if padding_mask is not None:
                padding_mask = rolled_tensors[next_rolled_tensor]

        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        if self.config.mtp_detach_heads:
            decoder_input = decoder_input.detach()

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # make_viewless_tensor no-ops when hidden_states is not a view (_base is None),
        # which happens after detach() with mtp_detach_heads. Activation
        # checkpointing requires at least one differentiable tensor input, so keep
        # MTP parameter gradients enabled explicitly.
        if not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        return input_ids, position_ids, padding_mask, decoder_input, hidden_states

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        decoder_input = apply_module(self.enorm)(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)

        if self.mhc_enabled:
            n = self.config.num_residual_streams
            h = self.config.hidden_size
            # hidden_states is [s, b, n*h] (multi-stream).
            # hnorm operates per-stream on the h dimension.
            s, b, _ = hidden_states.shape
            hs_streams = hidden_states.view(s, b, n, h)
            hs_streams = apply_module(self.hnorm)(hs_streams)
            hs_streams = make_viewless_tensor(inp=hs_streams, requires_grad=True, keep_graph=True)
            # e_proj/h_proj are column-parallel projections on the same TP group with
            # gather_output=False, so both outputs hold the same hidden partition.
            # Add within the partition first, then gather once across TP ranks.
            e_out, _ = self.e_proj(decoder_input)
            # h_proj: applied per-stream on the h dimension
            h_out, _ = self.h_proj(hs_streams)
            s, b, n, _ = h_out.shape
            hidden_states = e_out.unsqueeze(2) + h_out
            if not self.training:
                hidden_states = inference_all_gather_from_tensor_model_parallel_region(
                    hidden_states, self.tp_group, self.config
                )
            else:
                hidden_states = gather_from_tensor_model_parallel_region(
                    hidden_states, group=self.tp_group
                )
            # Combine and flatten back to [s, b, n*h].
            s, b, n, h = hidden_states.shape
            hidden_states = hidden_states.reshape(s, b, n * h)
            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(
                    hidden_states, group=self.tp_group
                )
        else:
            hidden_states = apply_module(self.hnorm)(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
            # At the (k - 1)-th MTP module, concatenates the i-th token's hidden_states
            # and the (i + K)-th token's embedding, and combine them with linear projection.
            hidden_states = torch.cat((decoder_input, hidden_states), -1)
            hidden_states, _ = self.eh_proj(hidden_states)
            # For tensor parallel we need to gather the tensor across the model-parallel
            # ranks after the linear projection.
            if not self.training:
                hidden_states = inference_all_gather_from_tensor_model_parallel_region(
                    hidden_states, self.tp_group, self.config
                )
            else:
                hidden_states = gather_from_tensor_model_parallel_region(
                    hidden_states, group=self.tp_group
                )
            # For sequence parallel, scatter after linear_fc and before transformer layer.
            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(
                    hidden_states, group=self.tp_group
                )
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in
        # different layers,currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        # TODO: currently ignoring FP4 in MTP layers because we need more numerical validation
        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            # Use a separate fp8 context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
            # True so that the fp8 weight caching can be triggered correctly.
            with transformer_layer_fp8_context:
                if self.mtp_layer_pattern is not None:
                    hidden_states = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_context=inference_params,
                        packed_seq_params=packed_seq_params,
                        input_ids=input_ids,
                    )
                else:
                    # GPT path: single TransformerLayer
                    hidden_states, _ = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        padding_mask=padding_mask,
                        input_ids=input_ids,
                    )

        if not self.mhc_enabled:
            hidden_states = self._postprocess(hidden_states)

        return hidden_states

    def _postprocess(self, hidden_states: torch.Tensor):
        """
        Postprocesses the output of the transformer layers.
        """

        if self.mhc_enabled:
            hidden_states = learned_output_contract(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
                self.config.num_residual_streams,
                self.config.layernorm_epsilon,
            )

        # Layer norm before shared head layer.
        hidden_states = apply_module(self.final_layernorm)(hidden_states)
        # TENorm produces a "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return hidden_states

    def forward_single_position(
        self,
        hidden_states: Tensor,
        next_token_ids: Tensor,
        position_ids: Tensor,
        embedding: Callable,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward for single positions without roll_tensor (speculative decoding).

        Unlike the regular forward which rolls input_ids to get the next token's
        embedding, this method directly takes the correct next_token_ids. This is
        used in speculative decoding where the correct next token is known after
        verification.

        Args:
            hidden_states (Tensor): Hidden states at positions of interest [N, B, H].
            next_token_ids (Tensor): The correct next token IDs [B, N].
            position_ids (Tensor): Position IDs for the next tokens [B, N].
            embedding (Callable): The embedding module.

        Returns:
            Tensor: MTP hidden states [N, B, H].
        """
        decoder_input = embedding(input_ids=next_token_ids, position_ids=position_ids)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=False, keep_graph=False
        )
        hidden_states = self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            input_ids=next_token_ids,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        return hidden_states

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """Forward through ``_proj_and_transformer_layer`` with activation
        recomputation.

        Mirrors ``transformer_block._checkpointed_forward``:

        * Non-tensor objects (``attention_bias``, ``inference_params``,
          ``packed_seq_params``) are captured by the ``custom_forward``
          closure; only tensor / ``None`` arguments flow positionally
          through the underlying checkpoint primitive. This is required
          by both backends: ``tensor_parallel.checkpoint`` because its
          ``save_for_backward`` only accepts tensors and ``None``, and
          ``te_checkpoint`` because its reentrant implementation only
          tracks positional tensor inputs as checkpoint inputs (kwarg
          tensors are not represented in the recompute backward path).
        * Quantized recipes (fp8, fp4) route through ``te_checkpoint``;
          everything else uses ``tensor_parallel.checkpoint``.
        * Only ``fp8 + delayed scaling`` needs an outer quantization
          context entered before ``te_checkpoint``; see the
          ``outer_quantization_context`` block below.
        """

        def custom_forward(
            hidden_states,
            decoder_input,
            input_ids,
            attention_mask,
            padding_mask,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ):
            return self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        # Decide the outer quantization context, matching
        # ``transformer_block._checkpointed_forward``. Only ``fp8 + delayed
        # scaling`` needs an active context at the ``te_checkpoint`` entry
        # point: TE's ``_CheckpointFunction.forward`` samples
        # ``FP8GlobalStateManager.is_fp8_enabled()`` there to gate the
        # phase-1 amax-buffer stash that phase-2 backward looks up via
        # ``global_fp8_buffer_pos_fwd_recompute``. With fp8 only entered
        # *inside* ``_proj_and_transformer_layer``, TE samples fp8 as off,
        # phase-1 skips the stash, and phase-2 raises ``KeyError``.
        # Non-delayed fp8 recipes (MXFP8BlockScaling, Float8CurrentScaling)
        # and fp4 (NVFP4BlockScaling) treat the stash/lookup as a noop, so
        # the inner context entered inside ``_proj_and_transformer_layer``
        # is sufficient.
        if self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed:
            outer_quantization_context = get_fp8_context(self.config)
        else:
            outer_quantization_context = nullcontext()

        def checkpoint_handler():
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # fp4 quantization is internally implemented via TE's
            # ``fp8_autocast`` (see ``fp4_utils.get_fp4_context``), so
            # quantized recompute on either fp8 or fp4 must go through
            # ``te_checkpoint``. Matches ``transformer_block``'s policy.
            if self.config.fp8 or self.config.fp4:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    decoder_input,
                    input_ids,
                    attention_mask,
                    padding_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                )
            else:
                # tensor_parallel.checkpoint stashes args via autograd's
                # ``save_for_backward``, which only accepts tensors and ``None``.
                # Pass tensor / ``None`` args positionally and capture the
                # non-tensor objects (``attention_bias``, ``inference_params``,
                # ``packed_seq_params``) via the ``custom_forward`` closure.
                return tensor_parallel.checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    decoder_input,
                    input_ids,
                    attention_mask,
                    padding_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            with outer_quantization_context:
                outputs = checkpoint_handler()
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
            )
            outputs = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_roll_context: Optional[MTPSequenceRollContext] = None,
        roll_depth: int = 0,
        sequence_len_offset: Optional[Tensor] = None,
        embedding=None,
        _inputs_pre_aligned: bool = False,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            sequence_roll_context: Layout-specific state shared by MTP rolls
                in this microbatch.
            roll_depth: Zero-based prediction depth selecting the prefetched
                successor row for this repeated roll.
            _inputs_pre_aligned: Internal block-to-layer contract indicating that
                sequence fields are already aligned to this absolute depth.
            sequence_len_offset (Tensor, optional): Offset for sequence length, if applicable.
            embedding (Callable): The embedding module from gpt model to compute the decoder input.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, "multi token prediction + cross attention is not yet supported."
        _orig_cp_group = self.cp_group
        self.cp_group = resolve_cp_group(self.cp_group, packed_seq_params)
        input_ids, position_ids, padding_mask, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            padding_mask=padding_mask,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
            sequence_roll_context=sequence_roll_context,
            roll_depth=roll_depth,
            _inputs_pre_aligned=_inputs_pre_aligned,
        )

        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        self.cp_group = _orig_cp_group
        return hidden_states, input_ids, position_ids, padding_mask

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        # Backward compatibility: GPT MTP checkpoints were saved with the submodule
        # named 'transformer_layer'. Remap checkpoint keys so old checkpoints load
        # correctly. Mamba MTP models keep 'mtp_model_layer' as their native format
        # since no older checkpoints exist for them.
        if self.mtp_layer_pattern is None:
            apply_prefix_mapping(
                sharded_state_dict, {f'{prefix}mtp_model_layer.': f'{prefix}transformer_layer.'}
            )

        return sharded_state_dict


@dataclass
class MultiTokenPredictionBlockSubmodules:
    """
    Dataclass for specifying the submodules of a multi token prediction block.

    This class defines the structure for configuring the layers, allowing for
    flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the multi token prediction block. Each specification typically
            defines a complete multi token prediction layer (e.g., shared embedding,
            projection matrix, transformer block, shared output head).
    """

    layer_specs: Optional[List[ModuleSpec]] = None


def _scatter_mtp_padding_mask(
    padding_mask: Optional[Tensor], *, sequence_parallel: bool, tp_group
) -> Optional[Tensor]:
    """Convert one globally aligned MTP routing mask to its TP-local layout."""
    if padding_mask is None or not sequence_parallel:
        return padding_mask
    if tp_group is None:
        raise ValueError("MTP sequence-parallel padding-mask scatter requires a TP group.")
    return (
        tensor_parallel.scatter_to_sequence_parallel_region(
            padding_mask.transpose(0, 1).contiguous(), group=tp_group
        )
        .transpose(0, 1)
        .contiguous()
    )


def _get_mtp_block_submodules(
    config: TransformerConfig, spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]
) -> MultiTokenPredictionBlockSubmodules:
    """
    Retrieve or construct MultiTokenPredictionBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]): Specification for the
            multi token prediction block submodules.
            Can be either a MultiTokenPredictionBlockSubmodules instance or a ModuleSpec.

    Returns:
        MultiTokenPredictionBlockSubmodules: The submodules for the multi token prediction block.
    """

    # Transformer block submodules.
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    When `mtp_use_repeated_layer=True` in config, instead of creating N separate MTP layers,
    only 1 layer is created and applied mtp_num_layers times.

    For more information, please refer to DeepSeek-V3 Technical Report
    https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        # New: For hybrid path with unified pattern syntax
        mtp_layer_pattern: Optional[str] = None,
        mtp_num_depths: int = 0,
        hybrid_submodules: Optional["HybridStackSubmodules"] = None,
        mamba_submodules: Optional["HybridStackSubmodules"] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config=config)
        if mamba_submodules is not None:
            if hybrid_submodules is not None:
                raise ValueError(
                    "Cannot specify both hybrid_submodules and mamba_submodules. "
                    "mamba_submodules has been deprecated; use hybrid_submodules instead."
                )
            warnings.warn(
                "mamba_submodules has been deprecated. Use hybrid_submodules instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hybrid_submodules = mamba_submodules
        self.submodules = _get_mtp_block_submodules(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.vp_stage = vp_stage
        self.mtp_layer_pattern = mtp_layer_pattern
        self.mtp_num_depths = mtp_num_depths
        self.hybrid_submodules = hybrid_submodules
        self.mtp_use_repeated_layer = self.config.mtp_use_repeated_layer
        self.name = name

        vp_size = config.virtual_pipeline_model_parallel_size
        assert is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size), (
            f"MTP layers must be placed on the last virtual pipeline stage. "
            f"Got vp_stage={vp_stage} with vp_size={vp_size}. "
            f"Placing MTP layers on different VPP stages is not currently supported."
        )

        # Initialize Context Parallelism (CP) support for MTP
        # This enables MTP to work with CP > 1 by providing the CP process group
        # to the roll_tensor function for proper boundary communication
        if pg_collection is None:
            # Use default MPU process groups if not provided
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp', 'tp'])
        else:
            # Ensure the provided process groups include CP
            assert hasattr(
                pg_collection, 'cp'
            ), "MultiTokenPredictionBlock pg_collection must have cp process group"

        self._build_layers(pg_collection)
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."
        self.cp_group = pg_collection.cp

        if self.config.mtp_detach_heads:
            # Tag MTP params so the optimizer can clip their gradients separately.
            for param in self.parameters():
                param.grad_norm_group = 'mtp'

    def _build_layers(self, pg_collection):
        # Determine number of depths to build
        if self.mtp_num_depths > 0:
            num_depths = self.mtp_num_depths
        else:
            num_depths = self.config.mtp_num_layers or len(self.submodules.layer_specs)

        def build_layer_legacy(layer_spec, layer_number):
            """Build layer using legacy spec-based approach."""
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    name=(self.name + f".layers.{layer_number}") if self.name is not None else None,
                )
            return module

        def build_layer_with_pattern(
            layer_spec, layer_number, mtp_layer_pattern, hybrid_submodules
        ):
            """Build layer using pattern-based approach (new Mamba path)."""
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=mtp_layer_pattern,
                    hybrid_submodules=hybrid_submodules,
                    name=(self.name + f".layers.{layer_number}") if self.name is not None else None,
                )
            return module

        # New Mamba path: use mtp_layer_pattern and hybrid_submodules
        if self.mtp_layer_pattern is not None and self.hybrid_submodules is not None:
            if self.mtp_use_repeated_layer:
                # Shared/repeated layer: build one layer, use it for all depths
                layer_spec = self.submodules.layer_specs[0]
                shared_layer = build_layer_with_pattern(
                    layer_spec,
                    layer_number=1,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    hybrid_submodules=self.hybrid_submodules,
                )
                self.layers = torch.nn.ModuleList([shared_layer])
            else:
                # Non-shared: each depth gets its own layers
                self.layers = torch.nn.ModuleList(
                    [
                        build_layer_with_pattern(
                            self.submodules.layer_specs[
                                min(i, len(self.submodules.layer_specs) - 1)
                            ],
                            layer_number=i + 1,
                            mtp_layer_pattern=self.mtp_layer_pattern,
                            hybrid_submodules=self.hybrid_submodules,
                        )
                        for i in range(num_depths)
                    ]
                )
        elif self.mtp_use_repeated_layer:
            # Legacy repeated layer mode
            if len(self.submodules.layer_specs) != 1:
                warnings.warn(
                    "Repeated MTP mode expects exactly 1 layer spec, got "
                    f"{len(self.submodules.layer_specs)} instead. "
                    f"The first layer will be applied {self.config.mtp_num_layers} times."
                )
            self.layers = torch.nn.ModuleList(
                [build_layer_legacy(self.submodules.layer_specs[0], layer_number=1)]
            )
        else:
            # Legacy mode: build from layer_specs
            self.layers = torch.nn.ModuleList(
                [
                    build_layer_legacy(layer_spec, i + 1)
                    for i, layer_spec in enumerate(self.submodules.layer_specs)
                ]
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_roll_context: Optional[MTPSequenceRollContext] = None,
        sequence_len_offset: Optional[Tensor] = None,
        extra_block_kwargs: Optional[dict] = None,
        embedding=None,
        mhc_multistream: Optional[Tensor] = None,
        sequence_roll_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Contracted decoder hidden states [s, b, h] when mHC is enabled.
            mhc_multistream (Tensor, optional): When mHC is enabled, the pre-contraction
                multi-stream decoder output [s, b, n*h] used as input to MTP depths.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            padding_mask (Tensor, optional): Padding mask for MoE routing (True = padded).
                This is already TP-local under sequence parallelism and remains the
                source for the cumulative-roll fallback.
            sequence_roll_padding_mask (Tensor, optional): Unsharded padding-mask source
                used to prepare and materialize absolute roll rows. Each row is scattered
                to the MTP layer's TP-local layout after materialization.
            sequence_roll_context: Layout-specific metadata shared across all MTP
                depths.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """
        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config, self.vp_stage)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        if mhc_multistream is not None:
            # mHC mode: use multi-stream for MTP depth input, contracted for loss list.
            mhc_chunks = list(torch.chunk(mhc_multistream, 1 + offset, dim=0))
            hidden_states = mhc_chunks[offset]
        else:
            hidden_states = hidden_states_list[offset]

        if self.config.mtp_detach_heads:
            hidden_states = hidden_states.detach()

        if sequence_roll_padding_mask is not None and padding_mask is None:
            raise ValueError(
                "MTP sequence_roll_padding_mask requires the corresponding TP-local padding_mask."
            )

        roll_position_ids = getattr(embedding, "add_position_embedding", True)
        roll_fields = [MTPSequenceRollField("input_ids", input_ids, -1, 0, 0)]
        if roll_position_ids:
            roll_fields.append(MTPSequenceRollField("position_ids", position_ids, -1, 0, 0))
        if sequence_roll_padding_mask is not None:
            roll_fields.append(
                MTPSequenceRollField("padding_mask", sequence_roll_padding_mask, -1, 0, True)
            )
        # A local padding mask without its unsharded source cannot participate in
        # absolute addressing. Fall back for the complete consumer group so input IDs,
        # positions, and padding retain the established cumulative-roll semantics.
        prepared_roll_context = None
        if padding_mask is None or sequence_roll_padding_mask is not None:
            prepared_roll_context = _prepare_mtp_sequence_roll_fields(
                sequence_roll_context, roll_fields, max_offset=self.config.mtp_num_layers
            )
        use_prepared_roll_rows = prepared_roll_context is not None
        aligned_rows = None
        if use_prepared_roll_rows:
            assert prepared_roll_context is not None
            # Materialize the complete group before invoking any MTP layer. A
            # depth therefore either consumes only absolute rows or stays wholly
            # on the established cumulative-roll path.
            aligned_rows = {
                field.key: prepared_roll_context.materialize_all(field.key) for field in roll_fields
            }

        for iteration in range(self.config.mtp_num_layers):
            layer_idx = 0 if self.mtp_use_repeated_layer else iteration
            if aligned_rows is not None:
                input_ids = aligned_rows["input_ids"][iteration]
                if roll_position_ids:
                    position_ids = aligned_rows["position_ids"][iteration]
                if sequence_roll_padding_mask is not None:
                    padding_mask = aligned_rows["padding_mask"][iteration]
                    padding_mask = _scatter_mtp_padding_mask(
                        padding_mask,
                        sequence_parallel=getattr(self.config, "sequence_parallel", False),
                        tp_group=getattr(self.layers[layer_idx], "tp_group", None),
                    )
            alignment_kwargs = {}
            if aligned_rows is not None:
                alignment_kwargs["_inputs_pre_aligned"] = True
            hidden_states, input_ids, position_ids, padding_mask = self.layers[layer_idx](
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_roll_context=sequence_roll_context,
                roll_depth=iteration,
                sequence_len_offset=sequence_len_offset,
                embedding=embedding,
                **alignment_kwargs,
                **(extra_block_kwargs or {}),
            )

            if mhc_multistream is not None:
                mhc_chunks.append(hidden_states)
                hidden_states_list.append(self.layers[layer_idx]._postprocess(hidden_states))
            else:
                # append the output hidden states of the current mtp layer
                # to the hidden_states_list
                hidden_states_list.append(hidden_states)

        # concat the hidden states of all mtp layers
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction module.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction module.
        """
        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'
        for layer in self.layers:
            offset = get_mtp_layer_offset(self.config, self.vp_stage)
            sharded_prefix = f'{layer_prefix}{layer.layer_number - 1}.'

            state_dict_prefix = f'{layer_prefix}{layer.layer_number - 1 - offset}.'
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict
