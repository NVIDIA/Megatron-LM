# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import Tensor


class AsyncTxnState(Enum):
    """Lifecycle states for one async decode transaction."""

    PREPARED = "prepared"
    LAUNCHED = "launched"
    RESOLVED = "resolved"
    COMMITTED = "committed"
    RETIRED = "retired"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class AsyncGraphShape:
    """CUDA graph and decode-stride shape used by an async forward."""

    active_request_count: int
    active_token_count: int
    padded_active_request_count: int | None
    tokens_per_request: int


@dataclass(frozen=True)
class AsyncLayoutSnapshot:
    """CPU-visible request, token, and state layout for an async decode forward."""

    request_ids: Tensor
    graph_shape: AsyncGraphShape
    request_query_lengths: Tensor | None = None
    request_kv_length_offsets: Tensor | None = None
    token_to_request_idx: Tensor | None = None
    token_to_pos_ids: Tensor | None = None
    token_to_block_idx: Tensor | None = None
    token_to_local_position_within_kv_block: Tensor | None = None
    mamba_read_indices: Tensor | None = None
    mamba_write_indices: Tensor | None = None

    @classmethod
    def from_pending_forward_view(
        cls, pending_view: Any, *, tokens_per_request: int
    ) -> "AsyncLayoutSnapshot":
        """Build a snapshot from the legacy pending-forward view."""
        request_ids = pending_view.pending_request_ids.clone()
        request_count = int(request_ids.numel())
        graph_shape = AsyncGraphShape(
            active_request_count=request_count,
            active_token_count=request_count * tokens_per_request,
            padded_active_request_count=pending_view.cuda_graph_request_count,
            tokens_per_request=tokens_per_request,
        )
        return cls(
            request_ids=request_ids,
            graph_shape=graph_shape,
            request_query_lengths=_clone_optional(pending_view.planned_request_query_lengths),
            request_kv_length_offsets=_clone_optional(
                pending_view.planned_request_kv_length_offsets
            ),
            token_to_request_idx=_clone_optional(pending_view.planned_token_to_request_idx),
            token_to_pos_ids=_clone_optional(pending_view.planned_token_to_pos_ids),
            token_to_block_idx=_clone_optional(pending_view.planned_token_to_block_idx),
            token_to_local_position_within_kv_block=_clone_optional(
                pending_view.planned_token_to_local_position_within_kv_block
            ),
            mamba_read_indices=_clone_optional(pending_view.planned_mamba_read_indices),
            mamba_write_indices=_clone_optional(pending_view.planned_mamba_write_indices),
        )

    @classmethod
    def from_context_current(
        cls, context: Any, *, tokens_per_request: int
    ) -> "AsyncLayoutSnapshot":
        """Build a snapshot from the context's current active rows."""
        active_slice = slice(context.paused_request_count, context.total_request_count)
        request_ids = context.request_ids[active_slice].clone()
        request_count = int(request_ids.numel())
        active_token_count = int(getattr(context, "active_token_count", 0))
        padded_request_count = (
            context.padded_active_request_count
            if context.using_cuda_graph_this_step()
            else None
        )
        graph_shape = AsyncGraphShape(
            active_request_count=request_count,
            active_token_count=active_token_count,
            padded_active_request_count=padded_request_count,
            tokens_per_request=tokens_per_request,
        )

        mamba_read_indices = None
        mamba_write_indices = None
        if getattr(context, "is_hybrid_model", False):
            mamba_read_indices = context._mamba_flat_indices(active_slice)[:request_count].clone()
            mamba_write_indices = context._mamba_flat_indices(
                active_slice, use_candidate_bank=True
            )[:request_count].clone()

        return cls(
            request_ids=request_ids,
            graph_shape=graph_shape,
            request_query_lengths=_clone_active(context, "request_query_lengths", active_slice),
            request_kv_length_offsets=_clone_active(
                context, "request_kv_length_offsets", active_slice
            ),
            token_to_request_idx=_clone_token_rows(
                context, "token_to_request_idx", request_count, tokens_per_request
            ),
            token_to_pos_ids=_clone_token_rows(
                context, "token_to_pos_ids", request_count, tokens_per_request
            ),
            token_to_block_idx=_clone_token_rows(
                context, "token_to_block_idx", request_count, tokens_per_request
            ),
            token_to_local_position_within_kv_block=_clone_token_rows(
                context,
                "token_to_local_position_within_kv_block",
                request_count,
                tokens_per_request,
            ),
            mamba_read_indices=mamba_read_indices,
            mamba_write_indices=mamba_write_indices,
        )

    def row_map_to_current(self, current_request_ids: Tensor) -> Tensor | None:
        """Map current request rows to this snapshot's pending rows."""
        current_request_ids = current_request_ids.to(device="cpu")
        pending_request_ids = self.request_ids.to(device="cpu")
        current_count = int(current_request_ids.numel())
        if current_count == 0 or int(pending_request_ids.numel()) < current_count:
            return None
        if torch.equal(pending_request_ids, current_request_ids):
            return torch.arange(current_count, dtype=torch.long, device="cpu")

        pending_row_by_request_id = {
            int(request_id): row for row, request_id in enumerate(pending_request_ids.tolist())
        }
        mapped_rows = []
        for request_id in current_request_ids.tolist():
            row = pending_row_by_request_id.get(int(request_id))
            if row is None:
                return None
            mapped_rows.append(row)
        return torch.tensor(mapped_rows, dtype=torch.long, device="cpu")

    def graph_compatible_with(self, current: "AsyncLayoutSnapshot") -> bool:
        """Return whether the pending graph execution shape can satisfy current rows."""
        return (
            self.graph_shape.tokens_per_request == current.graph_shape.tokens_per_request
            and self.graph_shape.padded_active_request_count
            == current.graph_shape.padded_active_request_count
        )

    def layout_compatible_with(
        self, current: "AsyncLayoutSnapshot", *, row_map: Tensor | None = None
    ) -> bool:
        """Return whether current CPU bookkeeping matches this pending layout."""
        planned_fields = (
            self.request_query_lengths,
            self.request_kv_length_offsets,
            self.token_to_request_idx,
            self.token_to_pos_ids,
            self.token_to_block_idx,
            self.token_to_local_position_within_kv_block,
            self.mamba_read_indices,
            self.mamba_write_indices,
        )
        if all(field is None for field in planned_fields):
            return True

        request_count = int(current.request_ids.numel())
        pending_request_count = int(self.request_ids.numel())
        if request_count == 0 or pending_request_count < request_count:
            return False

        row_map = self.row_map_to_current(current.request_ids) if row_map is None else row_map
        if row_map is None:
            return False
        row_map = row_map.to(dtype=torch.long, device="cpu")
        if int(row_map.numel()) != request_count:
            return False
        if bool((row_map < 0).any()) or bool((row_map >= pending_request_count).any()):
            return False

        if not _row_field_matches(self.request_query_lengths, current.request_query_lengths, row_map):
            return False
        if not _row_field_matches(
            self.request_kv_length_offsets, current.request_kv_length_offsets, row_map
        ):
            return False

        tokens_per_request = self.graph_shape.tokens_per_request
        if current.graph_shape.active_token_count != request_count * tokens_per_request:
            return False

        if self.token_to_request_idx is not None:
            if current.token_to_request_idx is None:
                return False
            current_token_rows = current.token_to_request_idx.to(dtype=torch.long, device="cpu")
            planned_token_rows = self.token_to_request_idx.index_select(0, row_map).to(
                dtype=torch.long, device="cpu"
            )
            if (
                bool((current_token_rows < 0).any())
                or bool((current_token_rows >= request_count).any())
                or bool((planned_token_rows < 0).any())
                or bool((planned_token_rows >= pending_request_count).any())
            ):
                return False
            current_token_request_ids = current.request_ids.to(device="cpu").index_select(
                0, current_token_rows.reshape(-1)
            ).view_as(current_token_rows)
            planned_token_request_ids = self.request_ids.to(device="cpu").index_select(
                0, planned_token_rows.reshape(-1)
            ).view_as(planned_token_rows)
            if not torch.equal(current_token_request_ids, planned_token_request_ids):
                return False

        for planned, current_field in (
            (self.token_to_pos_ids, current.token_to_pos_ids),
            (self.token_to_block_idx, current.token_to_block_idx),
            (
                self.token_to_local_position_within_kv_block,
                current.token_to_local_position_within_kv_block,
            ),
            (self.mamba_read_indices, current.mamba_read_indices),
            (self.mamba_write_indices, current.mamba_write_indices),
        ):
            if not _row_field_matches(planned, current_field, row_map):
                return False
        return True


def _clone_optional(value: Tensor | None) -> Tensor | None:
    if value is None:
        return None
    return value.clone()


def _clone_active(context: Any, name: str, active_slice: slice) -> Tensor | None:
    value = getattr(context, name, None)
    if value is None:
        return None
    return value[active_slice].clone()


def _clone_token_rows(
    context: Any, name: str, request_count: int, tokens_per_request: int
) -> Tensor | None:
    value = getattr(context, name, None)
    if value is None:
        return None
    token_count = request_count * tokens_per_request
    if token_count <= 0:
        return None
    return value[:token_count].view(request_count, tokens_per_request).clone()


def _row_field_matches(planned: Tensor | None, current: Tensor | None, row_map: Tensor) -> bool:
    if planned is None:
        return True
    if current is None:
        return False
    return torch.equal(current.to(device="cpu"), planned.index_select(0, row_map).to(device="cpu"))


@dataclass
class AsyncStepTransaction:
    """Owns one speculative async decode forward and all state tied to it."""

    step_id: int
    state: AsyncTxnState
    snapshot: AsyncLayoutSnapshot
    sample_ticket: object | None = None
    resources: object | None = None
    h2d_done_event: object | None = None
    forward_done_event: object | None = None
    ep_decision: object | None = None
    row_map: object | None = None
    discard_reason: str | None = None
    pending_forward_view: object | None = None

    def mark_launched(
        self,
        *,
        sample_ticket: object | None = None,
        resources: object | None = None,
        h2d_done_event: object | None = None,
        forward_done_event: object | None = None,
        ep_decision: object | None = None,
    ) -> None:
        """Mark the async forward as launched and attach launch-owned state."""
        self.sample_ticket = sample_ticket if sample_ticket is not None else self.sample_ticket
        self.resources = resources if resources is not None else self.resources
        self.h2d_done_event = h2d_done_event if h2d_done_event is not None else self.h2d_done_event
        self.forward_done_event = (
            forward_done_event if forward_done_event is not None else self.forward_done_event
        )
        self.ep_decision = ep_decision if ep_decision is not None else self.ep_decision
        self.state = AsyncTxnState.LAUNCHED

    def resolve_against_current(self, context: object) -> Tensor | None:
        """Resolve this transaction's pending rows against the current context."""
        current = AsyncLayoutSnapshot.from_context_current(
            context, tokens_per_request=self.snapshot.graph_shape.tokens_per_request
        )
        if not self.snapshot.graph_compatible_with(current):
            self.discard("graph shape mismatch")
            return None
        row_map = self.snapshot.row_map_to_current(current.request_ids)
        if row_map is None or not self.snapshot.layout_compatible_with(current, row_map=row_map):
            self.discard("layout mismatch")
            return None
        self.row_map = row_map
        self.state = AsyncTxnState.RESOLVED
        return row_map

    def mark_committed(self) -> None:
        """Mark transaction-owned side effects as committed."""
        self.state = AsyncTxnState.COMMITTED

    def mark_retired(self) -> None:
        """Mark the transaction as no longer owning in-flight state."""
        self.state = AsyncTxnState.RETIRED

    def discard(self, reason: str) -> None:
        """Discard the transaction and remember why it could not commit."""
        self.discard_reason = reason
        self.state = AsyncTxnState.DISCARDED

    @property
    def is_in_flight(self) -> bool:
        """Whether the transaction still represents a pending async forward."""
        return self.state in (AsyncTxnState.PREPARED, AsyncTxnState.LAUNCHED, AsyncTxnState.RESOLVED)


@dataclass(frozen=True)
class AsyncSampleTicket:
    """References the sample buffers and fences for one async readback slot."""

    slot: int
    active_request_count: int
    sampled_tokens_cuda: object
    sample_values_cuda: object | None
    sampled_tokens_cpu: object
    sampled_mtp_tokens_cuda: object | None = None
    sampled_mtp_tokens_cpu: object | None = None
    source_ready_event: object | None = None
    copy_done_event: object | None = None
    copy_stream: object | None = None


@dataclass
class AsyncSampleReadback:
    """Owns CUDA/CPU sample slots and asynchronous sample-copy fences."""

    sample_slot_count: int
    current_sample_slot: int
    sampled_tokens_cuda_slots: object
    sample_values_cuda_slots: object
    sampled_tokens_cpu_slots: object
    source_ready_events: tuple[object, ...]
    copy_done_events: tuple[object, ...]
    copy_stream: object
    sampled_mtp_tokens_cuda_slots: object | None = None
    sampled_mtp_tokens_cpu_slots: object | None = None

    @classmethod
    def allocate(
        cls,
        *,
        sample_slot_count: int,
        max_requests: int,
        logits_dtype: torch.dtype,
        device: object,
        num_speculative_tokens: int,
    ) -> "AsyncSampleReadback":
        """Allocate the stable sample buffers used by async decode."""
        sampled_tokens_cuda_slots = torch.empty(
            (sample_slot_count, max_requests), dtype=torch.int64, device=device
        )
        sample_values_cuda_slots = torch.empty(
            (sample_slot_count, max_requests), dtype=logits_dtype, device=device
        )
        sampled_tokens_cpu_slots = torch.empty(
            (sample_slot_count, max_requests),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        sampled_mtp_tokens_cuda_slots = None
        sampled_mtp_tokens_cpu_slots = None
        if num_speculative_tokens > 0:
            sampled_mtp_tokens_cuda_slots = torch.empty(
                [sample_slot_count, num_speculative_tokens, max_requests],
                dtype=torch.int64,
                device=device,
            )
            sampled_mtp_tokens_cpu_slots = torch.empty(
                [sample_slot_count, num_speculative_tokens, max_requests],
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
        return cls(
            sample_slot_count=sample_slot_count,
            current_sample_slot=0,
            sampled_tokens_cuda_slots=sampled_tokens_cuda_slots,
            sample_values_cuda_slots=sample_values_cuda_slots,
            sampled_tokens_cpu_slots=sampled_tokens_cpu_slots,
            sampled_mtp_tokens_cuda_slots=sampled_mtp_tokens_cuda_slots,
            sampled_mtp_tokens_cpu_slots=sampled_mtp_tokens_cpu_slots,
            source_ready_events=tuple(torch.cuda.Event() for _ in range(sample_slot_count)),
            copy_done_events=tuple(torch.cuda.Event() for _ in range(sample_slot_count)),
            copy_stream=torch.cuda.Stream(device=device),
        )

    def select_slot(
        self, sample_slot: int, *, num_speculative_tokens: int
    ) -> AsyncSampleTicket:
        """Select a readback slot and return references to its buffers and fences."""
        if sample_slot < 0 or sample_slot >= self.sample_slot_count:
            raise IndexError(f"async sample slot {sample_slot} is outside slot count")
        self.current_sample_slot = sample_slot
        sampled_mtp_tokens_cuda = None
        sampled_mtp_tokens_cpu = None
        if num_speculative_tokens > 0:
            if self.sampled_mtp_tokens_cuda_slots is not None:
                sampled_mtp_tokens_cuda = self.sampled_mtp_tokens_cuda_slots[sample_slot]
            if self.sampled_mtp_tokens_cpu_slots is not None:
                sampled_mtp_tokens_cpu = self.sampled_mtp_tokens_cpu_slots[sample_slot]
        return AsyncSampleTicket(
            slot=sample_slot,
            active_request_count=0,
            sampled_tokens_cuda=self.sampled_tokens_cuda_slots[sample_slot],
            sample_values_cuda=self.sample_values_cuda_slots[sample_slot],
            sampled_tokens_cpu=self.sampled_tokens_cpu_slots[sample_slot],
            sampled_mtp_tokens_cuda=sampled_mtp_tokens_cuda,
            sampled_mtp_tokens_cpu=sampled_mtp_tokens_cpu,
            source_ready_event=self.source_ready_events[sample_slot],
            copy_done_event=self.copy_done_events[sample_slot],
            copy_stream=self.copy_stream,
        )

    def transfer_to_cpu(
        self,
        *,
        active_request_count: int,
        num_speculative_tokens: int,
        sample_source_ready_event: object | None = None,
        sample_slot: int | None = None,
    ) -> AsyncSampleTicket:
        """Copy sampled tokens to pinned CPU memory without blocking the default stream."""
        sample_slot = self.current_sample_slot if sample_slot is None else sample_slot
        ticket = self.select_slot(sample_slot, num_speculative_tokens=num_speculative_tokens)
        if sample_source_ready_event is None:
            current_stream = torch.cuda.current_stream()
            sample_source_ready_event = ticket.source_ready_event
            sample_source_ready_event.record(current_stream)

        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(sample_source_ready_event)
            self.sampled_tokens_cpu_slots[sample_slot, :active_request_count].copy_(
                self.sampled_tokens_cuda_slots[sample_slot, :active_request_count],
                non_blocking=True,
            )
            sampled_mtp_tokens_cpu = None
            if num_speculative_tokens > 0:
                self.sampled_mtp_tokens_cpu_slots[
                    sample_slot, :, :active_request_count
                ].copy_(
                    self.sampled_mtp_tokens_cuda_slots[
                        sample_slot, :, :active_request_count
                    ],
                    non_blocking=True,
                )
                sampled_mtp_tokens_cpu = self.sampled_mtp_tokens_cpu_slots[
                    sample_slot, :, :active_request_count
                ]
            ticket.copy_done_event.record(self.copy_stream)

        return AsyncSampleTicket(
            slot=sample_slot,
            active_request_count=active_request_count,
            sampled_tokens_cuda=self.sampled_tokens_cuda_slots[sample_slot],
            sample_values_cuda=self.sample_values_cuda_slots[sample_slot],
            sampled_tokens_cpu=self.sampled_tokens_cpu_slots[
                sample_slot, :active_request_count
            ],
            sampled_mtp_tokens_cuda=(
                self.sampled_mtp_tokens_cuda_slots[sample_slot]
                if num_speculative_tokens > 0
                else None
            ),
            sampled_mtp_tokens_cpu=sampled_mtp_tokens_cpu,
            source_ready_event=sample_source_ready_event,
            copy_done_event=ticket.copy_done_event,
            copy_stream=self.copy_stream,
        )


@dataclass(frozen=True)
class AsyncKVReservation:
    """One KV block reserved by a speculative async decode plan."""

    request_id: int
    block_column: int
    block_id: int


@dataclass(frozen=True)
class AsyncMambaLease:
    """One Mamba slot/bank lease associated with an async forward."""

    request_id: int
    slot_id: int
    bank_id: int


@dataclass
class AsyncResourceLedger:
    """Owns resource lifetime for one speculative async forward."""

    kv_reservations: list[AsyncKVReservation] | None = None
    deferred_kv_blocks: list[int] | None = None
    deferred_mamba_slots: list[int] | None = None
    mamba_leases: list[AsyncMambaLease] | None = None
    in_flight: bool = False
    consumed_reservation_count: int = 0

    def __post_init__(self) -> None:
        if self.kv_reservations is None:
            self.kv_reservations = []
        if self.deferred_kv_blocks is None:
            self.deferred_kv_blocks = []
        if self.deferred_mamba_slots is None:
            self.deferred_mamba_slots = []
        if self.mamba_leases is None:
            self.mamba_leases = []

    @property
    def reservation_count(self) -> int:
        """Number of KV reservations still owned by the ledger."""
        return len(self.kv_reservations)

    def clear_reservations(self) -> None:
        """Forget all unused KV reservations without releasing them."""
        self.kv_reservations.clear()

    def record_reservations_from_plan(self, plan: object) -> None:
        """Record KV reservations produced by an async lifecycle plan."""
        self.record_reservations(
            request_ids=plan.reserved_request_ids,
            block_ids=plan.reserved_block_ids,
            block_columns=plan.reserved_block_columns,
        )

    def record_reservations(
        self, *, request_ids: Tensor, block_ids: Tensor, block_columns: Tensor
    ) -> None:
        """Replace current reservations with the provided request/block mapping."""
        self.kv_reservations = [
            AsyncKVReservation(
                request_id=int(request_id),
                block_column=int(block_column),
                block_id=int(block_id),
            )
            for request_id, block_id, block_column in zip(
                request_ids.tolist(), block_ids.tolist(), block_columns.tolist()
            )
        ]

    def consume_reserved_blocks(self, request_ids: Tensor, block_columns: Tensor) -> Tensor:
        """Return reserved blocks for request/block-column pairs, or -1 where absent."""
        reserved_block_ids = torch.full(
            (request_ids.numel(),), -1, dtype=torch.int32, device="cpu"
        )
        if not self.kv_reservations or request_ids.numel() == 0:
            return reserved_block_ids

        remaining = list(self.kv_reservations)
        for request_idx, (request_id_tensor, block_column_tensor) in enumerate(
            zip(request_ids.tolist(), block_columns.tolist())
        ):
            request_id = int(request_id_tensor)
            block_column = int(block_column_tensor)
            for reservation in tuple(remaining):
                if reservation.request_id != request_id:
                    continue
                if reservation.block_column != block_column:
                    continue
                reserved_block_ids[request_idx] = reservation.block_id
                remaining.remove(reservation)
                self.consumed_reservation_count += 1
                break
        self.kv_reservations = remaining
        return reserved_block_ids

    def defer_unused_reservations(self) -> None:
        """Move unconsumed KV reservations into the deferred release list."""
        if self.kv_reservations:
            self.deferred_kv_blocks.extend(
                reservation.block_id for reservation in self.kv_reservations
            )
        self.kv_reservations.clear()

    def defer_kv_blocks(self, blocks: Tensor) -> None:
        """Defer releasing KV blocks that may be visible to an in-flight forward."""
        self.deferred_kv_blocks.extend(_tensor_ints(blocks, skip_negative=True))

    def defer_mamba_slots(self, slots: Tensor) -> None:
        """Defer freeing Mamba slots that may still receive async writes."""
        self.deferred_mamba_slots.extend(_tensor_ints(slots, skip_negative=True))

    def release_deferred(self, context: object) -> None:
        """Release all deferred KV and Mamba resources through the context allocators."""
        self.in_flight = False
        if self.deferred_kv_blocks:
            blocks = torch.tensor(self.deferred_kv_blocks, dtype=torch.int32, device="cpu")
            context._async_deferred_kv_block_release_count += int(blocks.numel())
            context.kv_block_allocator.release_memory_blocks(blocks)
            self.deferred_kv_blocks.clear()
        self._release_deferred_mamba_slots(context)

    def drain(self, context: object) -> None:
        """Defer unused reservations and release every deferred resource."""
        self.defer_unused_reservations()
        self.release_deferred(context)

    def deferred_kv_tensor(self) -> Tensor:
        """Return deferred KV blocks as a CPU tensor for diagnostics/tests."""
        return torch.tensor(self.deferred_kv_blocks, dtype=torch.int32, device="cpu")

    def deferred_mamba_tensor(self) -> Tensor:
        """Return deferred Mamba slots as a CPU tensor for diagnostics/tests."""
        return torch.tensor(self.deferred_mamba_slots, dtype=torch.int32, device="cpu")

    def reserved_request_ids_tensor(self) -> Tensor:
        """Return reserved request ids as a CPU tensor for diagnostics/tests."""
        return torch.tensor(
            [reservation.request_id for reservation in self.kv_reservations],
            dtype=torch.int32,
            device="cpu",
        )

    def reserved_block_ids_tensor(self) -> Tensor:
        """Return reserved KV block ids as a CPU tensor for diagnostics/tests."""
        return torch.tensor(
            [reservation.block_id for reservation in self.kv_reservations],
            dtype=torch.int32,
            device="cpu",
        )

    def reserved_block_columns_tensor(self) -> Tensor:
        """Return reserved KV block columns as a CPU tensor for diagnostics/tests."""
        return torch.tensor(
            [reservation.block_column for reservation in self.kv_reservations],
            dtype=torch.int32,
            device="cpu",
        )

    def _release_deferred_mamba_slots(self, context: object) -> None:
        if not getattr(context, "is_hybrid_model", False) or not self.deferred_mamba_slots:
            return
        slots = torch.tensor(self.deferred_mamba_slots, dtype=torch.int32, device="cpu")
        context._async_deferred_mamba_slot_release_count += int(slots.numel())
        context.mamba_metadata.free_slot_ids(slots)
        self.deferred_mamba_slots.clear()


def _tensor_ints(values: Tensor, *, skip_negative: bool = False) -> list[int]:
    values = values.to(dtype=torch.int64, device="cpu").reshape(-1)
    if skip_negative:
        values = values[values != -1]
    return [int(value) for value in values.tolist()]


@dataclass(frozen=True)
class AsyncEligibilityDecision:
    """Classified async scheduling eligibility for the current controller/context state."""

    can_prepare: bool
    can_launch: bool
    reason: str | None
    requires_barrier: bool = False


def classify_async_eligibility(
    controller: object,
    context: object,
    *,
    allow_mtp: bool = False,
    check_context: bool = True,
) -> AsyncEligibilityDecision:
    """Return whether async decode can prepare and launch, preserving diagnostics."""
    reason = _classify_async_disabled_reason(
        controller, context, allow_mtp=allow_mtp, check_context=check_context
    )
    return AsyncEligibilityDecision(
        can_prepare=reason is None,
        can_launch=reason is None,
        reason=reason,
        requires_barrier=reason == "waiting request admission deferred",
    )


def _classify_async_disabled_reason(
    controller: object, context: object, *, allow_mtp: bool, check_context: bool
) -> str | None:
    if not controller._async_scheduling_enabled:
        return "disabled"
    if controller._async_step_barrier_reason is not None:
        return controller._async_step_barrier_reason
    if not controller._enable_cuda_graph:
        return "requires local cuda graphs"
    inference_cuda_graph_scope = controller.model_config.inference_cuda_graph_scope
    if getattr(inference_cuda_graph_scope, "name", inference_cuda_graph_scope) != "block":
        return "requires block-scope inference cuda graphs"
    if controller.model_is_pipeline_parallel:
        return "pipeline parallel is unsupported"
    if controller.num_speculative_tokens != 0 and not allow_mtp:
        return "mtp pre-sampling graph is unsupported"
    if (
        controller.num_speculative_tokens != 0
        and controller._num_mtp_depths != controller.num_speculative_tokens
    ):
        return "not enough mtp heads"
    if controller._sampling_backend != "torch":
        return "sampling backend is unsupported"
    if not check_context:
        return None

    if not context.is_decode_only():
        return "not decode-only"
    if not context.using_cuda_graph_this_step():
        return "not using cuda graph"

    active_request_count = context.total_request_count - context.paused_request_count
    if active_request_count <= 0:
        return "no active requests"
    if controller._async_admission_barrier_requested:
        controller._async_admission_barrier_requested = False
        return "waiting request admission deferred"

    tokens_per_request = controller.num_speculative_tokens + 1
    if (
        context.padded_batch_dimensions.token_count
        != context.padded_batch_dimensions.decode_req_count * tokens_per_request
    ):
        return "cuda graph shape does not match decode stride"

    return None
