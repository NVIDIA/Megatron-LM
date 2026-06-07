# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transaction objects for async decode scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
from torch import Tensor


class AsyncTransactionState(str, Enum):
    """Lifecycle states for one launched async decode transaction."""

    PREPARED = "prepared"
    IN_FLIGHT = "in_flight"
    READY = "ready"
    CONSUMED = "consumed"
    DROPPED = "dropped"


@dataclass(frozen=True)
class AsyncGraphShape:
    """CUDA graph and logical token shape for a pending async decode forward."""

    active_request_count: int
    active_token_count: int
    padded_active_request_count: int
    tokens_per_request: int = 1

    @classmethod
    def from_plan(cls, plan: Any, *, tokens_per_request: int = 1) -> "AsyncGraphShape":
        """Build a graph shape from a context async decode plan."""

        return cls(
            active_request_count=int(plan.active_request_count),
            active_token_count=int(plan.active_token_count),
            padded_active_request_count=int(plan.padded_active_request_count),
            tokens_per_request=int(tokens_per_request),
        )

    def compatible_with(self, other: "AsyncGraphShape") -> bool:
        """Return whether two shapes expose logits with the same captured layout."""

        return (
            self.padded_active_request_count == other.padded_active_request_count
            and self.tokens_per_request == other.tokens_per_request
            and self.active_token_count >= other.active_token_count
            and self.active_request_count >= other.active_request_count
        )


@dataclass(frozen=True)
class AsyncRowMap:
    """Map current live request rows onto rows produced by a pending forward."""

    pending_request_ids: Tensor
    current_request_ids: Tensor
    pending_rows_cpu: Tensor
    pending_rows: Optional[Tensor] = None

    @classmethod
    def for_current(
        cls,
        *,
        pending_request_ids: Tensor,
        current_request_ids: Tensor,
        device: Optional[torch.device | int | str] = None,
    ) -> Optional["AsyncRowMap"]:
        """Resolve current request rows against pending rows, or return None."""

        pending_cpu = pending_request_ids.to(dtype=torch.long, device="cpu")
        current_cpu = current_request_ids.to(dtype=torch.long, device="cpu")
        if current_cpu.numel() > pending_cpu.numel():
            return None
        if pending_cpu.unique().numel() != pending_cpu.numel():
            return None
        if current_cpu.unique().numel() != current_cpu.numel():
            return None

        row_by_request_id = {int(request_id): row for row, request_id in enumerate(pending_cpu)}
        mapped_rows = []
        for request_id in current_cpu.tolist():
            row = row_by_request_id.get(int(request_id))
            if row is None:
                return None
            mapped_rows.append(row)

        pending_rows_cpu = torch.tensor(mapped_rows, dtype=torch.long, device="cpu")
        identity = torch.arange(
            current_cpu.numel(), dtype=pending_rows_cpu.dtype, device="cpu"
        )
        if device is None or torch.equal(pending_rows_cpu, identity):
            pending_rows = None
        else:
            pending_rows = pending_rows_cpu.to(device=device)

        return cls(
            pending_request_ids=pending_cpu.clone(),
            current_request_ids=current_cpu.clone(),
            pending_rows_cpu=pending_rows_cpu,
            pending_rows=pending_rows,
        )

    @property
    def row_mapped(self) -> bool:
        """Whether current rows differ from the pending forward row prefix."""

        identity = torch.arange(
            self.current_request_ids.numel(), dtype=self.pending_rows_cpu.dtype, device="cpu"
        )
        return not bool(torch.equal(self.pending_rows_cpu, identity))

    def rows_for_device(self, device: torch.device | int | str) -> Tensor:
        """Return pending rows on the requested device."""

        rows = self.pending_rows_cpu.to(device=device)
        if self.pending_rows is not None and self.pending_rows.device == rows.device:
            return self.pending_rows
        return rows


@dataclass(frozen=True)
class AsyncKVBlockLease:
    """KV blocks reserved or deferred by one async transaction."""

    reserved_request_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    reserved_block_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    reserved_block_columns: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    deferred_block_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))

    @property
    def has_reservations(self) -> bool:
        """Whether this transaction owns reserved KV blocks."""

        return self.reserved_block_ids.numel() > 0


@dataclass(frozen=True)
class AsyncMambaLease:
    """Mamba state ownership for candidate/live state under async decode."""

    candidate_request_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    deferred_slot_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    uses_candidate_bank: bool = False


@dataclass
class AsyncSampleTransfer:
    """Sample buffer and D2H transfer state owned by a transaction."""

    active_request_count: int = 0
    started: bool = False
    finished: bool = False


@dataclass
class AsyncMTPState:
    """Speculative token rows and acceptance metadata owned by a transaction."""

    tokens_per_request: int
    accepted_counts: Optional[Tensor] = None
    accepted_tokens: Optional[Tensor] = None
    rewind_metadata: Optional[Any] = None


@dataclass
class AsyncDecodeTransaction:
    """Own one pending async decode forward from layout preparation to retirement."""

    transaction_id: int
    prepared_layout: Any
    graph_shape: AsyncGraphShape
    kv_lease: AsyncKVBlockLease = field(default_factory=AsyncKVBlockLease)
    mamba_lease: AsyncMambaLease = field(default_factory=AsyncMambaLease)
    sample_transfer: AsyncSampleTransfer = field(default_factory=AsyncSampleTransfer)
    mtp_state: Optional[AsyncMTPState] = None
    state: AsyncTransactionState = AsyncTransactionState.PREPARED
    cuda_graph_request_count: Optional[int] = None
    row_map: Optional[AsyncRowMap] = None
    drop_reason: Optional[str] = None

    @classmethod
    def from_plan(
        cls,
        *,
        transaction_id: int,
        prepared_layout: Any,
        tokens_per_request: int = 1,
        uses_mamba_candidate_bank: bool = False,
        cuda_graph_request_count: Optional[int] = None,
    ) -> "AsyncDecodeTransaction":
        """Build a transaction and derive all resource leases from the prepared plan."""

        request_ids = prepared_layout.request_ids.clone()
        transaction = cls(
            transaction_id=transaction_id,
            prepared_layout=prepared_layout,
            graph_shape=AsyncGraphShape.from_plan(
                prepared_layout, tokens_per_request=tokens_per_request
            ),
            kv_lease=AsyncKVBlockLease(
                reserved_request_ids=prepared_layout.reserved_request_ids.clone(),
                reserved_block_ids=prepared_layout.reserved_block_ids.clone(),
                reserved_block_columns=prepared_layout.reserved_block_columns.clone(),
            ),
            mamba_lease=AsyncMambaLease(
                candidate_request_ids=(
                    request_ids
                    if uses_mamba_candidate_bank
                    else torch.empty(0, dtype=request_ids.dtype, device=request_ids.device)
                ),
                uses_candidate_bank=uses_mamba_candidate_bank,
            ),
            mtp_state=(
                AsyncMTPState(tokens_per_request=tokens_per_request)
                if tokens_per_request > 1
                else None
            ),
        )
        transaction.launch(cuda_graph_request_count=cuda_graph_request_count)
        return transaction

    @property
    def request_ids(self) -> Tensor:
        """Request IDs in the pending forward row order."""

        return self.prepared_layout.request_ids

    @property
    def is_pending(self) -> bool:
        """Whether the transaction still owns an unconsumed forward."""

        return self.state in {
            AsyncTransactionState.PREPARED,
            AsyncTransactionState.IN_FLIGHT,
            AsyncTransactionState.READY,
        }

    def launch(self, *, cuda_graph_request_count: Optional[int]) -> None:
        """Mark the prepared transaction as launched."""

        self._require_state(AsyncTransactionState.PREPARED)
        self.cuda_graph_request_count = cuda_graph_request_count
        self.state = AsyncTransactionState.IN_FLIGHT

    def resolve_for_current(
        self,
        *,
        current_request_ids: Tensor,
        current_graph_shape: AsyncGraphShape,
        device: Optional[torch.device | int | str] = None,
    ) -> Optional[AsyncRowMap]:
        """Return the row map if current rows can consume this transaction."""

        if self.state != AsyncTransactionState.IN_FLIGHT:
            return None
        if not self.graph_shape.compatible_with(current_graph_shape):
            return None
        return AsyncRowMap.for_current(
            pending_request_ids=self.request_ids,
            current_request_ids=current_request_ids,
            device=device,
        )

    def mark_ready(self, row_map: AsyncRowMap) -> None:
        """Mark the pending forward ready for sampling."""

        self._require_state(AsyncTransactionState.IN_FLIGHT)
        self.row_map = row_map
        self.state = AsyncTransactionState.READY

    def consume(self) -> None:
        """Mark logits and owned resources as consumed."""

        self._require_state(AsyncTransactionState.READY)
        self.state = AsyncTransactionState.CONSUMED

    def drop(self, reason: str) -> None:
        """Drop the transaction and quarantine/release its owned resources."""

        if self.state in {AsyncTransactionState.CONSUMED, AsyncTransactionState.DROPPED}:
            raise RuntimeError(f"cannot drop async transaction in state {self.state.value}")
        self.drop_reason = reason
        self.state = AsyncTransactionState.DROPPED

    def _require_state(self, expected: AsyncTransactionState) -> None:
        if self.state != expected:
            raise RuntimeError(
                f"async transaction expected state {expected.value}, got {self.state.value}"
            )
