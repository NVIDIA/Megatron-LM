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
