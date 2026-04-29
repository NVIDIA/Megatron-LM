# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .metadata_base import MetadataBase


@dataclass(frozen=True)
class MHAMetadataSnapshot:
    """Per-snapshot MHA launch metadata bound to one GPU view."""

    gpu_view: Any
    state_data: Dict[str, Any]
    padded_active_request_count: int
    max_seqlen_q: int
    max_seqlen_k: int
    padded_batch_dimensions: Optional[Any] = None


class MHAMetadata(MetadataBase):
    """
    Metadata for MHA layer using flash-attention.

    GPU storage for the per-step fields (``query_lengths``,
    ``cu_query_seq_lengths``, ``kv_seq_lengths``, ``cu_kv_seq_lengths``,
    ``block_table``) lives inside the context's :class:`ContextGPUView`
    unified buffer. Both :class:`GraphedMHAMetadata` and
    :class:`NonGraphedMHAMetadata` bind to the same GPU views (only one is
    active per step), so the single coalesced H2D in
    :meth:`DynamicInferenceContext.transfer_bookkeeping_to_gpu` covers the
    MHA fields along with the rest of the bookkeeping state.
    """

    def __init__(
        self, block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen
    ):
        super().__init__()
        self.device = torch.cuda.current_device()
        self.max_blocks = block_count_total
        self.max_kv_blocks = max_kv_block_count
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0
        self.state_data = {}
        self.snapshot_state = None
        self._snapshot_states = {}
        # Set by bind_gpu_buffers(); references shared views in ContextGPUView._buf.
        self._gpu_view = None

    def bind_gpu_buffers(self, gpu_view) -> None:
        """Attach shared GPU buffer views from the context's ContextGPUView.

        Called by :class:`DynamicInferenceContext` after ``self.gpu_view`` is
        constructed. Both graphed and non-graphed MHA metadata bind to the
        same views; only one is active per step, so sharing storage is safe.
        """
        self._gpu_view = gpu_view

    def make_snapshot_state(
        self,
        padded_active_request_count: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        *,
        gpu_view=None,
        padded_batch_dimensions=None,
    ) -> MHAMetadataSnapshot:
        """Build immutable per-step slices into one snapshot GPU view."""
        v = gpu_view if gpu_view is not None else self._gpu_view
        assert v is not None, "bind_gpu_buffers() must be called first"
        n = padded_active_request_count
        state_data = {
            "query_lengths": v.mha_query_lengths[:n],
            "cu_query_seq_lengths": v.mha_cu_query_seq_lengths[: n + 1],
            "cu_kv_seq_lengths": v.mha_cu_kv_seq_lengths[: n + 1],
            "kv_seq_lengths": v.mha_kv_seq_lengths[:n],
            "block_table": v.mha_block_table[:n, :],
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
        }
        return MHAMetadataSnapshot(
            gpu_view=v,
            state_data=state_data,
            padded_active_request_count=n,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            padded_batch_dimensions=padded_batch_dimensions,
        )

    def activate_snapshot_state(self, snapshot_or_slot_id) -> MHAMetadataSnapshot:
        """Install a stored MHA snapshot state as the active layer-facing view."""
        if isinstance(snapshot_or_slot_id, MHAMetadataSnapshot):
            snapshot = snapshot_or_slot_id
        else:
            snapshot = self._snapshot_states[int(snapshot_or_slot_id)]

        self._gpu_view = snapshot.gpu_view
        self._max_seqlen_q = snapshot.max_seqlen_q
        self._max_seqlen_k = snapshot.max_seqlen_k
        self.state_data = snapshot.state_data
        self.snapshot_state = snapshot
        return snapshot

    def set_state_data(
        self,
        padded_active_request_count: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        *,
        gpu_view=None,
        snapshot_slot_id: Optional[int] = None,
        padded_batch_dimensions=None,
    ) -> MHAMetadataSnapshot:
        """Build ``state_data`` slices into the bound GPU buffers.

        Called once per step from ``transfer_bookkeeping_to_gpu`` after the
        coalesced H2D copy. No ``.copy_()`` calls, no kernel launches.
        """
        snapshot = self.make_snapshot_state(
            padded_active_request_count,
            max_seqlen_q,
            max_seqlen_k,
            gpu_view=gpu_view,
            padded_batch_dimensions=padded_batch_dimensions,
        )
        if snapshot_slot_id is not None:
            self._snapshot_states[int(snapshot_slot_id)] = snapshot
        self.activate_snapshot_state(snapshot)
        return snapshot

    def reset(self):
        """Reset the metadata for the next batch.

        The GPU buffers live in the context's unified buffer and are fully
        overwritten by the next H2D copy; clearing them here would launch
        redundant CUDA kernels with no correctness benefit.
        """
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0
        self.state_data = {}
        self.snapshot_state = None
        self._snapshot_states.clear()


class GraphedMHAMetadata(MHAMetadata):
    """MHA metadata for CUDA-graphed execution."""


class NonGraphedMHAMetadata(MHAMetadata):
    """MHA metadata for non-graphed (eager) execution."""
