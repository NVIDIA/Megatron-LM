# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch

from .metadata_base import MetadataBase


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
        # Set by bind_gpu_buffers(); references shared views in ContextGPUView._buf.
        self._gpu_view = None

    def bind_gpu_buffers(self, gpu_view) -> None:
        """Attach shared GPU buffer views from the context's ContextGPUView.

        Called by :class:`DynamicInferenceContext` after ``self.gpu_view`` is
        constructed. Both graphed and non-graphed MHA metadata bind to the
        same views; only one is active per step, so sharing storage is safe.
        """
        self._gpu_view = gpu_view

    def set_state_data(
        self, padded_active_request_count: int, max_seqlen_q: int, max_seqlen_k: int
    ) -> None:
        """Build ``state_data`` slices into the bound GPU buffers.

        Called once per step from ``transfer_bookkeeping_to_gpu`` after the
        coalesced H2D copy. No ``.copy_()`` calls, no kernel launches.
        """
        assert self._gpu_view is not None, "bind_gpu_buffers() must be called first"
        n = padded_active_request_count
        v = self._gpu_view
        self._max_seqlen_q = max_seqlen_q
        self._max_seqlen_k = max_seqlen_k
        self.state_data = {
            "query_lengths": v.mha_query_lengths[:n],
            "cu_query_seq_lengths": v.mha_cu_query_seq_lengths[: n + 1],
            "cu_kv_seq_lengths": v.mha_cu_kv_seq_lengths[: n + 1],
            "kv_seq_lengths": v.mha_kv_seq_lengths[:n],
            "block_table": v.mha_block_table[:n, :],
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
        }

    def reset(self):
        """Reset the metadata for the next batch.

        The GPU buffers live in the context's unified buffer and are fully
        overwritten by the next H2D copy; clearing them here would launch
        redundant CUDA kernels with no correctness benefit.
        """
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0


class GraphedMHAMetadata(MHAMetadata):
    """MHA metadata for CUDA-graphed execution."""


class NonGraphedMHAMetadata(MHAMetadata):
    """MHA metadata for non-graphed (eager) execution."""
