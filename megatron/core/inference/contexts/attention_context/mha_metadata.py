# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch

from .metadata_base import MetadataBase


class MHAMetadata(MetadataBase):
    """Metadata for MHA layer using flash-attention.

    The per-step fields (``query_lengths``, ``cu_query_seq_lengths``,
    ``kv_seq_lengths``, ``cu_kv_seq_lengths``, ``block_table``) are stored
    in static GPU tensors owned by :class:`DynamicInferenceContext`. The
    context populates them by running GPU kernels on the post-H2D
    bookkeeping in :class:`ContextGPUView`. Both
    :class:`GraphedMHAMetadata` and :class:`NonGraphedMHAMetadata` bind to
    the same buffers (only one is active per step, so sharing is safe).
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
        # Set by bind_active_tensors(); references shared static GPU tensors
        # on the context.
        self._query_lengths_buf = None
        self._cu_query_seq_lengths_buf = None
        self._kv_seq_lengths_buf = None
        self._cu_kv_seq_lengths_buf = None
        self._block_table_buf = None

    def bind_active_tensors(
        self,
        *,
        query_lengths_buf: torch.Tensor,
        cu_query_seq_lengths_buf: torch.Tensor,
        kv_seq_lengths_buf: torch.Tensor,
        cu_kv_seq_lengths_buf: torch.Tensor,
        block_table_buf: torch.Tensor,
    ) -> None:
        """Attach the static GPU buffers used to back ``state_data`` slices.

        Called once by :class:`DynamicInferenceContext` after the buffers are
        allocated. Each buffer is sized for ``max_requests`` (or
        ``max_requests + 1`` for cumulative tensors); per-step
        :meth:`set_state_data` slices them down to ``padded_active_request_count``.
        """
        self._query_lengths_buf = query_lengths_buf
        self._cu_query_seq_lengths_buf = cu_query_seq_lengths_buf
        self._kv_seq_lengths_buf = kv_seq_lengths_buf
        self._cu_kv_seq_lengths_buf = cu_kv_seq_lengths_buf
        self._block_table_buf = block_table_buf

    def set_state_data(
        self, padded_active_request_count: int, max_seqlen_q: int, max_seqlen_k: int
    ) -> None:
        """Build ``state_data`` slices into the bound active buffers.

        Called once per step after the GPU compute that fills those buffers.
        No ``.copy_()`` calls, no kernel launches.
        """
        assert (
            self._query_lengths_buf is not None
        ), "bind_active_tensors() must be called first"
        n = padded_active_request_count
        self._max_seqlen_q = max_seqlen_q
        self._max_seqlen_k = max_seqlen_k
        self.state_data = {
            "query_lengths": self._query_lengths_buf[:n],
            "cu_query_seq_lengths": self._cu_query_seq_lengths_buf[: n + 1],
            "cu_kv_seq_lengths": self._cu_kv_seq_lengths_buf[: n + 1],
            "kv_seq_lengths": self._kv_seq_lengths_buf[:n],
            "block_table": self._block_table_buf[:n, :],
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
        }

    def reset(self):
        """Reset the metadata for the next batch.

        The GPU buffers are owned by the context and fully overwritten by the
        next ``initialize_attention_state`` step; clearing them here would
        launch redundant CUDA kernels with no correctness benefit.
        """
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0


class GraphedMHAMetadata(MHAMetadata):
    """MHA metadata for CUDA-graphed execution."""


class NonGraphedMHAMetadata(MHAMetadata):
    """MHA metadata for non-graphed (eager) execution."""
