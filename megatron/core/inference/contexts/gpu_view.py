# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch


class ContextGPUView:
    """GPU-resident snapshot of context bookkeeping data for the forward pass.

    This is the ONLY interface GPU code (attention kernels, KV append, RoPE,
    sampling, log-probs, speculative verification) uses to read context state.
    CPU bookkeeping code accesses context tensors directly.

    Populated once per step by ``DynamicInferenceContext.transfer_bookkeeping_to_gpu()``.
    All tensors have fixed addresses for CUDA graph compatibility.

    Convention:
        ``context.foo``      -> CPU (source of truth, used by bookkeeping)
        ``context.gpu_view.foo`` -> GPU (snapshot, used by forward pass)

    Layout note: the 9 bookkeeping fields are backed by a single contiguous
    ``uint8`` buffer (``self._buf``). Each field is a ``view(dtype)`` onto a
    slice of that buffer. This matches the pinned-CPU-buffer layout in
    :class:`DynamicInferenceContext` so that the per-step H2D transfer is a
    single ``cudaMemcpyAsync`` instead of nine small ones.
    """

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        max_kv_blocks: int,
        device: torch.device,
    ):
        # Field layout (must match DynamicInferenceContext's CPU buffer layout):
        #   int64 token fields first (auto 8-byte alignment), then int32 token
        #   fields, then int32 request fields, then int32 MHA fields.
        tok_int64_bytes = max_tokens * 8  # 2 fields of int64 = 8 bytes/elem
        tok_int32_bytes = max_tokens * 4  # 4 fields of int32 = 4 bytes/elem
        req_int32_bytes = max_requests * 4  # 3 fields of int32

        # MHA section: 5 fields shared by both graphed and non-graphed MHAMetadata
        # (only one is active per step, so sharing storage is fine).
        #   mha_query_lengths          int32 (max_bs,)         = max_bs   * 4
        #   mha_cu_query_seq_lengths   int32 (max_bs + 1,)     = (max_bs+1) * 4
        #   mha_kv_seq_lengths         int32 (max_bs,)         = max_bs   * 4
        #   mha_cu_kv_seq_lengths      int32 (max_bs + 1,)     = (max_bs+1) * 4
        #   mha_block_table            int32 (max_bs, max_kv_blocks)
        # max_bs == max_requests in DynamicInferenceContext.
        max_bs = max_requests
        mha_query_lengths_bytes = max_bs * 4
        mha_cu_query_seq_lengths_bytes = (max_bs + 1) * 4
        mha_kv_seq_lengths_bytes = max_bs * 4
        mha_cu_kv_seq_lengths_bytes = (max_bs + 1) * 4
        mha_block_table_bytes = max_bs * max_kv_blocks * 4

        total_bytes = (
            2 * tok_int64_bytes
            + 4 * tok_int32_bytes
            + 3 * req_int32_bytes
            + mha_query_lengths_bytes
            + mha_cu_query_seq_lengths_bytes
            + mha_kv_seq_lengths_bytes
            + mha_cu_kv_seq_lengths_bytes
            + mha_block_table_bytes
        )

        # Zero-initialized so pre-transfer reads see zeros (matches prior semantics).
        self._buf = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

        # Token-level tensors (consumed by embedding, RoPE, KV append, Mamba).
        off = 0
        self.token_to_input_ids = self._buf[off : off + tok_int64_bytes].view(torch.long)
        off += tok_int64_bytes
        self.token_to_pos_ids = self._buf[off : off + tok_int64_bytes].view(torch.long)
        off += tok_int64_bytes
        self.token_to_block_idx = self._buf[off : off + tok_int32_bytes].view(torch.int32)
        off += tok_int32_bytes
        self.token_to_local_position_within_kv_block = self._buf[
            off : off + tok_int32_bytes
        ].view(torch.int32)
        off += tok_int32_bytes
        self.token_to_request_idx = self._buf[off : off + tok_int32_bytes].view(torch.int32)
        off += tok_int32_bytes
        self.token_to_position_in_request = self._buf[off : off + tok_int32_bytes].view(
            torch.int32
        )
        off += tok_int32_bytes

        # Request-level tensors (consumed by sampling, log-probs, speculative verification, MTP).
        self.request_in_prefill_status = self._buf[off : off + req_int32_bytes].view(torch.int32)
        off += req_int32_bytes
        self.request_query_lengths = self._buf[off : off + req_int32_bytes].view(torch.int32)
        off += req_int32_bytes
        self.request_kv_length_offsets = self._buf[off : off + req_int32_bytes].view(torch.int32)
        off += req_int32_bytes

        # MHA flash-attention metadata (shared between GraphedMHAMetadata and
        # NonGraphedMHAMetadata — only one is active per step).
        self.mha_query_lengths = self._buf[off : off + mha_query_lengths_bytes].view(torch.int32)
        off += mha_query_lengths_bytes
        self.mha_cu_query_seq_lengths = self._buf[
            off : off + mha_cu_query_seq_lengths_bytes
        ].view(torch.int32)
        off += mha_cu_query_seq_lengths_bytes
        self.mha_kv_seq_lengths = self._buf[off : off + mha_kv_seq_lengths_bytes].view(torch.int32)
        off += mha_kv_seq_lengths_bytes
        self.mha_cu_kv_seq_lengths = self._buf[
            off : off + mha_cu_kv_seq_lengths_bytes
        ].view(torch.int32)
        off += mha_cu_kv_seq_lengths_bytes
        self.mha_block_table = (
            self._buf[off : off + mha_block_table_bytes]
            .view(torch.int32)
            .view(max_bs, max_kv_blocks)
        )
        off += mha_block_table_bytes

        assert off == total_bytes, f"layout bug: wrote {off} of {total_bytes} bytes"
