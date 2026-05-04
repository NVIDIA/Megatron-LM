# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch


class ContextGPUView:
    """GPU-resident snapshot of context bookkeeping data for the forward pass.

    This is the ONLY interface GPU code (attention kernels, KV append, RoPE,
    sampling, log-probs, speculative verification) uses to read context state.
    CPU bookkeeping code accesses context tensors directly.

    Populated once per step inside ``DynamicInferenceContext.run_attn_init_graph_body``
    via a single coalesced cudaMemcpyAsync from the matching pinned CPU buffer.
    All tensors have fixed addresses for CUDA graph compatibility.

    Convention:
        ``context.foo``      -> CPU (source of truth, used by bookkeeping)
        ``context.gpu_view.foo`` -> GPU (snapshot, used by forward pass)

    Layout note: the bookkeeping fields are backed by a single contiguous
    ``uint8`` buffer (``self._buf``). Each field is a ``view(dtype)`` onto a
    slice of that buffer. This matches the pinned-CPU-buffer layout in
    :class:`DynamicInferenceContext` so that the per-step H2D transfer is a
    single ``cudaMemcpyAsync`` instead of one per field.

    Per-step MHA fields (cumulative query lengths, kv sequence lengths,
    padded block table) and per-step Mamba fields (batch indices, seq_idx,
    cu_seqlens, chunk and conv metadata) are derived on GPU from these
    sources after the H2D, so they live as separate static tensors on the
    context / on :class:`MambaMetadata` — not in this view.
    """

    def __init__(
        self,
        max_requests: int,
        max_tokens: int,
        max_kv_blocks: int,
        device: torch.device,
        is_hybrid: bool = False,
    ):
        # Field layout (must match DynamicInferenceContext's CPU buffer layout):
        #   int64 token fields first (auto 8-byte alignment), then int32 token
        #   fields, then int32/float32 request fields, then the
        #   request-to-kv-block-ids matrix, then the request-to-mamba-state-idx
        #   vector (hybrid models only).
        tok_int64_bytes = max_tokens * 8  # 2 fields of int64 = 8 bytes/elem
        tok_int32_bytes = max_tokens * 4  # 4 fields of int32 = 4 bytes/elem
        # Request-level fields are all 4 bytes wide. 3 int32 (in_prefill_status,
        # query_lengths, kv_length_offsets) + 1 int32 (top_k) + 2 float32
        # (temperature, top_p) + 1 int32 (active_request_last_token_idxs) = 7 fields.
        req_4byte_bytes = max_requests * 4

        # Per-request KV block table.
        request_to_kv_block_ids_bytes = max_requests * max_kv_blocks * 4

        # Per-request mamba slot index (hybrid models only). Source for the
        # GPU-side Mamba update kernels via ``gpu_view.request_to_mamba_state_idx``.
        request_to_mamba_state_idx_bytes = max_requests * 4 if is_hybrid else 0

        total_bytes = (
            2 * tok_int64_bytes
            + 4 * tok_int32_bytes
            + 7 * req_4byte_bytes
            + request_to_kv_block_ids_bytes
            + request_to_mamba_state_idx_bytes
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
        self.token_to_local_position_within_kv_block = self._buf[off : off + tok_int32_bytes].view(
            torch.int32
        )
        off += tok_int32_bytes
        self.token_to_request_idx = self._buf[off : off + tok_int32_bytes].view(torch.int32)
        off += tok_int32_bytes
        self.token_to_position_in_request = self._buf[off : off + tok_int32_bytes].view(torch.int32)
        off += tok_int32_bytes

        # Request-level tensors (consumed by sampling, log-probs, speculative verification, MTP).
        self.request_in_prefill_status = self._buf[off : off + req_4byte_bytes].view(torch.int32)
        off += req_4byte_bytes
        self.request_query_lengths = self._buf[off : off + req_4byte_bytes].view(torch.int32)
        off += req_4byte_bytes
        self.request_kv_length_offsets = self._buf[off : off + req_4byte_bytes].view(torch.int32)
        off += req_4byte_bytes
        # Sampling parameters (consumed by FlashInfer sampling).
        # Mirror the active slice of `active_request_metadata[{label}]`;
        # padded slots get neutral defaults from `pad_active_slices` (T=1.0, top_k=0, top_p=0.0).
        self.temperature = self._buf[off : off + req_4byte_bytes].view(torch.float32)
        off += req_4byte_bytes
        self.top_k = self._buf[off : off + req_4byte_bytes].view(torch.int32)
        off += req_4byte_bytes
        self.top_p = self._buf[off : off + req_4byte_bytes].view(torch.float32)
        off += req_4byte_bytes
        # Per-request last-token row indices (consumed by sampling kernels as `gather_indices`).
        # The CPU side of this slot IS `context.active_request_last_token_idxs`,
        # populated by `build_active_slices` and `pad_active_slices`.
        self.active_request_last_token_idxs = self._buf[off : off + req_4byte_bytes].view(
            torch.int32
        )
        off += req_4byte_bytes

        # Per-request KV block table — source for the GPU-side ``active_request_to_kv_block_ids``
        # padding, also read directly by RoPE / KV-append helpers via
        # ``gpu_view.request_to_kv_block_ids[req_idx, ...]``.
        max_bs = max_requests
        self.request_to_kv_block_ids = (
            self._buf[off : off + request_to_kv_block_ids_bytes]
            .view(torch.int32)
            .view(max_bs, max_kv_blocks)
        )
        off += request_to_kv_block_ids_bytes

        # Per-request Mamba slot index (hybrid models only). Source for
        # :meth:`MambaMetadata.update`'s GPU compute.
        if is_hybrid:
            self.request_to_mamba_state_idx = self._buf[
                off : off + request_to_mamba_state_idx_bytes
            ].view(torch.int32)
            off += request_to_mamba_state_idx_bytes
        else:
            self.request_to_mamba_state_idx = None

        assert off == total_bytes, f"layout bug: wrote {off} of {total_bytes} bytes"
