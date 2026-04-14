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
    """

    def __init__(self, max_requests: int, max_tokens: int, device: torch.device):
        # Token-level tensors (consumed by embedding, RoPE, KV append, Mamba).
        self.token_to_input_ids = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.token_to_pos_ids = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.token_to_block_idx = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.token_to_local_position_within_kv_block = torch.zeros(
            max_tokens, dtype=torch.int32, device=device,
        )
        self.token_to_request_idx = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.token_to_position_in_request = torch.zeros(
            max_tokens, dtype=torch.int32, device=device,
        )

        # Request-level tensors (consumed by sampling, log-probs, speculative verification, MTP).
        self.request_in_prefill_status = torch.zeros(
            max_requests, dtype=torch.int32, device=device,
        )
        self.request_query_lengths = torch.zeros(
            max_requests, dtype=torch.int32, device=device,
        )
        self.request_kv_length_offsets = torch.zeros(
            max_requests, dtype=torch.int32, device=device,
        )
