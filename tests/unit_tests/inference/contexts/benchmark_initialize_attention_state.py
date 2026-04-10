#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""End-to-end benchmark of initialize_attention_state.

Profiles the full function using CUDA events to measure wall-clock GPU time,
then uses torch.profiler to break down where time is spent (MHA metadata,
Mamba metadata, token fills, etc.).

Usage:
    python tests/unit_tests/inference/contexts/benchmark_initialize_attention_state.py
"""

import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def setup_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)


def teardown_parallel():
    Utils.destroy_model_parallel()


def make_context(num_requests, is_hybrid=False, max_seq_len=512):
    """Create a DynamicInferenceContext and add requests."""
    num_layers = 4
    if is_hybrid:
        layer_type_list = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]
        mamba_config = MambaInferenceStateConfig(
            layer_type_list=layer_type_list,
            conv_states_shape=(544, 4),
            ssm_states_shape=(8, 64, 16),
            conv_states_dtype=torch.float32,
            ssm_states_dtype=torch.float32,
        )
        num_layers = len(layer_type_list)
    else:
        layer_type_list = None
        mamba_config = None

    # Use rounder=1 so padded sizes are predictable.
    orig_token_rounder = DynamicInferenceContext.TOKEN_ROUNDER
    orig_request_rounder = DynamicInferenceContext.REQUEST_ROUNDER
    DynamicInferenceContext.TOKEN_ROUNDER = 1
    DynamicInferenceContext.REQUEST_ROUNDER = 1

    ctx = DynamicInferenceContext(
        model_config=TransformerConfig(
            params_dtype=torch.float32,
            num_layers=num_layers,
            kv_channels=8,
            num_attention_heads=2,
        ),
        inference_config=InferenceConfig(
            max_sequence_length=max_seq_len,
            num_cuda_graphs=0,
            buffer_size_gb=1.0,
            paused_buffer_size_gb=0.1,
            block_size_tokens=128,
            max_tokens=None,
            mamba_inference_state_config=mamba_config,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
            enable_chunked_prefill=False,
            max_requests=max(num_requests * 4, 512),
        ),
    )

    DynamicInferenceContext.TOKEN_ROUNDER = orig_token_rounder
    DynamicInferenceContext.REQUEST_ROUNDER = orig_request_rounder

    # Add requests with small prompts
    prompt_len = 10
    for i in range(num_requests):
        ctx.add_request(
            DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=torch.arange(prompt_len, dtype=torch.long, device="cuda"),
                sampling_params=SamplingParams(num_tokens_to_generate=max_seq_len - prompt_len),
            )
        )

    return ctx


def benchmark_cuda_events(ctx, warmup=20, iters=100):
    """Measure initialize_attention_state latency using CUDA events."""
    for _ in range(warmup):
        ctx.initialize_attention_state()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        ctx.initialize_attention_state()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1000 / iters  # us


def profile_breakdown(ctx, iters=20):
    """Use torch.profiler to get a per-kernel breakdown."""
    # Warmup
    for _ in range(10):
        ctx.initialize_attention_state()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            ctx.initialize_attention_state()
        torch.cuda.synchronize()

    # Print CUDA kernel summary
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30, max_name_column_width=80,
        )
    )
    return prof


def benchmark_update_requests(ctx, warmup=20, iters=100):
    """Measure update_requests latency using CUDA events.

    Simulates a decode step: initialize_attention_state → update_requests,
    measuring only the update_requests portion.
    """
    device = "cuda"
    active_count = ctx.total_request_count - ctx.paused_request_count

    for _ in range(warmup):
        ctx.initialize_attention_state()
        # Simulate all requests staying active (mask=1)
        mask = torch.ones(active_count, dtype=torch.uint8, device=device)
        new_tokens = torch.zeros(active_count, dtype=torch.long, device=device)
        ctx.update_requests(mask, new_tokens)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        ctx.initialize_attention_state()
        end_init = torch.cuda.Event(enable_timing=True)
        end_init.record()

        mask = torch.ones(active_count, dtype=torch.uint8, device=device)
        new_tokens = torch.zeros(active_count, dtype=torch.long, device=device)
        ctx.update_requests(mask, new_tokens)
    end.record()
    torch.cuda.synchronize()

    total_us = start.elapsed_time(end) * 1000 / iters
    return total_us


def profile_full_step(ctx, iters=20):
    """Profile the full init + update cycle."""
    device = "cuda"
    active_count = ctx.total_request_count - ctx.paused_request_count

    for _ in range(10):
        ctx.initialize_attention_state()
        mask = torch.ones(active_count, dtype=torch.uint8, device=device)
        new_tokens = torch.zeros(active_count, dtype=torch.long, device=device)
        ctx.update_requests(mask, new_tokens)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            ctx.initialize_attention_state()
            mask = torch.ones(active_count, dtype=torch.uint8, device=device)
            new_tokens = torch.zeros(active_count, dtype=torch.long, device=device)
            ctx.update_requests(mask, new_tokens)
        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30, max_name_column_width=80,
        )
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    setup_parallel()

    try:
        configs = [
            # (num_requests, is_hybrid, label)
            (1, False, "1 req, MHA-only"),
            (8, False, "8 req, MHA-only"),
            (32, False, "32 req, MHA-only"),
            (64, False, "64 req, MHA-only"),
            (128, False, "128 req, MHA-only"),
            (1, True, "1 req, hybrid (mamba)"),
            (8, True, "8 req, hybrid (mamba)"),
            (32, True, "32 req, hybrid (mamba)"),
            (64, True, "64 req, hybrid (mamba)"),
        ]

        print("=" * 70)
        print("initialize_attention_state latency (CUDA events)")
        print("=" * 70)
        print(f"{'config':<30} {'latency (us)':>15}")
        print("-" * 50)

        for num_req, is_hybrid, label in configs:
            ctx = make_context(num_req, is_hybrid=is_hybrid)
            us = benchmark_cuda_events(ctx)
            print(f"{label:<30} {us:>15.2f}")
            del ctx

        print()
        print("=" * 70)
        print("Full step (init + update_requests) latency (CUDA events)")
        print("=" * 70)
        print(f"{'config':<30} {'latency (us)':>15}")
        print("-" * 50)

        for num_req, is_hybrid, label in configs:
            ctx = make_context(num_req, is_hybrid=is_hybrid)
            us = benchmark_update_requests(ctx)
            print(f"{label:<30} {us:>15.2f}")
            del ctx

        # Detailed profiler breakdown for a representative case
        for num_req, is_hybrid, label in [(32, False, "32 req MHA-only"), (32, True, "32 req hybrid")]:
            print()
            print("=" * 70)
            print(f"Profiler breakdown (full step): {label}")
            print("=" * 70)
            ctx = make_context(num_req, is_hybrid=is_hybrid)
            profile_full_step(ctx)
            del ctx

    finally:
        teardown_parallel()


if __name__ == "__main__":
    main()
