#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Benchmark: fused Triton MHA metadata update vs original PyTorch multi-op.

Usage:
    python tests/unit_tests/inference/contexts/benchmark_fused_mha_metadata_update.py

Measures kernel-level latency using CUDA events across a range of batch sizes.
"""

import torch

from megatron.core.inference.contexts.attention_context.triton.fused_mha_metadata_update import (
    HAVE_TRITON,
    fused_mha_metadata_update,
)


def reference_mha_metadata_update(
    query_lengths,
    kv_length_offsets,
    query_lengths_buf,
    cu_query_seq_lengths_buf,
    kv_seq_lengths_buf,
    cu_kv_seq_lengths_buf,
    real_batch_size,
    padded_batch_size,
):
    """Original multi-op implementation (mirrors MHAMetadata.update 1D logic)."""
    # query_lengths copy + pad
    query_lengths_buf[:real_batch_size] = query_lengths[:real_batch_size]
    query_lengths_buf[real_batch_size:padded_batch_size] = 0

    # cumsum(query_lengths) copy + pad
    cu_query_seq_lengths_buf[0] = 0
    if real_batch_size > 0:
        cumsum_q = torch.cumsum(query_lengths[:real_batch_size], dim=0)
        cu_query_seq_lengths_buf[1 : real_batch_size + 1] = cumsum_q
        cu_query_seq_lengths_buf[real_batch_size + 1 : padded_batch_size + 1] = cumsum_q[-1]
    else:
        cu_query_seq_lengths_buf[1 : padded_batch_size + 1] = 0

    # kv_seq_lengths = offsets + query_lengths, copy + pad
    kv_seq = kv_length_offsets[:real_batch_size] + query_lengths[:real_batch_size]
    kv_seq_lengths_buf[:real_batch_size] = kv_seq
    kv_seq_lengths_buf[real_batch_size:padded_batch_size] = 0

    # cumsum(kv_seq_lengths) copy + pad
    cu_kv_seq_lengths_buf[0] = 0
    if real_batch_size > 0:
        cumsum_kv = torch.cumsum(kv_seq, dim=0)
        cu_kv_seq_lengths_buf[1 : real_batch_size + 1] = cumsum_kv
        cu_kv_seq_lengths_buf[real_batch_size + 1 : padded_batch_size + 1] = cumsum_kv[-1]
    else:
        cu_kv_seq_lengths_buf[1 : padded_batch_size + 1] = 0


def benchmark_one(real_bs, padded_bs, max_bs, warmup=50, iters=200):
    """Benchmark a single (real_bs, padded_bs) configuration. Returns (us_ref, us_fused)."""
    device = "cuda"
    query_lengths = torch.randint(1, 128, (real_bs,), dtype=torch.int32, device=device)
    kv_length_offsets = torch.randint(0, 512, (real_bs,), dtype=torch.int32, device=device)

    def alloc():
        return (
            torch.zeros(max_bs, dtype=torch.int32, device=device),
            torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
            torch.zeros(max_bs, dtype=torch.int32, device=device),
            torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    ql_r, cu_q_r, kvl_r, cu_kv_r, mq_r, mk_r = alloc()
    ql_f, cu_q_f, kvl_f, cu_kv_f, mq_f, mk_f = alloc()

    # --- Reference ---
    for _ in range(warmup):
        reference_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_r, cu_q_r, kvl_r, cu_kv_r, real_bs, padded_bs,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        reference_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_r, cu_q_r, kvl_r, cu_kv_r, real_bs, padded_bs,
        )
    end.record()
    torch.cuda.synchronize()
    us_ref = start.elapsed_time(end) * 1000 / iters  # ms → us

    # --- Fused Triton ---
    for _ in range(warmup):
        fused_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_f, cu_q_f, kvl_f, cu_kv_f, mq_f, mk_f, real_bs, padded_bs,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fused_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_f, cu_q_f, kvl_f, cu_kv_f, mq_f, mk_f, real_bs, padded_bs,
        )
    end.record()
    torch.cuda.synchronize()
    us_fused = start.elapsed_time(end) * 1000 / iters

    return us_ref, us_fused


def benchmark_full_initialize_like(real_bs, padded_bs, max_bs, max_tokens, warmup=50, iters=200):
    """Benchmark the full initialize_attention_state GPU workload (non-hybrid path).

    Includes the 3 token-position fills + mha_metadata.update(), comparing
    original multi-op vs fused kernel for the mha_metadata portion.
    """
    device = "cuda"
    active_token_count = real_bs * 4  # approximate: 4 tokens per request
    padded_token_count = padded_bs * 4

    # Per-request inputs
    query_lengths = torch.randint(1, 128, (real_bs,), dtype=torch.int32, device=device)
    kv_length_offsets = torch.randint(0, 512, (real_bs,), dtype=torch.int32, device=device)

    # Token-level tensors (simulating the slice fills at lines 1699-1708)
    token_to_block_idx = torch.zeros(max_tokens, dtype=torch.int32, device=device)
    token_to_local_pos = torch.zeros(max_tokens, dtype=torch.int32, device=device)
    token_to_pos_in_req = torch.zeros(max_tokens, dtype=torch.int32, device=device)
    dummy_block_idx = 42

    def alloc():
        return (
            torch.zeros(max_bs, dtype=torch.int32, device=device),
            torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
            torch.zeros(max_bs, dtype=torch.int32, device=device),
            torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    ql_r, cu_q_r, kvl_r, cu_kv_r, mq_r, mk_r = alloc()
    ql_f, cu_q_f, kvl_f, cu_kv_f, mq_f, mk_f = alloc()

    def full_reference():
        # Token position fills (3 kernel launches)
        token_to_block_idx[active_token_count:padded_token_count] = dummy_block_idx
        token_to_local_pos[active_token_count:padded_token_count] = 0
        token_to_pos_in_req[active_token_count:padded_token_count] = 0
        # MHA metadata (13+ kernel launches)
        reference_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_r, cu_q_r, kvl_r, cu_kv_r, real_bs, padded_bs,
        )

    def full_fused():
        # Token position fills (3 kernel launches — unchanged)
        token_to_block_idx[active_token_count:padded_token_count] = dummy_block_idx
        token_to_local_pos[active_token_count:padded_token_count] = 0
        token_to_pos_in_req[active_token_count:padded_token_count] = 0
        # MHA metadata (1 kernel launch)
        fused_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_f, cu_q_f, kvl_f, cu_kv_f, mq_f, mk_f, real_bs, padded_bs,
        )

    # --- Reference ---
    for _ in range(warmup):
        full_reference()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        full_reference()
    end.record()
    torch.cuda.synchronize()
    us_ref = start.elapsed_time(end) * 1000 / iters

    # --- Fused ---
    for _ in range(warmup):
        full_fused()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        full_fused()
    end.record()
    torch.cuda.synchronize()
    us_fused = start.elapsed_time(end) * 1000 / iters

    return us_ref, us_fused


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    if not HAVE_TRITON:
        print("Triton not available, skipping benchmark.")
        return

    configs = [
        # (real_bs, padded_bs)
        (1, 1),
        (1, 8),
        (4, 8),
        (8, 8),
        (16, 16),
        (16, 32),
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (128, 256),
        (256, 256),
        (256, 512),
        (512, 512),
        (1024, 1024),
    ]
    max_bs = 2048

    print("=" * 60)
    print("Benchmark 1: mha_metadata.update() only (fused 1D ops)")
    print("=" * 60)
    print(f"{'real_bs':>8} {'pad_bs':>8} {'ref (us)':>10} {'fused (us)':>12} {'speedup':>8}")
    print("-" * 56)

    for real_bs, padded_bs in configs:
        us_ref, us_fused = benchmark_one(real_bs, padded_bs, max_bs)
        speedup = us_ref / us_fused if us_fused > 0 else float("inf")
        print(f"{real_bs:>8} {padded_bs:>8} {us_ref:>10.2f} {us_fused:>12.2f} {speedup:>7.2f}x")

    print()
    print("=" * 60)
    print("Benchmark 2: full initialize_attention_state GPU workload")
    print("  (3 token-position fills + mha_metadata.update)")
    print("=" * 60)
    max_tokens = 8192
    print(f"{'real_bs':>8} {'pad_bs':>8} {'ref (us)':>10} {'fused (us)':>12} {'speedup':>8}")
    print("-" * 56)

    for real_bs, padded_bs in configs:
        us_ref, us_fused = benchmark_full_initialize_like(
            real_bs, padded_bs, max_bs, max_tokens,
        )
        speedup = us_ref / us_fused if us_fused > 0 else float("inf")
        print(f"{real_bs:>8} {padded_bs:>8} {us_ref:>10.2f} {us_fused:>12.2f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
