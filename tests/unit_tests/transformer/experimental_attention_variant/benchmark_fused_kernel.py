"""
Test suite for DSA (Dynamic Sparse Attention) indexer loss backward pass.

This file contains:
1. backward_native: Pure PyTorch implementation of the manual backward pass
2. Test functions to validate both implementations against PyTorch autograd

The DSA indexer loss computes KL divergence between:
- Index scores: Predicted importance scores from the indexer network
- Attention scores: True attention scores from the full attention mechanism

Backward pass computes gradients w.r.t.:
- q: Indexer query embeddings [Sq, B, H, D]
- weights: Indexer attention weights [Sq, B, H]  
- k: Indexer key embeddings [Sk, B, D]
"""
import os

import torch
import torch.distributed as dist

import numpy as np

from megatron.core.transformer.experimental_attention_variant.dsa import compute_dsa_indexer_loss, fwd_fused_indexer_loss
from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive

from tests.unit_tests.test_utilities import Utils
from megatron.core.process_groups_config import ProcessGroupCollection

USE_TILELANG = os.environ.get('USE_FUSED_INDEXER_LOSS', '0') == '2'

def bench(fn, num_warmups: int = 5, num_tests: int = 50, post_fn=None, is_async=True):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    if is_async:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        for i in range(num_tests):
            # Record
            start_events[i].record()
            fn()
            end_events[i].record()
            if post_fn is not None:
                post_fn()

        torch.cuda.synchronize()

        times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
        return np.median(times), np.min(times), np.max(times)
    else:
        start_events = torch.cuda.Event(enable_timing=True)
        end_events = torch.cuda.Event(enable_timing=True)
        start_events.record()
        for _ in range(num_tests):
            fn()
            if post_fn is not None:
                post_fn()
        end_events.record()
        
        torch.cuda.synchronize()

        times = start_events.elapsed_time(end_events) / 1e3 / num_tests

        return times, times, times

def benchmark_fused_loss_forward():
    """Benchmark compute_index_scores_topk + DSA indexer loss: native PyTorch vs Triton."""
    
    configs = [
        # (Sq, Sk, B, H, D, topk)
        (4096, 4096, 1, 128, 7168, 2048),
        (6144, 6144, 1, 128, 7168, 2048),
        # (8192, 8192, 1, 128, 7168, 2048),
    ]

    Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
    
    metrics_collection = []
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        dtype = torch.bfloat16
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=dtype)
        k = torch.randn(Sk, B, D, device='cuda', dtype=dtype)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32)
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )

        attn_query = torch.randn(Sq, B, H, D, device='cuda', dtype=dtype)
        attn_key = torch.randn(Sk, B, H, D, device='cuda', dtype=dtype)
        softmax_scale = 1.0
        loss_coeff = 1.0
        for sparse_loss in [False, True]:
            index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)

            native_lambda = lambda: compute_dsa_indexer_loss(
                index_scores, 
                topk_indices, 
                attn_query, 
                attn_key, 
                softmax_scale, 
                loss_coeff, 
                sparse_loss,
                pg_collection,
            )
            # TODO: remove accuracy_check after topk is fixed
            triton_lambda = lambda: fwd_fused_indexer_loss(
                index_scores,
                attn_query,
                attn_key,
                softmax_scale,
                loss_coeff,
                sparse_loss,
                topk_indices,
            )

            # check correctness, topk_indices has its unit test
            native_indexer_loss = native_lambda()
            triton_indexer_loss = triton_lambda()

            match = torch.allclose(native_indexer_loss, triton_indexer_loss, atol=1e-4, rtol=1e-4)
            torch.cuda.synchronize()
            
            # Benchmark PyTorch (reset peak stats first so memory is measured during bench)
            torch.cuda.reset_peak_memory_stats()
            pytorch_time, _, _ = bench(native_lambda, is_async=False, num_warmups=1, num_tests=1)
            pytorch_time *= 1000
            native_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

            torch.cuda.synchronize()

            # Benchmark Triton (reset peak stats first so memory is measured during bench)
            torch.cuda.reset_peak_memory_stats()
            triton_time, _, _ = bench(triton_lambda, is_async=False, num_warmups=1, num_tests=1)
            triton_time *= 1000
            triton_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

            mem_saved_gb = triton_mem_gb / native_mem_gb
            speedup = pytorch_time / triton_time
            marker = "🚀" if speedup > 1.0 else ""
            sparse_str = "Yes" if sparse_loss else "No"

            metrics_collection.append((Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match, native_mem_gb, triton_mem_gb, mem_saved_gb))

            print(f"[Sq={Sq}, Sk={Sk}, B={B}, H={H}, D={D}, topk={topk}, sparse_loss={sparse_loss}] completes.")

    print("\n" + "=" * 80)
    kernel_name = "Triton" if not USE_TILELANG else "TileLang"
    print(f"Benchmark: DSA Indexer (TopK + Loss) - PyTorch vs {kernel_name}")
    print("=" * 80)

    print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {f'{kernel_name} (ms)':>13} {'Speedup':>8} {'Match':>10} | {'Native (GB)':>12} {f'{kernel_name} (GB)':>12} {'Saved (GB)':>11}")
    print("-" * 135)

    for Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match, native_mem_gb, triton_mem_gb, mem_saved_gb in metrics_collection:
        saved_marker = "✓" if mem_saved_gb > 0 else ""
        print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>14.2f}   {triton_time:>13.2f}   {speedup:>8.2f}x {marker} {str(match):>10} | {native_mem_gb:>12.3f} {triton_mem_gb:>12.3f} {mem_saved_gb:>10.3f} {saved_marker}")

    print("\n" + "=" * 135)

def main():
    benchmark_fused_loss_forward()

if __name__ == "__main__":
    main()