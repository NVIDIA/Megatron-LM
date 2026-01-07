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
import argparse
import sys 

import torch
import torch.distributed as dist

import triton
import triton.language as tl

import numpy as np

from megatron.core.transformer.experimental_attention_variant.fused_loss import fwd_fused_indexer_loss as compute_dsa_indexer_loss_triton
from megatron.core.transformer.experimental_attention_variant.fused_loss import bwd_fused_indexer_loss as backward_triton_full
from megatron.core.process_groups_config import ProcessGroupCollection
import megatron.core.parallel_state as parallel_state

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


def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Apply ReLU activation.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1).contiguous()

    return index_scores


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection = None,
    attention_scores: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    if attention_scores is None:
        # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
        query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
        # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
        key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
        # Compute attention scores [b * np, sq, sk]
        attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
        # Reshape to [b, np, sq, sk]
        attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection is not None and pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss, kl_per_element


def forward_native(q, weights, k, mask, index_topk, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection=None):
    index_scores = _compute_index_scores(q, weights, k)

    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    seqlen = index_scores.size(-1)
    topk_k = min(index_topk, seqlen)
    # [batch, seqlen, index_topk]
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    indexer_loss, kl_per_element = compute_dsa_indexer_loss(
        index_scores.clone(), topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection=pg_collection
    )
    # indexer_loss = torch.zeros(1, device=q.device, dtype=torch.float32)
    # kl_per_element = torch.zeros((q.shape[1], q.shape[1], k.shape[0]), device=q.device, dtype=torch.float32)

    return topk_indices, indexer_loss, kl_per_element, index_scores


def backward_native(
    q, weights, k, query, key, topk_indices, 
    softmax_scale, loss_coeff, sparse_loss,
    grad_loss
):
    """
    Pure PyTorch implementation of backward pass for DSA indexer loss.
    
    This function computes gradients of the KL divergence loss w.r.t. the indexer
    parameters (q, weights, k). It uses recomputation to save memory - forward
    values are recomputed during backward instead of being cached.
    
    Args:
        q: Indexer query embeddings [Sq, B, H, D]
        weights: Indexer attention weights [Sq, B, H]
        k: Indexer key embeddings [Sk, B, D]
        query: Attention query embeddings [Sq, B, AH, AD]
        key: Attention key embeddings [Sk, B, AH, AD]
        topk_indices: Top-k indices from forward pass [B, Sq, topk]
        softmax_scale: Scaling factor for attention scores
        loss_coeff: Coefficient for the loss
        sparse_loss: Whether to use sparse loss (only topk positions)
        grad_loss: Gradient from upstream (typically 1.0)
    
    Returns:
        grad_q: Gradient w.r.t. q [Sq, B, H, D]
        grad_weights: Gradient w.r.t. weights [Sq, B, H]
        grad_k: Gradient w.r.t. k [Sk, B, D]
    
    Algorithm:
        1. Recompute index_scores and attention_scores
        2. Apply masks (causal + optional sparse)
        3. Compute softmax for both
        4. Compute gradient of KL divergence w.r.t. index_scores_softmax
        5. Backpropagate through softmax
        6. Backpropagate through index_scores computation (einsum, relu, weights)
    """
    # Recompute index_scores (this is the "unfused" part in backward)
    # Trade-off: extra computation vs memory saving
    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # Apply causal mask to both attention and index scores
    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    # [b, sq, sk] + [1, sq, sk] -> [b, sq, sk]  
    index_scores = index_scores + causal_mask.unsqueeze(0)
    
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask
    
    # Compute softmax for both
    attention_scores_softmax = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    
    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # L1 normalize
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(dim=-1, keepdim=True)
    
    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar
    
    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar value for each row
    
    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = torch.full((b, sq, sk), grad_kl_per_row.item(), 
                                      device=index_scores.device, dtype=torch.float32)
    
    # Backward through kl_per_element = target * (log(target) - log(index))
    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    
    # Backward through softmax: ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    
    # Zero out gradients for masked positions
    # Create a mask for valid (non-masked) positions
    # Causal mask: position (i, j) is valid if j <= i
    causal_valid_mask = torch.tril(torch.ones((sq, sk), device=index_scores.device, dtype=torch.bool))  # [sq, sk]
    if sparse_loss:
        # Also apply index mask - only topk positions are valid
        index_valid_mask = (index_mask == 0)  # [b, sq, sk]
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask  # [b, sq, sk]
    else:
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)  # [b, sq, sk]
    
    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    
    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    
    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    
    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]
    scores_after_relu = torch.relu(scores)
    
    # Backward through multiplication by weights: index_scores_per_head * weights
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]
    
    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    
    # Backward through ReLU
    relu_mask = (scores > 0).float()
    grad_scores = grad_scores_after_relu * relu_mask  # [sq, b, h, sk]
    
    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


def benchmark_fused_loss_forward():
    """Benchmark compute_index_scores_topk + DSA indexer loss: native PyTorch vs Triton."""
    
    configs = [
        # (Sq, Sk, B, H, D, topk)
        (2048, 2048, 1, 8, 128, 1024),
        (2048, 2048, 1, 8, 128, 2048),
        (2048, 2048, 1, 32, 128, 2048),
        (8192, 8192, 1, 8, 128, 2048),
        (8192, 8192, 1, 32, 128, 2048),
        (16384, 16384, 1, 8, 128, 2048),
        (16384, 16384, 1, 32, 128, 2048),
    ]
    
    metrics_collection = []
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32)
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )

        attn_query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        attn_key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.bfloat16)
        softmax_scale = 1.0
        loss_coeff = 1.0
        for sparse_loss in [False, True]:
            native_lambda = lambda: forward_native(q, weights, k, mask, topk, attn_query, attn_key, softmax_scale, loss_coeff, sparse_loss)
            # TODO: remove accuracy_check after topk is fixed
            triton_acc_lambda = lambda: compute_dsa_indexer_loss_triton(q, weights, k, attn_query, attn_key, topk, softmax_scale, loss_coeff, mask=mask, sparse_loss=sparse_loss, accuracy_check=True)
            triton_ben_lambda = lambda: compute_dsa_indexer_loss_triton(q, weights, k, attn_query, attn_key, topk, softmax_scale, loss_coeff, mask=mask, sparse_loss=sparse_loss, accuracy_check=False)

            # check correctness, topk_indices has its unit test
            native_topk_indices, native_indexer_loss, native_kl_per_element, native_index_scores = native_lambda()
            triton_topk_indices, triton_indexer_loss, triton_kl_per_element = triton_acc_lambda()

            match = torch.allclose(native_indexer_loss, triton_indexer_loss, atol=1e-4, rtol=1e-4)
            torch.cuda.synchronize()
            
            # Benchmark PyTorch
            pytorch_time, _, _ = bench(native_lambda, is_async=False)
            pytorch_time *= 1000
            
            # Benchmark Triton
            triton_time, _, _ = bench(triton_ben_lambda, is_async=False)
            triton_time *= 1000
            
            speedup = pytorch_time / triton_time
            marker = "🚀" if speedup > 1.0 else ""
            sparse_str = "Yes" if sparse_loss else "No"
            
            metrics_collection.append((Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match))

    print("\n" + "=" * 80)
    print("Benchmark: DSA Indexer (TopK + Loss) - PyTorch vs Triton")
    print("=" * 80)

    print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>8} {'Match':>10}")
    print("-" * 95)

    for Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match in metrics_collection:
        print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>12.2f}   {triton_time:>11.2f}   {speedup:>6.2f}x {marker} {match}")

    print("\n" + "=" * 95)

def benchmark_fused_loss_backward_tensor_parallel():
    from tests.unit_tests.test_utilities import Utils
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    tensor_model_parallel_size = 8

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
    )

    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    triton_pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    native_pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

    torch.cuda.set_device(dist.get_rank())
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    """Benchmark compute_index_scores_topk + DSA indexer loss: native PyTorch vs Triton."""
    configs = [
        # (Sq, Sk, B, H, D, topk)
        (2048, 2048, 1, 8, 128, 1024),
        (2048, 2048, 1, 8, 128, 2048),
        (2048, 2048, 1, 32, 128, 2048),
        (8192, 8192, 1, 8, 128, 2048),
        (8192, 8192, 1, 32, 128, 2048),
        (16384, 16384, 1, 8, 128, 2048),
        (16384, 16384, 1, 32, 128, 2048),
    ]
    
    if tp_rank == 0:
        metrics_collection = []
    
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32)
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )

        attn_query = torch.zeros(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        attn_key = torch.zeros(Sk, B, H, D, device='cuda', dtype=torch.bfloat16)
        for _h in range(H):
            attn_query[:, :, _h, :] = _h
            attn_key[:, :, _h, :] = _h

        # split attn heads by TP size
        assert H % tensor_model_parallel_size == 0
        head_per_rank = H // tensor_model_parallel_size
        start_head = tp_rank * head_per_rank
        end_head = (tp_rank + 1) * head_per_rank
        attn_query_tp = attn_query[:, :, start_head:end_head, :].clone()
        attn_key_tp = attn_key[:, :, start_head:end_head, :].clone()

        softmax_scale = 1.0
        loss_coeff = 1.0
        for sparse_loss in [False, True]:
            native_lambda = lambda: forward_native(q, weights, k, mask, topk, attn_query_tp, attn_key_tp, softmax_scale, loss_coeff, sparse_loss, native_pg_collection)
            triton_acc_lambda = lambda: compute_dsa_indexer_loss_triton(q, weights, k, attn_query_tp, attn_key_tp, topk, softmax_scale, loss_coeff, mask=mask, sparse_loss=sparse_loss, pg_collection=triton_pg_collection, accuracy_check=True)
            triton_ben_lambda = lambda: compute_dsa_indexer_loss_triton(q, weights, k, attn_query_tp, attn_key_tp, topk, softmax_scale, loss_coeff, mask=mask, sparse_loss=sparse_loss, pg_collection=triton_pg_collection, accuracy_check=False)

            # check correctness
            native_topk_indices, native_indexer_loss, native_kl_per_element, native_index_scores = native_lambda()
            triton_topk_indices, triton_indexer_loss, triton_kl_per_element = triton_acc_lambda()

            match = torch.allclose(native_indexer_loss, triton_indexer_loss, atol=1e-4, rtol=1e-4)

            print(f"[Rank {tp_rank}] {Sq=}, {Sk=}, {B=}, {H=}, {D=}, {topk=} passed.")

            dist.barrier()

            pytorch_time, _, _ = bench(native_lambda)
            pytorch_time *= 1000

            triton_time, _, _ = bench(triton_ben_lambda)
            triton_time *= 1000

            speedup = pytorch_time / triton_time
            marker = "🚀" if speedup > 1.0 else ""
            sparse_str = "Yes" if sparse_loss else "No"

            if tp_rank == 0:
                metrics_collection.append((Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match))
        
    dist.barrier()
    if tp_rank == 0:
        print("\n" + "=" * 80)
        print("Benchmark: TP DSA Indexer (TopK + Loss) - PyTorch vs Triton")
        print("=" * 80)
        print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>8} {'Match':>10}")
        print("-" * 95)

        for Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match in metrics_collection:
            print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>12.2f}   {triton_time:>11.2f}   {speedup:>6.2f}x {marker} {match}")
        
        print("\n" + "=" * 95)

    dist.barrier()
    Utils.destroy_model_parallel()


def test_fused_loss_backward_native():
    """
    Test backward_native by comparing with PyTorch autograd.
    
    This test validates the manual backward implementation of the DSA indexer loss
    by comparing gradients with PyTorch's automatic differentiation.
    
    Tests multiple configurations with varying:
    - Sequence lengths (Sq, Sk)
    - Batch sizes (B)
    - Number of heads (H)
    - Head dimensions (D)
    - Top-k values
    - Sparse loss (True/False)
    
    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 80)
    print("Test: backward_native vs autograd")
    print("=" * 80)
    
    # Test configurations: (Sq, Sk, B, H, D, topk, sparse_loss)
    configs = [
        # (Sq, Sk, B, H, D, topk, sparse_loss)
        (64, 128, 1, 2, 64, 64, False),
        (64, 128, 1, 2, 64, 64, True),
        (128, 256, 2, 4, 128, 128, False),
        (128, 256, 2, 4, 128, 128, True),
    ]
    
    all_passed = True
    
    for config_idx, (Sq, Sk, B, H, D, topk, sparse_loss) in enumerate(configs):
        print(f"\n[{config_idx+1}/{len(configs)}] Testing: Sq={Sq}, Sk={Sk}, B={B}, H={H}, D={D}, topk={topk}, sparse_loss={sparse_loss}")
        
        # Create inputs for autograd path (requires_grad=True)
        torch.manual_seed(42 + config_idx)
        q_autograd = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        weights_autograd = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32, requires_grad=True)
        k_autograd = torch.randn(Sk, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Create mask
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )
        
        # Create query and key for attention
        query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.float32)
        
        softmax_scale = 1.0 / (D ** 0.5)
        loss_coeff = 0.1
        
        # Compute forward
        index_scores = _compute_index_scores(q_autograd, weights_autograd, k_autograd)
        if mask is not None:
            index_scores = index_scores + mask
        topk_indices = index_scores.topk(topk, dim=-1)[1]
        
        indexer_loss, kl_per_element = compute_dsa_indexer_loss(
            index_scores.clone(), topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss
        )
        
        # Get autograd gradients
        indexer_loss.backward()
        
        grad_q_autograd = q_autograd.grad.clone() if q_autograd.grad is not None else torch.zeros_like(q_autograd)
        grad_weights_autograd = weights_autograd.grad.clone() if weights_autograd.grad is not None else torch.zeros_like(weights_autograd)
        grad_k_autograd = k_autograd.grad.clone() if k_autograd.grad is not None else torch.zeros_like(k_autograd)
        
        # Create inputs for manual backward (no requires_grad)
        q_manual = q_autograd.detach().clone()
        weights_manual = weights_autograd.detach().clone()
        k_manual = k_autograd.detach().clone()
        
        # Compute manual gradients
        grad_loss = torch.ones_like(indexer_loss)
        grad_q_manual, grad_weights_manual, grad_k_manual = backward_native(
            q_manual, weights_manual, k_manual, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss, grad_loss
        )
        
        # Compare gradients
        rtol = 5e-2
        atol = 1e-4
        
        q_match = torch.allclose(grad_q_autograd, grad_q_manual, rtol=rtol, atol=atol)
        weights_match = torch.allclose(grad_weights_autograd, grad_weights_manual, rtol=rtol, atol=atol)
        k_match = torch.allclose(grad_k_autograd, grad_k_manual, rtol=rtol, atol=atol)
        
        # Print results
        if q_match and weights_match and k_match:
            print(f"  ✓ All gradients match! (loss={indexer_loss.item():.6f})")
        else:
            all_passed = False
            print(f"  ✗ Gradient mismatch detected:")
            if not q_match:
                q_rel_diff = (grad_q_autograd - grad_q_manual).abs() / (grad_q_autograd.abs() + 1e-8)
                print(f"    - grad_q: max_rel_diff={q_rel_diff.max():.6f}, max_abs_diff={(grad_q_autograd - grad_q_manual).abs().max():.6f}")
            if not weights_match:
                w_rel_diff = (grad_weights_autograd - grad_weights_manual).abs() / (grad_weights_autograd.abs() + 1e-8)
                print(f"    - grad_weights: max_rel_diff={w_rel_diff.max():.6f}, max_abs_diff={(grad_weights_autograd - grad_weights_manual).abs().max():.6f}")
            if not k_match:
                k_rel_diff = (grad_k_autograd - grad_k_manual).abs() / (grad_k_autograd.abs() + 1e-8)
                print(f"    - grad_k: max_rel_diff={k_rel_diff.max():.6f}, max_abs_diff={(grad_k_autograd - grad_k_manual).abs().max():.6f}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All backward_native tests passed!")
    else:
        print("✗ Some backward_native tests failed")
    print("=" * 80)
    
    return all_passed


def benchmark_fused_loss_backward():
    """
    Benchmark ONLY the backward pass: autograd vs backward_triton_full.
    
    This isolates just the backward computation by precomputing all forward values.
    """
    print("\n" + "=" * 100)
    print("Benchmark: Backward Pass ONLY - PyTorch Autograd vs Triton Full")
    print("=" * 100)
    
    # Test configurations: (Sq, Sk, B, H, D, topk, sparse_loss)
    configs = [
        # Small to medium sizes
        (64, 128, 2, 8, 128, 16, False),
        (64, 128, 2, 8, 128, 16, True),
        
        # Very large sizes
        (1024, 2048, 2, 8, 128, 256, False),
        (1024, 2048, 2, 8, 128, 256, True),

        # Huge sizes
        (4096, 4096, 2, 8, 128, 2048, False),
        (4096, 4096, 2, 8, 128, 2048, True),
        (8192, 8192, 2, 8, 128, 2048, False),
        (8192, 8192, 2, 8, 128, 2048, True),
        (16384, 16384, 1, 8, 128, 2048, False),
        (16384, 16384, 1, 8, 128, 2048, True),
    ]
    
    print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>10} {'Q Match':>7} {'W Match':>7} {'K Match':>7}")
    print("-" * 100)
    
    for Sq, Sk, B, H, D, topk, sparse_loss in configs:
        torch.manual_seed(42)
        
        # Create inputs
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.float32)
        query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.float32)
        
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )
        
        softmax_scale = 1.0 / (D ** 0.5)
        loss_coeff = 0.1
        
        # Precompute forward pass
        with torch.no_grad():
            index_scores = _compute_index_scores(q, weights, k)
            index_scores_masked = index_scores + mask
            topk_indices = index_scores_masked.topk(topk, dim=-1)[1]
            indexer_loss, _ = compute_dsa_indexer_loss(
                index_scores_masked.clone(), topk_indices, query, key,
                softmax_scale, loss_coeff, sparse_loss
            )
            grad_loss = torch.ones_like(indexer_loss)
        
        # Benchmark PyTorch autograd backward
        native_lambda = lambda: backward_native(
            q, weights, k, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss, grad_loss
        )

        triton_lambda = lambda: backward_triton_full(
            q, weights, k, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss, grad_loss
        )
        
        # Compare gradients
        grad_q_native, grad_weights_native, grad_k_native = native_lambda()
        grad_q_triton, grad_weights_triton, grad_k_triton = triton_lambda()
        
        rtol = 1e-1  # Relaxed tolerance for Triton
        atol = 1e-3

        q_match = torch.allclose(grad_q_native, grad_q_triton, rtol=rtol, atol=atol)
        weights_match = torch.allclose(grad_weights_native, grad_weights_triton, rtol=rtol, atol=atol)
        k_match = torch.allclose(grad_k_native, grad_k_triton, rtol=rtol, atol=atol)
        
        # Warmup
        for _ in range(5):
            native_lambda()
            triton_lambda()
        torch.cuda.synchronize()
        
        # Benchmark
        pytorch_time = triton.testing.do_bench(native_lambda) * 1000
        triton_time = triton.testing.do_bench(triton_lambda) * 1000
        
        speedup = pytorch_time / triton_time
        marker = "🚀" if speedup > 1.0 else "⚠️"
        sparse_str = "Yes" if sparse_loss else "No"
        
        print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>14.2f} {triton_time:>13.2f} {speedup:>7.2f}x {marker} {str(q_match):>7} {str(weights_match):>7} {str(k_match):>7}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="DSA Triton/PyTorch test harness")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--backward-native", 
        action="store_true", 
        help="Test the backward_native implementation"
    )
    group.add_argument(
        "--backward-kernel", 
        action="store_true", 
        help="Test the fully-fused Triton backward implementation"
    )
    group.add_argument(
        "--forward-kernel", 
        action="store_true", 
        help="Benchmark compute_index_scores_topk and DSA indexer loss (native vs Triton)"
    )
    group.add_argument(
        "--forward-tensor-parallel", 
        action="store_true", 
        help="Benchmark tensor parallel variant (requires torchrun with --nproc_per_node)"
    )
    args = parser.parse_args()

    any_run = False

    if args.backward_native:
        test_fused_loss_backward_native()
        any_run = True

    if args.backward_kernel:
        benchmark_fused_loss_backward()
        any_run = True

    if args.forward_kernel:
        benchmark_fused_loss_forward()
        any_run = True

    if args.forward_tensor_parallel:
        benchmark_fused_loss_backward_tensor_parallel()
        any_run = True

    if not any_run:
        print(
            "Nothing selected to run. Please specify one of the following:\n"
            "--backward-native\n"
            "--backward-kernel\n"
            "--forward-kernel\n"
            "--forward-tensor-parallel"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()