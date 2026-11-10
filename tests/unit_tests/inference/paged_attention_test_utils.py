# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference paged attention for checking varlen flash-attention outputs in tests."""

import torch


def reference_paged_attention(q, k_cache, v_cache, cu_seqlens_q, kv_lengths, block_table):
    """Naive causal attention over a paged KV cache, one sequence at a time.

    Args:
        q: [total_q, num_heads, head_dim] queries; rows past cu_seqlens_q[-1] are ignored.
        k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim].
        cu_seqlens_q: [num_seqs + 1] cumulative query lengths.
        kv_lengths: [num_seqs] KV length per sequence (queries are its last tokens).
        block_table: [num_seqs, max_blocks] block ids per sequence.

    Returns:
        [total_q, num_heads, head_dim] float32 output; rows of zero-length or padded
        sequences are zero.
    """
    num_heads, head_dim = q.shape[1], q.shape[2]
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = num_heads // num_kv_heads
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    cu = cu_seqlens_q.tolist()
    for i, kv_len in enumerate(kv_lengths.tolist()):
        q_len = cu[i + 1] - cu[i]
        if q_len == 0:
            continue
        num_blocks = (kv_len + block_size - 1) // block_size
        blocks = block_table[i, :num_blocks].long()
        k = k_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        v = v_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[cu[i] : cu[i + 1]].float(), k) * head_dim**-0.5
        q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)[:, None]
        k_pos = torch.arange(kv_len, device=q.device)[None, :]
        scores = scores.masked_fill((k_pos > q_pos)[None], float("-inf"))
        out[cu[i] : cu[i + 1]] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    return out
