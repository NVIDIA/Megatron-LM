# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Window/sink normalization for CSA2's bounded-workspace indexer objective."""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _window_lse(
        q,
        kv,
        sink,
        indices,
        output,
        q_stride_row: tl.constexpr,
        q_stride_head: tl.constexpr,
        kv_stride_row: tl.constexpr,
        index_stride: tl.constexpr,
        heads: tl.constexpr,
        dim: tl.constexpr,
        key_rows: tl.constexpr,
        window: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)
        d = tl.arange(0, BLOCK_D)
        query = tl.load(
            q + row * q_stride_row + h[:, None] * q_stride_head + d[None, :],
            (h[:, None] < heads) & (d[None, :] < dim),
            other=0,
        )
        maximum = tl.load(sink + h, h < heads, other=0).to(tl.float32)
        total = tl.full((BLOCK_H,), 1.0, tl.float32)
        for start in range(tl.cdiv(window, BLOCK_K)):
            slot = start * BLOCK_K + tl.arange(0, BLOCK_K)
            key = tl.load(indices + row * index_stride + slot, slot < window, other=-1)
            valid = (slot < window) & (key >= 0) & (key < key_rows)
            values = tl.load(
                kv + key[:, None] * kv_stride_row + d[None, :],
                valid[:, None] & (d[None, :] < dim),
                other=0,
            )
            scores = tl.dot(query, tl.trans(values), allow_tf32=False) * scale
            scores = tl.where(valid[None, :], scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 1))
            total = total * tl.exp(maximum - next_maximum) + tl.sum(
                tl.exp(scores - next_maximum[:, None]), 1
            )
            maximum = next_maximum
        tl.store(output + row * heads + h, maximum + tl.log(total), h < heads)


@torch.no_grad()
def fused_csa_window_lse(query, local_kv, sink, window_indices, softmax_scale):
    """Compute [query rows, heads] logsumexp over valid window keys and the sink."""
    rows, heads, dim = query.shape
    if rows == 0:
        return query.new_empty((0, heads), dtype=torch.float32)
    if not query.is_cuda:
        # Small CPU numerical checks use the same physical-index contract.
        output = []
        for start in range(0, rows, 16):
            ids = window_indices[start : start + 16]
            keys = local_kv[ids.clamp_min(0).long()].float()
            scores = torch.einsum("qhd,qkd->qhk", query[start : start + 16].float(), keys)
            scores = (scores * softmax_scale).masked_fill(ids[:, None, :] < 0, -torch.inf)
            output.append(torch.logaddexp(scores.logsumexp(-1), sink.float()[None, :]))
        return torch.cat(output)
    if not HAVE_TRITON:
        raise RuntimeError("CSA2 fused window LSE requires Triton")
    output = query.new_empty((rows, heads), dtype=torch.float32)
    _window_lse[(rows,)](
        query,
        local_kv,
        sink,
        window_indices,
        output,
        query.stride(0),
        query.stride(1),
        local_kv.stride(0),
        window_indices.stride(0),
        heads,
        dim,
        local_kv.shape[0],
        window_indices.shape[-1],
        softmax_scale,
        max(16, triton.next_power_of_2(heads)),
        triton.next_power_of_2(dim),
        32,
        num_warps=8,
    )
    return output
