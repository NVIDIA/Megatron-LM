# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused (cuDNN DSA) vs reference dense indexer scoring of the packed CSA2 indexer (GPU)."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa2.indexer import (
    _fused_indexer_score_fn,
    fused_indexer_scores_rows,
    indexer_scores_rows,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("ratio,rows,n_keys", [(1, 512, 2048), (2, 300, 1024)])
def test_fused_matches_reference_scores(ratio, rows, n_keys):
    if _fused_indexer_score_fn() is None:
        pytest.skip("cuDNN DSA namespace not available")
    torch.manual_seed(0)
    heads, dim = 32, 128
    device = "cuda"
    q = (torch.randn(rows, heads, dim, device=device) * 0.5).to(torch.bfloat16)
    keys = (torch.randn(n_keys, dim, device=device) * 0.5).to(torch.bfloat16)
    raw_w = torch.randn(rows, heads, device=device).to(torch.bfloat16)
    scale = (dim * heads) ** -0.5
    first_position = 7
    fused = fused_indexer_scores_rows(q, keys, raw_w, scale, ratio, first_position)
    assert fused is not None and fused.shape == (rows, n_keys)
    ref = indexer_scores_rows(q, keys.float().t(), raw_w.float() * scale)
    positions = torch.arange(first_position, first_position + rows, device=device)
    visible = ((positions + 1) // ratio).clamp_max(n_keys)
    reachable = torch.arange(n_keys, device=device).view(1, -1) < visible.view(-1, 1)
    # bf16 inputs on both sides; fp32 accumulation in the kernel vs fp32 einsum per head
    diff = (fused - ref).abs().masked_fill(~reachable, 0.0)
    tol = 2e-2 * ref.abs().masked_fill(~reachable, 0.0).amax().clamp_min(1e-6)
    assert diff.max().item() < tol.item(), (diff.max().item(), tol.item())
    # top-k agreement on rows with enough reachable keys
    k = 64
    full_rows = visible >= 4 * k
    if full_rows.any():
        ref_masked = ref.masked_fill(~reachable, float("-inf"))
        fused_masked = fused.masked_fill(~reachable, float("-inf"))
        ref_top = ref_masked[full_rows].topk(k, dim=-1).indices
        fused_top = fused_masked[full_rows].topk(k, dim=-1).indices
        overlap = []
        for a, b in zip(ref_top, fused_top):
            overlap.append(len(set(a.tolist()) & set(b.tolist())) / k)
        assert sum(overlap) / len(overlap) > 0.95, sum(overlap) / len(overlap)
