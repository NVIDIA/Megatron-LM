# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused candidate-set indexer scoring (Triton) vs the dense reference path (GPU)."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa2.candidate_kernels import (
    _candidate_scores_reference,
    candidate_scores,
    candidate_scores_available,
    candidate_topk_ids,
)
from megatron.core.transformer.experimental_attention_variant.csa2.indexer import (
    indexer_scores_rows,
)
from megatron.core.transformer.experimental_attention_variant.csa2.reference import (
    candidate_blocks_to_mask,
    indexer_topk_indices,
)


def test_candidate_kernel_geometry_guard():
    from megatron.core.transformer.experimental_attention_variant.csa2.candidate_kernels import (
        candidate_kernel_supported,
    )

    assert candidate_kernel_supported(32, 128, 8, 2048)
    assert not candidate_kernel_supported(48, 128, 8, 2048)  # heads not a power of two
    assert not candidate_kernel_supported(32, 96, 8, 2048)  # dim not a power of two
    assert not candidate_kernel_supported(32, 128, 6, 2048)  # block size
    assert not candidate_kernel_supported(32, 128, 8, 2040)  # blocks do not tile
    assert not candidate_kernel_supported(8, 128, 8, 2048)  # tl.dot minimum


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "rows,n_keys,n_blocks,block_size,topk", [(256, 8192, 128, 8, 64), (100, 3000, 64, 8, 32)]
)
def test_candidate_kernel_matches_dense_reference(rows, n_keys, n_blocks, block_size, topk):
    if not candidate_scores_available():
        pytest.skip("Triton not available")
    torch.manual_seed(0)
    dev = "cuda"
    heads, dim = 32, 128
    q = (torch.randn(rows, heads, dim, device=dev) * 0.5).to(torch.bfloat16)
    keys = (torch.randn(n_keys, dim, device=dev) * 0.5).to(torch.bfloat16)
    raw_w = torch.randn(rows, heads, device=dev).to(torch.bfloat16)
    scale = (heads * dim) ** -0.5
    total_blocks = -(-n_keys // block_size)
    # random candidate blocks per row, some rows with fewer valid blocks (-1 padding)
    perm = torch.rand(rows, total_blocks, device=dev).argsort(dim=-1)[:, :n_blocks].to(torch.int32)
    n_valid = torch.randint(n_blocks // 2, n_blocks + 1, (rows,), device=dev)
    invalid = torch.arange(n_blocks, device=dev).view(1, -1) >= n_valid.view(-1, 1)
    cand_blocks = perm.masked_fill(invalid, -1)
    visible = torch.randint(1, n_keys + 1, (rows,), device=dev)

    fused = candidate_scores(q, keys, raw_w, scale, cand_blocks, block_size, visible)
    ref = _candidate_scores_reference(q, keys, raw_w, scale, cand_blocks, block_size, visible)
    assert fused.shape == ref.shape
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(fused), finite)  # identical masks
    diff = (fused - ref).abs().masked_fill(~finite, 0.0).max().item()
    tol = 2e-2 * ref.masked_fill(~finite, 0.0).abs().amax().item() + 1e-3
    assert diff <= tol, (diff, tol)

    # top-k ids vs the dense reference pipeline (mask -> topk over the full width)
    dense = indexer_scores_rows(q, keys.float().t(), raw_w.float() * scale).unsqueeze(1)
    reachable = torch.arange(n_keys, device=dev).view(1, 1, -1) < visible.view(-1, 1, 1)
    dense = dense.masked_fill(~reachable, float("-inf"))
    mask = candidate_blocks_to_mask(cand_blocks.unsqueeze(1), n_keys, block_size)
    dense = dense.masked_fill(~mask, float("-inf"))
    ref_ids = indexer_topk_indices(dense, visible, topk).squeeze(1)
    ids = candidate_topk_ids(fused, cand_blocks, block_size, topk)
    assert ids.shape == ref_ids.shape
    # same number of valid ids per row (the fused path sorts -1 last, the dense path leaves -1
    # at the sorted position of the invalid entry; consumers treat -1 position-independently)
    assert torch.equal((ids >= 0).sum(1), (ref_ids >= 0).sum(1))
    valid_sorted = ids.masked_fill(ids < 0, 2**30).sort(1).values
    assert torch.equal(valid_sorted.masked_fill(valid_sorted == 2**30, -1), ids)
    # high overlap of the selected ids (bf16 inputs on both sides; near-ties may flip)
    overlap = []
    for a, b in zip(ids.tolist(), ref_ids.tolist()):
        sa, sb = {x for x in a if x >= 0}, {x for x in b if x >= 0}
        if sb:
            overlap.append(len(sa & sb) / len(sb))
    assert sum(overlap) / len(overlap) > 0.97, sum(overlap) / len(overlap)
