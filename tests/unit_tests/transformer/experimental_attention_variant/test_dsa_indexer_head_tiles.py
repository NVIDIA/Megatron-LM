# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads", [1, 7, 17])
@pytest.mark.parametrize("budget_heads", [1, 4, 8])
@pytest.mark.parametrize("explicit_lengths", [False, True])
def test_indexer_head_tiles_match_sequential_scores(
    monkeypatch, device, dtype, heads, budget_heads, explicit_lengths
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(1234)
    b, sq, sk, hd = 2, 9, 37, 16
    # Exercise noncontiguous inputs, including a batch with a different set of keys.
    q = torch.randn(b, sq * 2, heads, hd, device=device, dtype=dtype)[:, ::2]
    k = torch.randn(b, sk * 2, 1, hd, device=device, dtype=dtype)[:, ::2]
    w = torch.randn(b, sq * 2, heads, device=device, dtype=dtype)[:, ::2]
    k_bdk = k[:, :, 0].float().transpose(1, 2).contiguous()
    row_start, ratio, scale = 5, 2, 0.37
    lengths = (
        torch.tensor([0, 1, 5, 2, 37, 40, 7, 9, 13], device=device)
        if explicit_lengths
        else (torch.arange(row_start, row_start + sq, device=device) + 1) // ratio
    )
    expected = torch.zeros(b, sq, sk, device=device)
    for head in range(heads):
        expected.add_(torch.bmm(q[:, :, head].float(), k_bdk).relu() * w[:, :, head, None].float())
    expected.mul_(scale)
    positions = torch.arange(sk, device=device)
    expected.masked_fill_(positions.view(1, 1, sk) >= lengths.view(1, sq, 1), -float("inf"))

    monkeypatch.setattr(
        dsa_cudnn_kernels, "_INDEXER_SCORE_HEAD_TILE_MAX_BYTES", b * sq * sk * 4 * budget_heads
    )
    original_bmm = torch.bmm
    gemm_rows = []

    def bounded_bmm(query, key):
        gemm_rows.append(query.size(1))
        assert query.size(1) <= sq * min(heads, budget_heads)
        assert key.shape == (b, hd, sk)
        return original_bmm(query, key)

    monkeypatch.setattr(torch, "bmm", bounded_bmm)
    actual = dsa_cudnn_kernels._compute_indexer_scores_chunk_with_global_rows(
        q,
        k,
        w,
        row_start=row_start,
        indexer_ratio=ratio,
        sm_scale=scale,
        seq_lens=lengths if explicit_lengths else None,
        k_bdk=k_bdk if explicit_lengths else None,
        key_positions=positions if explicit_lengths else None,
    )
    assert actual.dtype == torch.float32
    tile = min(heads, budget_heads)
    assert len(gemm_rows) == (heads + tile - 1) // tile
    assert sum(gemm_rows) == sq * heads
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    # Check discrete top-k membership on rows with at least three valid keys.
    valid_rows = lengths >= 3
    torch.testing.assert_close(
        actual[:, valid_rows].topk(3).indices, expected[:, valid_rows].topk(3).indices
    )


@pytest.mark.parametrize(
    "b,sq,sk,heads,expected",
    [(1, 512, 4096, 64, 8), (2, 512, 16384, 64, 4), (1, 8192, 65536, 64, 1), (1, 1, 1, 3, 3)],
)
def test_indexer_head_tile_budget(b, sq, sk, heads, expected):
    tile = dsa_cudnn_kernels._indexer_score_head_tile_size(b, sq, sk, heads)
    assert tile == expected
    assert (
        tile == 1 or b * sq * sk * tile * 4 <= dsa_cudnn_kernels._INDEXER_SCORE_HEAD_TILE_MAX_BYTES
    )
