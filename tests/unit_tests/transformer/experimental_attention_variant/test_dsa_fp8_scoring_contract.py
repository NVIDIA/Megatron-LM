# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels as kernels


def _operands():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Native FP8 folded-weight contract requires SM90")
    kernels._ensure_dsa_namespace()
    torch.manual_seed(17)
    q = torch.randn(1, 256, 32, 128, device="cuda")
    k = torch.randn(1, 256, 1, 128, device="cuda")
    raw = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
    return q, k, raw


def test_fp8_scoring_preserves_normalized_weights_and_replays():
    q, k, raw = _operands()
    q8, qs = kernels._fp8_ue8m0_quantize_last_dim(q)
    k8, ks = kernels._fp8_ue8m0_quantize_last_dim(k)
    # The working native scorer normalizes the original projection in FP32.
    expected = kernels._cudnn_dsa.indexer_forward_wrapper(
        q8,
        k8,
        (raw.float() * qs * (128**-0.5) * (32**-0.5)).contiguous(),
        ratio=1,
        sm_scale=1.0,
        precision="fp8",
        q_scale=torch.ones_like(qs),
        k_scale=ks,
        q_causal_offsets=torch.zeros(1, device="cuda", dtype=torch.int32),
    )["scores"]
    weights = raw.float() * (32**-0.5) * (128**-0.5)
    for _ in range(2):
        actual = kernels._indexer_forward(
            q, k, weights, ratio=1, sm_scale=1.0, scoring_precision="fp8"
        )["scores"]
        assert torch.equal(actual, expected)


def test_fp8_chunk_and_bottom_right_match_dense():
    q, k, raw = _operands()
    weights = raw.float() * (32**-0.5) * (128**-0.5)
    dense = kernels._indexer_forward(q, k, weights, ratio=1, sm_scale=1.0, scoring_precision="fp8")[
        "scores"
    ]
    for start, end in [(0, 64), (64, 128), (128, 256)]:
        chunk = kernels._compute_indexer_scores_chunk_with_global_rows(
            q[:, start:end].contiguous(),
            k,
            weights[:, start:end].contiguous(),
            row_start=start,
            indexer_ratio=1,
            sm_scale=1.0,
            scoring_precision="fp8",
        )
        torch.testing.assert_close(chunk, dense[:, start:end], rtol=0, atol=0)
    tail = kernels._indexer_forward(
        q[:, 128:].contiguous(),
        k,
        weights[:, 128:].contiguous(),
        ratio=1,
        sm_scale=1.0,
        scoring_precision="fp8",
    )["scores"]
    torch.testing.assert_close(tail, dense[:, 128:], rtol=0, atol=0)


def test_fp8_explicit_bounds_reset_and_empty_rows():
    q, k, raw = _operands()
    weights = raw.float() * (32**-0.5) * (128**-0.5)
    bounds = torch.cat([torch.zeros(4), torch.arange(1, 125), torch.arange(1, 129)]).to(
        device="cuda", dtype=torch.int64
    )
    actual = kernels._compute_indexer_scores_chunk_with_global_rows(
        q,
        k,
        weights,
        row_start=0,
        indexer_ratio=1,
        sm_scale=1.0,
        seq_lens=bounds,
        scoring_precision="fp8",
    )
    assert torch.isneginf(actual[:, :4]).all()
    for start, end in [(4, 128), (128, 256)]:
        expected = kernels._indexer_forward(
            q[:, start:end].contiguous(),
            k,
            weights[:, start:end].contiguous(),
            ratio=1,
            sm_scale=1.0,
            scoring_precision="fp8",
            q_causal_offsets=torch.zeros(1, device="cuda", dtype=torch.int32),
        )["scores"]
        torch.testing.assert_close(actual[:, start:end], expected, rtol=0, atol=0)


def test_bf16_dispatch_matches_original_kernel():
    q, k, raw = _operands()
    q, k = q.bfloat16(), k.bfloat16()
    expected = kernels._cudnn_dsa.indexer_forward_wrapper(q, k, raw, ratio=1, sm_scale=1.0)[
        "scores"
    ]
    actual = kernels._indexer_forward(q, k, raw, ratio=1, sm_scale=1.0)["scores"]
    assert torch.equal(actual, expected)


def test_fp8_packed_thd_bottom_right_offsets():
    q, k, raw = _operands()
    weights = raw.float() * (32**-0.5) * (128**-0.5)
    cu_q = torch.tensor([0, 32, 80], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 64, 144], device="cuda", dtype=torch.int32)
    actual = kernels._indexer_forward(
        q[0, :80].contiguous(),
        k[0, :144].contiguous(),
        weights[0, :80].contiguous(),
        ratio=1,
        sm_scale=1.0,
        scoring_precision="fp8",
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=48,
        max_seqlen_k=80,
    )["scores"]
    for qstart, qend, kstart, kend in [(0, 32, 0, 64), (32, 80, 64, 144)]:
        expected = kernels._indexer_forward(
            q[:, qstart:qend].contiguous(),
            k[:, kstart:kend].contiguous(),
            weights[:, qstart:qend].contiguous(),
            ratio=1,
            sm_scale=1.0,
            scoring_precision="fp8",
        )["scores"]
        torch.testing.assert_close(
            actual[qstart:qend, : kend - kstart], expected[0], rtol=0, atol=0
        )
        assert torch.isneginf(actual[qstart:qend, kend - kstart :]).all()


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
@pytest.mark.parametrize("packed", [False, True])
def test_indexer_rope_preserves_adjacent_layout_for_fp8(precision, packed):
    from types import SimpleNamespace

    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer

    if not torch.cuda.is_available():
        pytest.skip("Indexer RoPE regression requires CUDA")
    torch.manual_seed(19)
    x = torch.randn(64, 1, 32, 128, device="cuda")
    angles = torch.randn(48 if packed else 64, 1, 1, 32, device="cuda")
    freqs = torch.cat((angles, angles), dim=-1)
    cu_seqlens = torch.tensor([0, 16, 64], device="cuda", dtype=torch.int32) if packed else None
    positions = torch.cat((torch.arange(16), torch.arange(48))) if packed else torch.arange(64)
    cos, sin = angles[positions].cos(), angles[positions].sin()
    even, odd = x[..., :64:2], x[..., 1:64:2]
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin
    if precision == "fp8":
        expected_pe = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
    else:
        # Preserve the existing BF16 scoring path's grouped even/odd layout.
        expected_pe = torch.cat((rotated_even, rotated_odd), dim=-1)
    expected = torch.cat((expected_pe, x[..., 64:]), dim=-1)
    indexer = SimpleNamespace(
        use_rope=True,
        qk_pos_emb_head_dim=64,
        index_head_dim=128,
        config=SimpleNamespace(
            apply_rope_fusion=False,
            rotary_interleaved=False,
            dsa_indexer_rope_interleaved=True,
            dsa_indexer_rope_fp32=True,
            dsa_indexer_scoring_precision=precision,
        ),
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0)),
    )
    actual = DSAIndexer._apply_rope(indexer, x, freqs, 1.0, cu_seqlens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
