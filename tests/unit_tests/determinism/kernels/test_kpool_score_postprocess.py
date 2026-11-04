# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Replay fused detached KPool postprocessing under side-stream contention."""

import pytest
import torch

from megatron.core.ops.attention.dsa.backends import select_kpool_score_postprocess
from megatron.core.transformer.experimental_attention_variant import dsa
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


@pytest.mark.parametrize("use_relu", [False, True])
@pytest.mark.parametrize("batch_major", [False, True])
@pytest.mark.parametrize("keys", [1, 513])
def test_kpool_score_postprocess_replays_bit_exact(use_relu, batch_major, keys):
    pytest.importorskip("triton")
    implementation = select_kpool_score_postprocess("triton", 32)
    seeded(37)
    scores = torch.randn(67, 2, 32, keys, device="cuda")
    if batch_major:
        scores = scores.transpose(0, 1).contiguous().transpose(0, 1)
    weights = torch.randn(67, 2, 64, device="cuda")[..., ::2]

    assert_replays_bit_exact(
        lambda scores, weights: implementation(scores, weights, use_relu),
        {"scores": scores, "weights": weights},
        replays=5,
        backward=False,
        contention=True,
    )


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("use_relu", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_kpool_selection_replays_bit_exact(backend, packed, use_relu, tail, monkeypatch):
    if backend == "triton":
        pytest.importorskip("triton")
    implementation = select_kpool_score_postprocess(backend, 32)
    seeded(43)
    queries, keys, batch, dim = 11, 37, 1 if packed else 2, 128
    q = torch.randn(queries, batch, 32, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(queries, batch, 32, device="cuda")
    positions = torch.arange(queries, device="cuda") * 3
    kwargs = dict(
        index_topk=8,
        pool_size=4,
        use_relu=use_relu,
        always_select_tail=tail,
        return_index_scores=False,
        score_postprocess=implementation,
    )
    if packed:
        kwargs.update(
            cu_seqlens_kv=torch.tensor([0, 13, 37], device="cuda", dtype=torch.int32),
            varlen_starts=torch.where(positions < 13, 0, 13),
            varlen_ends=positions + 1,
            key_positions=torch.arange(keys, device="cuda"),
        )
    else:
        mask = torch.full((batch, queries, keys), -float("inf"), device="cuda")
        for i, end in enumerate(positions.tolist()):
            mask[:, i, : end + 1] = 0.0
        mask[1, :, 2] = -float("inf")
        kwargs["mask"] = mask
    monkeypatch.setattr(dsa, "_KPOOL_SCORE_CHUNK_BYTES", batch * 32 * 9 * 4 * 3)

    def select(q, k, weights, gate_score, ape):
        with torch.no_grad():
            scores, indices = dsa.fused_qk_topk_kpool(
                q, k, weights, gate_score=gate_score, ape=ape, **kwargs
            )
        assert scores is None
        return indices

    assert_replays_bit_exact(
        select,
        {
            "q": q,
            "k": k,
            "weights": weights,
            "gate_score": torch.randn_like(k),
            "ape": torch.randn(4, dim, device="cuda"),
        },
        replays=5,
        backward=False,
        contention=True,
    )
