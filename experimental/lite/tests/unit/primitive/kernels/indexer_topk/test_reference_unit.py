# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the matched-precision reference selector.

The DeepGEMM score kernel is replaced by a float64 emulation over the same quantized operands
and the top-k kernel by an exact sort, so these tests check the batching around them: row
windows, row-relative scores, chunking, id offsets, head padding and the head-count probe.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import (
    IndexerTopKConfigError,
    IndexerTopKRuntimeError,
    QueryLayout,
    QuerySegment,
    quantize_indexer_fp8_rows,
)
from megatron.lite.primitive.kernels.indexer_topk import reference as reference_module
from megatron.lite.primitive.kernels.indexer_topk import (
    reference_topk,
    sort_topk_rows_,
)
from megatron.lite.primitive.kernels.indexer_topk.reference import (
    ReferenceSelector,
    quantize_keys,
    quantize_queries,
    score_kernel_heads,
)

pytestmark = pytest.mark.mlite


def _operands(q, kv, weights):
    """float64 queries [n, H, D], keys [N, D] and per-(row, head, key) scale folding."""
    (q_data, q_sf), (k_data, k_scale) = q, kv
    assert q_sf is None  # FP8: the query scales are folded into the weights, key scales outside.
    return q_data.double(), k_data.double(), k_scale.double()


class FakeScoreKernel:
    """float64 emulation of DeepGEMM fp8_fp4_mqa_logits (plain and row-relative score rows)."""

    def __init__(self, supported_heads=None):
        self.calls = []
        self.supported_heads = supported_heads

    def __call__(self, q, kv, weights, ks, ke, *, max_seqlen_k):
        heads = q[0].shape[1]
        if self.supported_heads is not None and heads not in self.supported_heads:
            raise RuntimeError(
                "Assertion error (attention.hpp:120): num_heads == 16 or num_heads == 32"
            )
        self.calls.append(
            dict(
                rows=q[0].shape[0],
                heads=heads,
                keys=kv[0].shape[0],
                ks=ks.tolist(),
                ke=ke.tolist(),
                width=max_seqlen_k,
                q=q,
                weights=weights,
                scale_aligned=kv[1].data_ptr() % 16 == 0,
                scale_is_view=kv[1]._base is not None,
            )
        )
        queries, keys, key_scale = _operands(q, kv, weights)
        # max_seqlen_k == 0: plain rows, one column per key; otherwise row-relative columns.
        columns = max_seqlen_k or keys.shape[0]
        stride = columns + 5  # padded rows, as the kernel allocates them
        logits = torch.full((queries.shape[0], stride), float("nan"), dtype=torch.float32)
        for row in range(queries.shape[0]):
            start, end = int(ks[row]), int(ke[row])
            if end > start:
                dots = torch.relu(queries[row] @ keys[start:end].T)  # [H, keys]
                scores = (dots * weights[row].double()[:, None]).sum(0)
                if key_scale is not None:
                    scores = scores * key_scale[start:end]
                first = start if max_seqlen_k == 0 else 0
                logits[row, first : first + end - start] = scores.float()
        return logits[:, :columns]


def exact_topk_kernel(scores, lengths, top_k):
    """Score descending, lower column first on equal scores; -1 for missing columns."""
    ids = torch.full((scores.shape[0], top_k), -1, dtype=torch.int32)
    for row in range(scores.shape[0]):
        length = int(lengths[row])
        order = torch.sort(-scores[row, :length].double(), stable=True).indices[:top_k]
        ids[row, : order.numel()] = order.int()
    return ids


def _expected(q, k, weights, layout, topk, softmax_scale):
    """Exact top-k of every row over its visible keys, from the quantized operands."""
    q_data, q_scale = quantize_indexer_fp8_rows(q)
    k_data, k_scale = quantize_indexer_fp8_rows(k)
    queries, keys = q_data.double(), k_data.double()
    head_weights = (weights.float() * softmax_scale * q_scale).double()
    out = torch.full((layout.rows, topk), -1, dtype=torch.int32)
    for segment in layout.segments:
        for row in range(segment.row_start, segment.row_end):
            visible = layout.visible_keys(segment, row)
            window = keys[segment.key_start : segment.key_start + visible]
            scores = (torch.relu(queries[row] @ window.T) * head_weights[row][:, None]).sum(0)
            scores = scores * k_scale[segment.key_start : segment.key_start + visible].double()
            order = torch.sort(-scores.float().double(), stable=True).indices[:topk]
            out[row, : order.numel()] = order.int() + segment.index_base
    return sort_topk_rows_(out)


def _inputs(rows, keys, heads, seed):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((rows, heads, 128), generator=generator).to(torch.bfloat16)
    k = torch.randn((keys, 128), generator=generator).to(torch.bfloat16)
    # Few distinct key rows make exact score ties at the cutoff common.
    k[keys // 2 :] = k[: keys - keys // 2]
    weights = torch.randn((rows, heads), generator=generator).to(torch.bfloat16)
    return q, k, weights


def _selector(fmt, kernel_heads, **kwargs):
    return ReferenceSelector(
        fmt=fmt,
        topk_kernel=exact_topk_kernel,
        kernel_heads=kernel_heads,
        budget_bytes=kwargs.pop("budget_bytes", 1 << 30),
        rows_per_call=kwargs.pop("rows_per_call", None),
        num_sms=kwargs.pop("num_sms", 4),
    )


def _select(selector, q, k, weights, layout, topk, softmax_scale, row_ranges=None):
    """Select into a destination prefilled with 99; also returns the valid ids per row."""
    keys = quantize_keys(k, selector.fmt)
    out = torch.full((layout.rows, topk), 99, dtype=torch.int32)
    calls = selector.select(
        q,
        weights,
        keys,
        layout=layout,
        row_ranges=((0, layout.rows),) if row_ranges is None else row_ranges,
        topk=topk,
        softmax_scale=softmax_scale,
        out=out,
    )
    lengths = ((out >= 0) & (out != 99)).sum(1, dtype=torch.int32)
    return out, lengths, calls


def test_select_batches_segments_into_varlen_calls(monkeypatch):
    kernel = FakeScoreKernel()
    monkeypatch.setattr(reference_module, "_mqa_logits", kernel)
    cu_seqlens = [0, 23, 23, 71, 90]
    layout = QueryLayout.packed(cu_seqlens, row_start=10, rows=84, absolute_ids=True)
    q, k, weights = _inputs(84, cu_seqlens[-1], 16, seed=1)
    out, lengths, calls = _select(_selector("fp8", 16), q, k, weights, layout, 8, 0.25)

    expected = _expected(q, k, weights, layout, 8, 0.25)
    assert torch.equal(sort_topk_rows_(out.clone()), expected)
    assert torch.equal(lengths, (expected >= 0).sum(1, dtype=torch.int32))
    assert (out[80:] == -1).all() and (lengths[80:] == 0).all()  # padding rows past the pack
    # One score call spans all three sequences, each row with its own key window.
    (call,) = kernel.calls
    assert calls == 1 and call["rows"] == 80
    widths = [
        layout.visible_keys(s, r) for s in layout.segments for r in range(s.row_start, s.row_end)
    ]
    starts = [s.key_start for s in layout.segments for _ in range(s.rows)]
    assert call["ks"] == starts and call["ke"] == [a + b for a, b in zip(starts, widths)]
    assert call["width"] == max(widths) and call["keys"] == max(call["ke"])


def test_select_chunking_invariant(monkeypatch):
    fmt = "fp8"
    monkeypatch.setattr(reference_module, "_mqa_logits", FakeScoreKernel())
    layout = QueryLayout.contiguous(96, position=40, keys=160)
    q, k, weights = _inputs(96, 160, 16, seed=2)
    reference, reference_lengths, calls = _select(
        _selector(fmt, 16), q, k, weights, layout, 12, 1.0
    )
    assert calls == 1
    for kwargs, expected_calls in (
        (dict(budget_bytes=4 * 136 * 12), 8),  # 12 rows of the widest row's 136 keys per call
        (dict(rows_per_call=7), 14),
        (dict(rows_per_call=7, budget_bytes=4 * 136 * 5), 24),  # the budget bounds rows_per_call
    ):
        out, lengths, calls = _select(_selector(fmt, 16, **kwargs), q, k, weights, layout, 12, 1.0)
        assert calls == expected_calls
        assert torch.equal(out, reference) and torch.equal(lengths, reference_lengths)
    # Only the requested rows are written.
    out, lengths, _ = _select(
        _selector(fmt, 16), q, k, weights, layout, 12, 1.0, ((5, 9), (60, 61))
    )
    written = torch.zeros(96, dtype=torch.bool)
    written[5:9] = written[60] = True
    assert torch.equal(out[written], reference[written]) and (out[~written] == 99).all()
    assert (lengths[~written] == 0).all()


def test_select_empty_windows_and_uncovered_rows(monkeypatch):
    kernel = FakeScoreKernel()
    monkeypatch.setattr(reference_module, "_mqa_logits", kernel)
    layout = QueryLayout(
        rows=12,
        segments=(
            QuerySegment(2, 5, 0, 0, 0, 0),
            QuerySegment(5, 6, 0, 0, 1, 0),
            QuerySegment(8, 10, 0, 10, 0, 10),
        ),
    )
    q, k, weights = _inputs(12, 20, 16, seed=3)
    out, lengths, calls = _select(_selector("fp8", 16), q, k, weights, layout, 4, 1.0)
    # Every covered row sees at most one key; rows 2-4 and 8-9 see none.
    assert calls == 1 and kernel.calls[0]["width"] == 1
    assert out[5].tolist() == [0, -1, -1, -1] and lengths[5] == 1
    assert (out[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]] == -1).all()
    assert lengths.sum() == 1
    # A single segment with an id offset: its rows are scored against a view of its keys (plain
    # score rows, windows from column 0) and its ids are returned as key tensor rows.
    layout = QueryLayout(rows=6, segments=(QuerySegment(0, 6, 2, 8, 9, 8),))
    out, lengths, calls = _select(_selector("fp8", 16), q[:6], k, weights[:6], layout, 4, 1.0)
    call = kernel.calls[-1]
    assert calls == 1 and (call["width"], call["keys"]) == (0, 8)
    assert call["ks"] == [0] * 6 and call["ke"] == [3, 4, 5, 6, 7, 8]
    expected = _expected(q[:6], k, weights[:6], layout, 4, 1.0)
    assert torch.equal(sort_topk_rows_(out), expected)
    assert int(expected[expected >= 0].min()) >= 8 and lengths.tolist() == [3, 4, 4, 4, 4, 4]
    # The kernel needs 16-byte aligned operands: the 4-byte scales of a key view that starts at
    # key 7 are not, so the selector scores against an aligned copy of them, still in the plain
    # mode (the key rows of the view are aligned).
    layout = QueryLayout(rows=6, segments=(QuerySegment(0, 6, 2, 7, 9, 7),))
    out, lengths, calls = _select(_selector("fp8", 16), q[:6], k, weights[:6], layout, 4, 1.0)
    call = kernel.calls[-1]
    assert (call["width"], call["keys"]) == (0, 8) and call["ks"] == [0] * 6
    assert call["ke"] == [3, 4, 5, 6, 7, 8]
    assert call["scale_aligned"] and not call["scale_is_view"]
    assert torch.equal(sort_topk_rows_(out), _expected(q[:6], k, weights[:6], layout, 4, 1.0))
    # A view whose scales are aligned is used in place.
    layout = QueryLayout(rows=6, segments=(QuerySegment(0, 6, 2, 8, 9, 8),))
    _select(_selector("fp8", 16), q[:6], k, weights[:6], layout, 4, 1.0)
    assert kernel.calls[-1]["scale_is_view"] and kernel.calls[-1]["width"] == 0
    # Rows that see no key at all need no score call.
    layout = QueryLayout(rows=3, segments=(QuerySegment(0, 3, 0, 0, 0, 0),))
    out, lengths, calls = _select(_selector("fp8", 16), q[:3], k, weights[:3], layout, 4, 1.0)
    assert calls == 0 and (out == -1).all() and (lengths == 0).all()


def test_select_pads_heads(monkeypatch):
    kernel = FakeScoreKernel()
    monkeypatch.setattr(reference_module, "_mqa_logits", kernel)
    layout = QueryLayout.full(40, keys=40)
    q, k, weights = _inputs(40, 40, 12, seed=4)
    out, lengths, _ = _select(_selector("fp8", 32), q, k, weights, layout, 6, 0.5)
    assert torch.equal(sort_topk_rows_(out), _expected(q, k, weights, layout, 6, 0.5))
    (call,) = kernel.calls
    q_data, q_sf = call["q"]
    assert call["heads"] == 32 and (call["weights"][:, 12:] == 0).all()
    assert (q_data[:, 12:].view(torch.uint8) == 0).all() and q_sf is None


def test_rows_per_scoring_call():
    fp8 = _selector("fp8", 32, budget_bytes=2 << 30, num_sms=148)
    assert [fp8.rows_per_scoring_call(keys) for keys in (262144, 524288, 188416)] == [
        1776,
        592,
        2368,
    ]
    assert fp8.rows_per_scoring_call(16) == 32768  # the top-k kernel row limit
    # Explicit rows per call replace the wave plan and are bounded by the budget.
    h64 = _selector("fp8", 64, budget_bytes=1 << 30, rows_per_call=4096, num_sms=148)
    assert h64.rows_per_scoring_call(65536) == 4096
    assert h64.rows_per_scoring_call(131072) == 2048
    h64 = _selector("fp8", 64, budget_bytes=1 << 30, num_sms=148)
    assert h64.rows_per_scoring_call(65536) == 3848


def test_quantize_keys_chunks_rows(monkeypatch):
    k = torch.randn((50, 128), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    whole = quantize_keys(k, "fp8")
    monkeypatch.setattr(reference_module, "_KEY_QUANTIZE_ROWS", 7)
    chunked = quantize_keys(k, "fp8")
    assert torch.equal(chunked.data.view(torch.uint8), whole.data.view(torch.uint8))
    assert torch.equal(chunked.scale, whole.scale)
    prefix = quantize_keys(k, "fp8", rows=20)
    assert torch.equal(prefix.data.view(torch.uint8), whole.data[:20].view(torch.uint8))
    fp8_data, fp8_scale = quantize_indexer_fp8_rows(k)
    assert torch.equal(whole.data.view(torch.uint8), fp8_data.view(torch.uint8))
    assert torch.equal(whole.scale, fp8_scale)


def test_score_kernel_heads_probe(monkeypatch):
    kernel = FakeScoreKernel(supported_heads={16, 32, 64})
    monkeypatch.setattr(reference_module, "_mqa_logits", kernel)
    monkeypatch.setattr(reference_module, "_HEAD_SUPPORT", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    cpu = torch.device("cpu")
    assert score_kernel_heads(32, fmt="fp8", head_dim=128, device=cpu) == 32
    assert score_kernel_heads(48, fmt="fp8", head_dim=128, device=cpu) == 64
    assert score_kernel_heads(8, fmt="fp8", head_dim=128, device=cpu) == 16
    probes = len(kernel.calls)
    assert score_kernel_heads(48, fmt="fp8", head_dim=128, device=cpu) == 64
    assert len(kernel.calls) == probes  # cached
    with pytest.raises(IndexerTopKConfigError, match="supports no head count"):
        score_kernel_heads(72, fmt="fp8", head_dim=128, device=cpu)

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(reference_module, "_mqa_logits", broken)
    with pytest.raises(IndexerTopKRuntimeError, match="probing the head counts"):
        score_kernel_heads(24, fmt="fp8", head_dim=128, device=cpu)


def test_reference_topk_validation():
    layout = QueryLayout.full(4, keys=4)
    q, k, weights = torch.zeros((4, 16, 128)), torch.zeros((4, 128)), torch.zeros((4, 16))
    with pytest.raises(ValueError, match="CUDA"):
        reference_topk(q, k, weights, layout=layout, topk=2, softmax_scale=1.0, fmt="fp8")
    with pytest.raises(ValueError, match="fmt"):
        reference_topk(q, k, weights, layout=layout, topk=2, softmax_scale=1.0, fmt="fp4")
    with pytest.raises(ValueError, match="fmt"):
        quantize_queries(q, weights, "bf16", softmax_scale=1.0, kernel_heads=16)
