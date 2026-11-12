# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The matched-precision reference selector on a Blackwell GPU (optional).

Needs DeepGEMM. The exact-tie top-k package is not part of Megatron Lite; the tests that use it
skip unless its location is given as JSON, inline or as the path of a JSON file::

    LITETOPK_TEST_EXACT_TOPK='{"source": "/path/to/exact-tie", "pythonpath": ["/path/to/cudnn"]}' \\
    experimental/lite/tests/run_tests.sh \\
        experimental/lite/tests/smoke/primitive/indexer_topk/test_reference_gpu.py

The entry holds the ``ExactTopKConfig`` fields plus optional ``pythonpath`` entries to prepend,
for example a cuDNN frontend that provides the DeepSeek sparse attention compiler.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import (
    ExactTopKConfig,
    QueryLayout,
    QuerySegment,
    quantize_indexer_fp8_rows,
    reference_topk,
    sort_topk_rows_,
)

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]

_EXACT_VARIABLE = "LITETOPK_TEST_EXACT_TOPK"


@pytest.fixture(scope="module")
def exact_topk() -> ExactTopKConfig:
    raw = os.environ.get(_EXACT_VARIABLE, "").strip()
    if not raw:
        pytest.skip(f"{_EXACT_VARIABLE} is not set")
    spec = json.loads(raw if raw[0] == "{" else Path(raw).read_text(encoding="utf-8"))
    for entry in reversed(spec.pop("pythonpath", [])):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return ExactTopKConfig(**spec)


@pytest.fixture(autouse=True)
def _deep_gemm():
    pytest.importorskip("deep_gemm")


def _exact_inputs(rows: int, keys: int, heads: int, seed: int):
    """Queries, keys and weights whose quantized scores are exact multiples of 1/8.

    FP8 rows hold small integers plus one entry of 448 (so the row scale is exactly 1), in
    different columns for queries and keys. Every product and sum is then exact in float32, so
    the float64 reference below is exactly what DeepGEMM computes, and the small value range
    makes equal scores frequent.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def draw(count: int, hot_column: int) -> torch.Tensor:
        values = torch.randint(-3, 4, (count, 128), generator=generator, device="cuda")
        values[:, hot_column] = 448
        return values.float()

    q = draw(rows * heads, 0).reshape(rows, heads, 128).to(torch.bfloat16)
    k = draw(keys, 1).to(torch.bfloat16)
    weights = torch.randint(-2, 5, (rows, heads), generator=generator, device="cuda").float()
    return q, k, weights


def _dequantize(value: torch.Tensor) -> torch.Tensor:
    data, scale = quantize_indexer_fp8_rows(value)
    return data.double() * scale.double()[..., None]


def _expected(q, k, weights, layout, topk, softmax_scale, ties=None):
    """Exact top-k (score descending, lower id first) in float64 over the quantized operands.

    ``ties`` (a list) receives the number of rows whose cutoff falls between equal scores.
    """
    queries, keys = _dequantize(q), _dequantize(k)
    out = torch.full((layout.rows, topk), -1, dtype=torch.int32, device=q.device)
    for segment in layout.segments:
        window = keys[segment.key_start : segment.key_start + segment.key_count]
        columns = torch.arange(segment.key_count, device=q.device)
        for first in range(segment.row_start, segment.row_end, 256):
            rows = torch.arange(first, min(first + 256, segment.row_end), device=q.device)
            scores = torch.einsum("rhd,kd->rhk", queries[rows], window).relu()
            scores = (scores * (weights[rows].double() * softmax_scale)[..., None]).sum(1)
            positions = segment.position + rows - segment.row_start
            visible = (positions + 1).clamp(max=segment.key_count)
            scores = scores.masked_fill(columns[None, :] >= visible[:, None], float("-inf"))
            # A stable sort by descending score keeps equal scores in ascending id order.
            ranked, order = torch.sort(-scores, dim=1, stable=True)
            if ties is not None and ranked.shape[1] > topk:
                cut = visible > topk
                ties.append(int((cut & (ranked[:, topk - 1] == ranked[:, topk])).sum()))
            order = order[:, :topk]
            ids = torch.where(order < visible[:, None], order + segment.index_base, -1)
            out[rows, : order.shape[1]] = ids.int()
    return sort_topk_rows_(out)


def _layouts():
    return {
        "full": (QueryLayout.full(3000, keys=3000), 3000),
        "cp_shard": (QueryLayout.contiguous(1100, position=2900, keys=4000), 4000),
        "packed": (
            QueryLayout.packed(
                [0, 700, 701, 2400, 3001], row_start=500, rows=2600, absolute_ids=True
            ),
            3001,
        ),
    }


# 48 heads are padded to a head count DeepGEMM supports (64 with DeepGEMM 0.1.3).
@pytest.mark.parametrize("heads", [32, 48])
def test_reference_matches_torch_exact_small(exact_topk, heads):
    ties = []
    for name, (layout, keys) in _layouts().items():
        q, k, weights = _exact_inputs(layout.rows, keys, heads, seed=len(name))
        for topk in (64, 700):
            selected = reference_topk(
                q,
                k,
                weights,
                layout=layout,
                topk=topk,
                softmax_scale=0.5,
                fmt="fp8",
                exact_topk=exact_topk,
            )
            expected = _expected(q, k, weights, layout, topk, 0.5, ties)
            mismatched = (selected != expected).any(dim=1).nonzero().flatten()
            assert mismatched.numel() == 0, (heads, name, topk, mismatched[:8].tolist())
    assert sum(ties) > 100  # the inputs put many cutoffs between equal scores


def test_reference_budget_chunking_invariant(exact_topk):
    heads = 32
    generator = torch.Generator(device="cuda").manual_seed(7)
    cu_seqlens = [0, 3000, 3003, 9000, 12288]
    rows = keys = cu_seqlens[-1]
    q = torch.randn((rows, heads, 128), generator=generator, device="cuda").to(torch.bfloat16)
    k = torch.randn((keys, 128), generator=generator, device="cuda").to(torch.bfloat16)
    weights = torch.randn((rows, heads), generator=generator, device="cuda").to(torch.bfloat16)
    layout = QueryLayout.packed(cu_seqlens, row_start=0, rows=rows, absolute_ids=True)

    def select(layout, q, weights, **kwargs):
        return reference_topk(
            q,
            k,
            weights,
            layout=layout,
            topk=512,
            softmax_scale=128**-0.5,
            fmt="fp8",
            exact_topk=exact_topk,
            **kwargs,
        )

    reference = select(layout, q, weights)
    for kwargs in (dict(budget_bytes=1 << 20), dict(rows_per_call=1000), dict(rows_per_call=4)):
        assert torch.equal(select(layout, q, weights, **kwargs), reference), kwargs
    # One batched call over all sequences equals one call per sequence.
    for segment in layout.segments:
        alone = QuerySegment(
            0,
            segment.rows,
            segment.position,
            segment.key_start,
            segment.key_count,
            segment.index_base,
        )
        single = QueryLayout(rows=segment.rows, segments=(alone,))
        part = slice(segment.row_start, segment.row_end)
        assert torch.equal(select(single, q[part], weights[part]), reference[part])


def test_unaligned_key_view_plain_scores_bitwise():
    # The keys of a sequence that starts at key 3003 have scales at an address 12 bytes past a
    # 16-byte boundary. The selector scores such a view in the plain mode against an aligned
    # copy of its scales; the scores equal the row-relative mode's on the unmoved keys bit for
    # bit. (test_reference_budget_chunking_invariant compares the selections of such a sequence
    # alone with those of a batched call, whose chunks span sequences.)
    from megatron.lite.primitive.kernels.indexer_topk import reference as reference_module

    fmt, heads = "fp8", 32
    generator = torch.Generator(device="cuda").manual_seed(13)
    rows, first, width = 1024, 3003, 4096
    q = torch.randn((rows, heads, 128), generator=generator, device="cuda").to(torch.bfloat16)
    k = torch.randn((first + width, 128), generator=generator, device="cuda").to(torch.bfloat16)
    weights = torch.randn((rows, heads), generator=generator, device="cuda").to(torch.bfloat16)
    keys = reference_module.quantize_keys(k, fmt)
    data, scales, folded = reference_module.quantize_queries(
        q, weights, fmt, softmax_scale=0.5, kernel_heads=heads
    )
    assert keys.scale[first:].data_ptr() % 16 != 0
    view = reference_module._plain_key_view((keys.data, keys.scale), first, width)
    assert view is not None and view[0].data_ptr() == keys.data[first:].data_ptr()
    assert view[1].data_ptr() % 16 == 0 and torch.equal(view[1], keys.scale[first:])
    ends = torch.randint(1, width + 1, (rows,), generator=generator, device="cuda")
    ends = ends.to(torch.int32)
    zeros = torch.zeros_like(ends)
    plain = reference_module._mqa_logits((data, scales), view, folded, zeros, ends, max_seqlen_k=0)
    plain = plain[:, :width]
    relative = reference_module._mqa_logits(
        (data, scales),
        (keys.data, keys.scale),
        folded,
        zeros + first,
        ends + first,
        max_seqlen_k=width,
    )[:, :width]
    valid = torch.arange(width, device="cuda")[None, :] < ends[:, None]
    assert torch.equal(plain[valid].view(torch.int32), relative[valid].view(torch.int32))


def test_reference_radix_topk_selects_the_best_scores():
    pytest.importorskip("cudnn")
    generator = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn((2048, 32, 128), generator=generator, device="cuda").to(torch.bfloat16)
    k = torch.randn((2048, 128), generator=generator, device="cuda").to(torch.bfloat16)
    weights = torch.rand((2048, 32), generator=generator, device="cuda")
    layout = QueryLayout.full(2048, keys=2048)
    selected = reference_topk(q, k, weights, layout=layout, topk=256, softmax_scale=1.0, fmt="fp8")
    expected = _expected(q, k, weights, layout, 256, 1.0)
    # Without the exact-tie package equal scores may be ordered either way: compare the scores
    # (float64 here, so keys whose float32 scores are nearly equal may also swap).
    queries, keys = _dequantize(q), _dequantize(k)
    scores = torch.cat(
        [
            (
                torch.einsum("rhd,kd->rhk", queries[first : first + 256], keys).relu()
                * weights[first : first + 256].double()[..., None]
            ).sum(1)
            for first in range(0, 2048, 256)
        ]
    )
    for ids in (selected, expected):
        assert ((ids >= 0).sum(1) == torch.arange(1, 2049, device="cuda").clamp(max=256)).all()
    gathered = [
        torch.sort(
            torch.gather(scores, 1, ids.clamp(min=0).long()).masked_fill(ids < 0, 0), 1
        ).values
        for ids in (selected, expected)
    ]
    assert torch.allclose(gathered[0], gathered[1], rtol=1e-6, atol=0)
