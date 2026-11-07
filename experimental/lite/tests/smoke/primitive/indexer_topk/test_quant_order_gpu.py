# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU equivalence of the indexer top-k row-order Triton kernels and their torch semantics.

Run explicitly (optional suite)::

    experimental/lite/tests/run_tests.sh \
        experimental/lite/tests/smoke/primitive/indexer_topk/test_quant_order_gpu.py
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import compact_valid_topk_, order, sort_topk_rows_

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]

pytest.importorskip("triton")

_INT32_MAX = torch.iinfo(torch.int32).max


def _index_chunk(start: int, stop: int, keys: int, topk: int) -> torch.Tensor:
    """Deterministic selector-like rows: random ids with -1 holes at a random per-row density."""
    generator = torch.Generator(device="cuda").manual_seed(4242 + start)
    rows = stop - start
    ids = torch.randint(
        0, keys, (rows, topk), device="cuda", dtype=torch.int32, generator=generator
    )
    density = torch.rand((rows, 1), device="cuda", generator=generator)
    density[::7] = 0.0
    density[3::11] = 1.0
    holes = torch.rand((rows, topk), device="cuda", generator=generator) < density
    return ids.masked_fill_(holes, -1)


@pytest.mark.parametrize("rows", [1776, 1 << 20])
@torch.no_grad()
def test_sort_matches_torch_sort_1m(rows):
    topk, keys, chunk = 2048, 1 << 20, 65536
    indices = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        indices[start:stop] = _index_chunk(start, stop, keys, topk)

    assert sort_topk_rows_(indices) is indices

    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        original = _index_chunk(start, stop, keys, topk)
        expected = torch.sort(torch.where(original < 0, _INT32_MAX, original), dim=-1).values
        expected = torch.where(expected == _INT32_MAX, -1, expected)
        assert torch.equal(indices[start:stop], expected)


@pytest.mark.parametrize("width,out_width", [(128, 128), (73, 2048), (512, 512), (0, 16)])
@torch.no_grad()
def test_compact_valid_topk_triton_matches_torch(monkeypatch, width, out_width):
    generator = torch.Generator().manual_seed(width * 7919 + out_width)
    rows = 1029
    # Row-strided selector outputs on both devices.
    storage = torch.randint(-12, 4100, (rows, width + 9), generator=generator, dtype=torch.int32)
    ends = torch.randint(0, 4100, (rows,), generator=generator, dtype=torch.int32)
    ends[:4] = torch.tensor([0, 1, 4099, 4100], dtype=torch.int32)
    expected = torch.full((rows, out_width), -99, dtype=torch.int32)
    expected_lengths = torch.empty(rows, dtype=torch.int32)
    compact_valid_topk_(storage[:, :width], ends, expected, expected_lengths)

    def _no_fallback(*args, **kwargs):
        raise AssertionError("CUDA inputs within the kernel limits must use the Triton kernel")

    monkeypatch.setattr(order, "_compact_valid_topk_torch", _no_fallback)
    selected = storage.cuda()[:, :width]
    out_storage = torch.full((rows, out_width + 13), -99, device="cuda", dtype=torch.int32)
    out = out_storage[:, :out_width]
    lengths = torch.full((rows,), -99, device="cuda", dtype=torch.int32)
    compact_valid_topk_(selected, ends.cuda(), out, lengths)
    assert torch.equal(out.cpu(), expected) and torch.equal(lengths.cpu(), expected_lengths)
    assert bool((out_storage[:, out_width:] == -99).all())

    without_lengths = torch.full((rows, out_width), -99, device="cuda", dtype=torch.int32)
    compact_valid_topk_(selected, ends.cuda(), without_lengths)
    assert torch.equal(without_lengths.cpu(), expected)
