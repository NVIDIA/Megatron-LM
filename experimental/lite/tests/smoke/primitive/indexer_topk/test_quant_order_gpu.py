# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU equivalence of the indexer top-k Triton kernels and the torch semantics they implement.

Run explicitly (optional suite)::

    experimental/lite/tests/run_tests.sh \
        experimental/lite/tests/smoke/primitive/indexer_topk/test_quant_order_gpu.py
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import (
    compact_valid_topk_,
    order,
    quantize_indexer_mxfp4_rows,
    quantize_indexer_mxfp4_rows_reference,
    sort_topk_rows_,
)

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]

pytest.importorskip("triton")

_INT32_MAX = torch.iinfo(torch.int32).max
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _finite_bf16_values() -> torch.Tensor:
    values = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return values[torch.isfinite(values)]


def _midpoint_sweep() -> torch.Tensor:
    """Every E2M1 midpoint, and its bfloat16 neighbours, at every reachable finite group scale.

    A group of amax ``6 * 2**(e - 127)`` has exactly the scale ``2**(e - 127)``; e = 114 is the
    1e-4 floor and e = 252 the largest scale whose amax is finite in bfloat16.
    """
    middle = torch.tensor(_MIDPOINTS, dtype=torch.bfloat16)
    below = (middle.view(torch.int16) - 1).view(torch.bfloat16)
    above = (middle.view(torch.int16) + 1).view(torch.bfloat16)
    group = torch.cat(
        [
            torch.tensor([6.0, -6.0]),
            below.float(),
            middle.float(),
            above.float(),
            -middle.float(),
            torch.tensor([-0.0, 0.0]),
        ]
    )
    groups = [group * 2.0 ** (exponent - 127) for exponent in range(114, 253)]
    groups += [torch.zeros(32)] * (-len(groups) % 4)
    return torch.stack(groups).to(torch.bfloat16).reshape(-1, 128)


def _mxfp4_case(name: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260930)
    if name == "random_rows":
        return torch.randn(4103, 128, generator=generator).to(torch.bfloat16) * 3
    if name == "row_tails":
        return torch.randn(33, 128, generator=generator).to(torch.bfloat16)
    if name == "leading_dims":
        return torch.randn(5, 64, 128, generator=generator).to(torch.bfloat16)
    if name == "all_finite_bf16_shuffled":
        values = _finite_bf16_values()
        return values[torch.randperm(values.numel(), generator=generator)].reshape(-1, 128)
    if name == "all_finite_bf16_by_magnitude":
        values = _finite_bf16_values()
        return values[values.float().abs().argsort(stable=True)].reshape(-1, 128)
    if name == "midpoint_sweep":
        return _midpoint_sweep()
    if name == "specials":
        special = torch.zeros(4, 128)
        special[0, :8] = torch.tensor([*_MIDPOINTS, 6.0])
        special[1, :32] = 1.0e-5
        special[1, 32:64] = -0.0
        special[2, 0], special[2, 1], special[2, 2] = float("inf"), float("-inf"), 1.0
        special[3] = torch.finfo(torch.bfloat16).max
        return special.to(torch.bfloat16)
    if name == "float16":
        return torch.randn(257, 128, generator=generator).to(torch.float16) * 100
    if name == "float32":
        magnitude = torch.exp(torch.randn(257, 128, generator=generator) * 8)
        return magnitude * torch.randn(257, 128, generator=generator).sign()
    if name == "non_contiguous":
        return torch.randn(128, 257, generator=generator).to(torch.bfloat16).t()
    raise AssertionError(name)


@pytest.mark.parametrize(
    "case",
    [
        "random_rows",
        "row_tails",
        "leading_dims",
        "all_finite_bf16_shuffled",
        "all_finite_bf16_by_magnitude",
        "midpoint_sweep",
        "specials",
        "float16",
        "float32",
        "non_contiguous",
        "large_row_offsets",
    ],
)
@torch.no_grad()
def test_mxfp4_triton_matches_reference_bitwise(case):
    if case == "large_row_offsets":
        # With 2**24 + 17 rows the last rows start beyond 2**31 elements of the source.
        torch.manual_seed(871)
        rows = 2**24 + 17
        value = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        packed, scales = quantize_indexer_mxfp4_rows(value)
        for part in (slice(0, 17), slice(rows - 17, rows)):
            expected_packed, expected_scales = quantize_indexer_mxfp4_rows_reference(value[part])
            assert torch.equal(packed[part], expected_packed)
            assert torch.equal(scales[part], expected_scales)
        return

    source = _mxfp4_case(case)
    packed, scales = quantize_indexer_mxfp4_rows(source.cuda())
    reference_packed, reference_scales = quantize_indexer_mxfp4_rows_reference(source.cuda())
    cpu_packed, cpu_scales = quantize_indexer_mxfp4_rows_reference(source)

    assert packed.shape == (*source.shape[:-1], 64) and scales.shape == source.shape[:-1]
    assert torch.equal(packed, reference_packed) and torch.equal(scales, reference_scales)
    assert torch.equal(packed.cpu(), cpu_packed) and torch.equal(scales.cpu(), cpu_scales)


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
