# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the indexer top-k row-order kernels."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import compact_valid_topk_, sort_topk_rows_

pytestmark = pytest.mark.mlite

_INT32_MAX = torch.iinfo(torch.int32).max


def _sorted_reference(indices: torch.Tensor) -> torch.Tensor:
    values = torch.where(indices < 0, _INT32_MAX, indices).sort(dim=-1).values
    return torch.where(values == _INT32_MAX, -1, values)


def test_sort_topk_rows_invalid_last():
    indices = torch.tensor(
        [[5, -1, 3, -1, 0, 7], [-1] * 6, [9, 8, 7, 6, 5, 4], [2, 2, -1, 1, -5, 0]],
        dtype=torch.int32,
    )
    result = sort_topk_rows_(indices)
    assert result is indices
    assert indices.tolist() == [
        [0, 3, 5, 7, -1, -1],
        [-1] * 6,
        [4, 5, 6, 7, 8, 9],
        [0, 1, 2, 2, -1, -1],
    ]

    # In place on a strided view; the surrounding storage is untouched.
    storage = torch.full((3, 9), 42, dtype=torch.int32)
    storage[:, 1:7] = torch.tensor([[3, -1, 1, 2, -1, 0]] * 3, dtype=torch.int32)
    sort_topk_rows_(storage[:, 1:7], rows_per_chunk=2)
    assert storage.tolist() == [[42, 0, 1, 2, 3, -1, -1, 42, 42]] * 3

    with pytest.raises(ValueError, match="int32"):
        sort_topk_rows_(torch.zeros(2, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="int32"):
        sort_topk_rows_(torch.zeros(4, dtype=torch.int32))
    with pytest.raises(ValueError, match="rows_per_chunk"):
        sort_topk_rows_(torch.zeros(2, 4, dtype=torch.int32), rows_per_chunk=0)


def test_sort_chunking_invariant():
    generator = torch.Generator().manual_seed(1234)
    base = torch.randint(0, 1 << 20, (37, 64), generator=generator, dtype=torch.int32)
    base[torch.rand(37, 64, generator=generator) < 0.3] = -1
    base[5] = -1
    base[6, :3] = 7
    expected = _sorted_reference(base)

    for rows_per_chunk in (1, 2, 5, 36, 37, 100, 32768):
        assert torch.equal(sort_topk_rows_(base.clone(), rows_per_chunk=rows_per_chunk), expected)
    assert sort_topk_rows_(torch.empty(0, 64, dtype=torch.int32)).shape == (0, 64)


def test_compact_valid_topk_semantics():
    generator = torch.Generator().manual_seed(532)
    # A row-strided selector output and a row-strided view of a wider output buffer.
    selected = torch.randint(-12, 513, (7, 82), generator=generator, dtype=torch.int32)[:, :73]
    ends = torch.tensor([0, 1, 7, 63, 127, 255, 512], dtype=torch.int32)
    storage = torch.full((7, 2048 + 13), -99, dtype=torch.int32)
    out = storage[:, :2048]
    lengths = torch.full((7,), -99, dtype=torch.int32)

    compact_valid_topk_(selected, ends, out, lengths)

    for row in range(7):
        valid = [value for value in selected[row].tolist() if 0 <= value < ends[row].item()]
        assert out[row].tolist() == valid + [-1] * (2048 - len(valid))
        assert lengths[row].item() == len(valid)
    assert (storage[:, 2048:] == -99).all()

    same_width = torch.empty(7, 73, dtype=torch.int32)
    compact_valid_topk_(selected, ends, same_width)
    assert torch.equal(same_width, out[:, :73])

    compact_valid_topk_(torch.empty(0, 4, dtype=torch.int32), ends[:0], out[:0])
    with pytest.raises(ValueError, match="at least 73 columns"):
        compact_valid_topk_(selected, ends, torch.empty(7, 72, dtype=torch.int32))
    with pytest.raises(ValueError, match="int32"):
        compact_valid_topk_(selected, ends.long(), out)
    with pytest.raises(ValueError, match="shape"):
        compact_valid_topk_(selected, ends[:6], out)
