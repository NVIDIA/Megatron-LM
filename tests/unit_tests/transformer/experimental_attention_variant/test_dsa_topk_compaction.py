# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The sort-free top-k compaction order equals the stable argsort it replaces."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels import (
    _compact_valid_topk_indices,
    _stable_compaction_order,
)

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _argsort_order(valid):
    positions = torch.arange(valid.size(-1), device=valid.device).expand_as(valid)
    key = torch.where(valid, positions, torch.full_like(positions, valid.size(-1)))
    return key.argsort(dim=-1, stable=True)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape", [(1, 7, 1), (2, 5, 64), (3, 33, 2049)])
@pytest.mark.parametrize("density", [0.0, 0.3, 0.97, 1.0])
def test_compaction_order_equals_stable_argsort(device, shape, density):
    generator = torch.Generator().manual_seed(hash((shape, density)) % (2**31))
    valid = (torch.rand(shape, generator=generator) < density).to(device)

    order = _stable_compaction_order(valid)

    assert order.dtype == torch.int64
    assert torch.equal(order, _argsort_order(valid))


@pytest.mark.parametrize("device", _DEVICES)
def test_compacted_indices_keep_valid_entries_in_order(device):
    generator = torch.Generator().manual_seed(1234)
    indices = torch.randint(0, 1000, (2, 9, 300), generator=generator, dtype=torch.int32)
    indices[torch.rand(indices.shape, generator=generator) < 0.4] = -1
    indices = indices.to(device)

    compacted, lengths = _compact_valid_topk_indices(indices)

    for row, out, length in zip(
        indices.reshape(-1, 300), compacted.reshape(-1, 300), lengths.flatten()
    ):
        kept = row[row >= 0]
        assert int(length) == kept.numel()
        assert torch.equal(out[: kept.numel()], kept)
        assert bool((out[kept.numel() :] == -1).all())
