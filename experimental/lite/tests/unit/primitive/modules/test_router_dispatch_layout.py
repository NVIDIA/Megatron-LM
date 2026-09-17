# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.lite.primitive.modules.router import _ordered_topk_from_routing_map


@pytest.mark.parametrize("rows", [1, 3, 17])
def test_ordered_topk_dispatch_layout_preserves_values_and_gradients(rows):
    scores = torch.randn(rows, 128, requires_grad=True)
    selected = torch.topk(scores.detach(), 8, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True)
    expected_ids = selected.sort(dim=-1).values
    expected = scores.gather(1, expected_ids)
    actual, indices = _ordered_topk_from_routing_map(scores, mask, 8)
    assert indices.is_contiguous()
    assert torch.equal(indices, expected_ids)
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), scores)[0]
    assert torch.equal(actual_grad, expected_grad)
