# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.moe_layer import MoELayer


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_router_padding_mask_uses_explicit_tp_group(tp_rank):
    """Non-first chunks scatter raw masks using their own attention TP group."""
    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=True),
        attn_tp_group=SimpleNamespace(size=lambda: 2, rank=lambda: tp_rank),
    )
    mask = torch.tensor([[False, True, True, False], [True, False, True, False]])
    original = mask.clone()
    hidden_states = torch.empty(2, 2, 8)
    result = MoELayer._normalize_padding_mask(layer, hidden_states, mask)
    expected = mask[:, 2 * tp_rank : 2 * (tp_rank + 1)].transpose(0, 1)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(mask, original)


def test_router_padding_mask_already_scattered_is_only_transposed():
    """The embedding chunk's local [B,S] mask must not be scattered again."""
    layer = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True))
    mask = torch.tensor([[False, True, False], [True, True, False]])
    result = MoELayer._normalize_padding_mask(layer, torch.empty(3, 2, 8), mask)
    torch.testing.assert_close(result, mask.transpose(0, 1))
    assert MoELayer._normalize_padding_mask(layer, torch.empty(3, 2, 8), None) is None


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_router_padding_mask_rejects_incompatible_sequence_length(sequence_parallel):
    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=sequence_parallel),
        attn_tp_group=SimpleNamespace(size=lambda: 2),
    )
    with pytest.raises(AssertionError, match="cannot be aligned"):
        MoELayer._normalize_padding_mask(layer, torch.empty(3, 2, 8), torch.zeros(2, 5))
