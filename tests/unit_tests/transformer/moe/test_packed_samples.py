# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sample boundaries must survive per-document CP padding before MoE routing."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer


def test_padded_samples_keep_the_router_hidden_dimension():
    layer = SimpleNamespace(is_moe_layer=True, config=SimpleNamespace(sequence_parallel=False))
    packed = PackedSeqParams(tokens_per_sample=8, moe_sample_lengths=(6, 6, 6))
    hidden = torch.arange(18 * 64, dtype=torch.float32).view(18, 1, 64).requires_grad_()

    routed, mask, _, lengths = TransformerLayer._maybe_unflatten_for_moe(
        layer, hidden, None, None, packed
    )

    assert routed.shape == (6, 3, 64)
    torch.testing.assert_close(routed[:, 1], hidden[6:12, 0])
    assert not mask.any()
    assert torch.nn.Linear(64, 2)(routed).shape == (6, 3, 2)
    restored = TransformerLayer._maybe_reflatten_from_moe(layer, routed, packed, lengths)
    torch.testing.assert_close(restored, hidden)
    restored.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))


def test_unequal_samples_keep_padding_and_input_ids_aligned():
    layer = SimpleNamespace(is_moe_layer=True, config=SimpleNamespace(sequence_parallel=False))
    packed = PackedSeqParams(tokens_per_sample=8, moe_sample_lengths=(6, 4))
    hidden = torch.arange(20, dtype=torch.float32).view(10, 1, 2)
    padding_mask = torch.tensor(
        [[False, True, False, False, True, True, False, False, False, False]]
    )
    input_ids = torch.arange(10).view(1, 10)

    routed, mask, ids, lengths = TransformerLayer._maybe_unflatten_for_moe(
        layer, hidden, padding_mask, input_ids, packed
    )

    assert routed.shape == (6, 2, 2)
    torch.testing.assert_close(routed[:4, 1], hidden[6:, 0])
    torch.testing.assert_close(routed[4:, 1], torch.zeros(2, 2))
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [[False, True, False, False, True, True], [False, False, False, False, True, True]]
        ),
    )
    torch.testing.assert_close(ids, torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 0, 0]]))
    restored = TransformerLayer._maybe_reflatten_from_moe(layer, routed, packed, lengths)
    torch.testing.assert_close(restored, hidden)


def test_sequence_parallel_samples_keep_a_common_router_shape():
    layer = SimpleNamespace(
        is_moe_layer=True, config=SimpleNamespace(sequence_parallel=True), tp_group=Mock()
    )
    layer.tp_group.rank.return_value = 1
    packed = PackedSeqParams(tokens_per_sample=8, moe_sample_lengths=(6, 4))
    # TP rank 1 owns the final token of sample 0 and all four tokens of sample 1.
    hidden = torch.arange(10, 20, dtype=torch.float32).view(5, 1, 2)

    routed, mask, _, lengths = TransformerLayer._maybe_unflatten_for_moe(
        layer, hidden, None, None, packed
    )

    assert routed.shape == (6, 2, 2)
    assert lengths == (1, 4)
    torch.testing.assert_close(routed[0, 0], hidden[0, 0])
    torch.testing.assert_close(routed[:4, 1], hidden[1:, 0])
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [[False, True, True, True, True, True], [False, False, False, False, True, True]]
        ),
    )
    restored = TransformerLayer._maybe_reflatten_from_moe(layer, routed, packed, lengths)
    torch.testing.assert_close(restored, hidden)
