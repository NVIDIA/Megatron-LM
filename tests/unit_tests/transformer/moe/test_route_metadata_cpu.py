# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import moe_layer


def test_hash_ids_and_padding_use_owning_tp_group(monkeypatch):
    group = SimpleNamespace(size=lambda: 2)
    calls = []

    def scatter(value, group=None):
        calls.append(group)
        return value[2:]

    monkeypatch.setattr(moe_layer.tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
    received = []

    class Router(torch.nn.Module):
        is_hash_layer = True

        def forward(self, hidden, padding, input_ids=None, packed_seq_params=None):
            received.append((padding, input_ids))
            return hidden, input_ids

    layer = object.__new__(moe_layer.MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        sequence_parallel=True, tensor_model_parallel_size=2, cuda_graph_impl="none"
    )
    layer.tp_group = group
    layer.router = Router()
    mask = torch.tensor([[False, True, False, True]])
    ids = torch.tensor([[11, 12, 13, 14]])
    hidden = torch.zeros(2, 1, 4)
    moe_layer.MoELayer.route(layer, hidden, padding_mask=mask, input_ids=ids)
    assert calls == [group, group]
    torch.testing.assert_close(received[0][0], mask[:, 2:].T)
    torch.testing.assert_close(received[0][1], ids[:, 2:])


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_flex_preprocess_aligns_padding_to_owning_sp_rank(monkeypatch, tp_rank):
    group = SimpleNamespace(size=lambda: 2)
    calls = []

    def scatter(value, group=None):
        calls.append(group)
        return value.chunk(2, dim=0)[tp_rank]

    monkeypatch.setattr(moe_layer.tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
    received = []

    class Dispatcher(moe_layer.MoEFlexTokenDispatcher):
        def __init__(self):
            pass

        def dispatch_preprocess(self, hidden, routing, probs, padding_mask=None):
            received.append(padding_mask)
            assert padding_mask.shape == (hidden.shape[1], hidden.shape[0])
            return hidden, probs

    layer = object.__new__(moe_layer.MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        sequence_parallel=True, cuda_graph_impl="none", moe_latent_size=None
    )
    layer.tp_group = group
    layer.token_dispatcher = Dispatcher()
    mask = torch.tensor([[False, True, True, False], [True, False, False, True]])
    hidden = torch.zeros(2, 2, 4)
    probs = torch.ones(4, 2)
    routing = torch.ones(4, 2, dtype=torch.bool)
    layer.preprocess(hidden, probs, routing, mask)
    expected = mask[:, tp_rank * 2 : (tp_rank + 1) * 2]
    torch.testing.assert_close(received[0], expected)
    # Already local metadata must not be scattered a second time.
    layer.preprocess(hidden, probs, routing, expected)
    torch.testing.assert_close(received[1], expected)
    assert calls == [group]
