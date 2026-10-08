# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

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

        def forward(self, hidden, padding, ids, packed):
            received.append((padding, ids))
            return hidden, ids

    layer = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=True, tensor_model_parallel_size=2, cuda_graph_impl="none"
        ),
        tp_group=group,
        router=Router(),
    )
    mask = torch.tensor([[False, True, False, True]])
    ids = torch.tensor([[11, 12, 13, 14]])
    hidden = torch.zeros(2, 1, 4)
    moe_layer.MoELayer.route(layer, hidden, padding_mask=mask, input_ids=ids)
    assert calls == [group, group]
    torch.testing.assert_close(received[0][0], mask[:, 2:].T)
    torch.testing.assert_close(received[0][1], ids[:, 2:])
