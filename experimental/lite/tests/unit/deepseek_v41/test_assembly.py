# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model assembly boundaries: pipeline source ownership and release loading."""

import pytest
import torch

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.components.carrier import pack_sources, unpack_sources
from megatron.lite.model.deepseek_v41.lite.protocol import ImplConfig, _validate
from megatron.lite.primitive.modules.engram import NgramHash
from megatron.lite.primitive.parallel.source_state import KVSourceState


def test_pipeline_source_carrier_preserves_values_and_owner_gradients():
    hidden = torch.randn(5, 1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    grad_alignment = []
    hidden.register_hook(lambda grad: grad_alignment.append(grad.data_ptr() % 16))
    pre = torch.sigmoid(torch.randn(5, 1, 4)).requires_grad_()
    owner = torch.randn(5, 8)
    owner[0, :2] = torch.tensor([-0.0, 0.0])
    owner.requires_grad_()
    prefix, new_prefix = torch.tensor([0, 2, 5], dtype=torch.int32), torch.tensor(
        [0, 2, 5], dtype=torch.int32
    )
    group, new_group = object(), object()
    state = KVSourceState()
    key = (3, ("source", prefix.data_ptr(), id(group), torch.bfloat16, torch.device("cpu")))
    pool = torch.tensor([[True, False], [False, True]])
    state.values = {key: (owner, prefix), ("candidate_pool", 3): pool}
    carried = pack_sources(hidden, pre, state, prefix, group)
    h, p, values = unpack_sources(carried, new_prefix, new_group)
    new_key = (
        3,
        ("source", new_prefix.data_ptr(), id(new_group), torch.bfloat16, torch.device("cpu")),
    )
    recovered, recovered_prefix = values[new_key]
    assert h.data_ptr() % 16 == 0 and torch.equal(h, hidden) and torch.equal(p, pre)
    assert torch.equal(recovered.view(torch.int32), owner.view(torch.int32)), "FP32 raw bits"
    assert torch.equal(recovered_prefix, prefix)
    assert torch.equal(values[("candidate_pool", 3)], pool)
    (h.float().sum() + p.sum() + recovered.sum() * 2 + recovered.sum() * 3).backward()
    assert torch.equal(owner.grad, torch.full_like(owner, 5))
    assert torch.equal(hidden.grad, torch.ones_like(hidden)) and grad_alignment == [0]
    assert torch.equal(pre.grad, torch.ones_like(pre))


def test_hash_resets_history_at_each_packed_document():
    hash_fn = NgramHash(
        torch.arange(16), 0, torch.tensor([[3, 5, 7]]), torch.tensor([[[17], [19]]])
    )
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    prefix = torch.tensor([0, 2, 5], dtype=torch.int32)
    packed = hash_fn(ids, cu_seqlens=prefix)
    expected = torch.cat([hash_fn(ids[:, :2]), hash_fn(ids[:, 2:])], dim=1)
    assert torch.equal(packed, expected)


@pytest.mark.parametrize(
    "option,reason",
    [
        ({"quantized": True}, "BF16"),
        ({"w4a8_experts": True}, "BF16"),
        ({"dtype": torch.float32}, "BF16"),
        ({"optimizer": "muon"}, "optimizer"),
        ({"optimizer_config": {"optimizer": "muon"}}, "optimizer"),
        ({"trainable_engram": True}, "Trainable Engram"),
    ],
)
def test_unsupported_training_modes_fail_before_allocation(option, reason):
    with pytest.raises(ValueError, match=reason):
        _validate(DeepseekV41Config(num_hidden_layers=4), ImplConfig(**option))
