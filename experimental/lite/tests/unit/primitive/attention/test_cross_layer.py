# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.lite.primitive.modules.attention.cross_layer import IndexProjection
from megatron.lite.primitive.parallel.source_state import KVSourceState


class Projection(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_cross_segment_sources_native_core_checkpoint(segments, cpu_core_checkpoint):
    """CPU adapts only RNG snapshots; native Core checkpoint/autograd execute."""

    torch.manual_seed(7)
    state = KVSourceState()
    owner = IndexProjection(Projection(4, 4, bias=False), state, 0, 0, "index_q")
    readers = [IndexProjection(None, state, i + 1, 0, "index_q") for i in range(segments)]
    x = torch.randn(3, 1, 4, requires_grad=True)
    initial_weight = owner.linear.weight.detach().clone()
    calls = []

    def owner_segment(value):
        calls.append(("owner", torch.is_grad_enabled()))
        return owner(value)[0].sin()

    def reader_segment(reader, value):
        calls.append(("reader", torch.is_grad_enabled()))
        return value.cos() + reader(value)[0].square()

    with state.microbatch():
        actual = state.checkpoint(owner_segment, x, checkpoint_fn=cpu_core_checkpoint)
        for reader in readers:
            actual = state.checkpoint(
                lambda value, reader=reader: reader_segment(reader, value),
                actual,
                checkpoint_fn=cpu_core_checkpoint,
            )
    assert not state.active and not state.values
    actual.sum().backward()
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = initial_weight.requires_grad_()
    shared = torch.nn.functional.linear(reference_x, reference_weight)
    reference = shared.sin()
    for _ in readers:
        reference = reference.cos() + shared.square()
    reference.sum().backward()
    assert torch.equal(actual, reference)
    assert torch.equal(x.grad, reference_x.grad)
    assert torch.equal(owner.linear.weight.grad, reference_weight.grad)
    assert torch.isfinite(x.grad).all() and torch.isfinite(owner.linear.weight.grad).all()
    reverse = [name for name, grad in calls if grad]
    assert reverse == ["reader"] * segments + ["owner"]
    assert not state.active and not state.values
