# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.wgrad_overlap import release_deferred_wgrads_after


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.events = []

    def forward(self, hidden_states, scale=1.0):
        out = self.linear(hidden_states) * scale
        if out.requires_grad:
            out.register_hook(lambda _: self.events.append("module backward"))
        return out


@pytest.mark.parametrize("by_keyword", [False, True])
def test_callback_runs_right_after_the_module_backward(by_keyword):
    """TransformerBlock passes hidden_states by keyword, TransformerLayer positionally."""
    module = _Linear()
    release_deferred_wgrads_after(module, lambda: module.events.append("released"))
    x = torch.randn(2, 4, requires_grad=True)

    out = module(hidden_states=x, scale=2.0) if by_keyword else module(x, 2.0)
    out.sum().backward()

    assert module.events == ["module backward", "released"]
    assert module.linear.weight.grad is not None
    assert torch.equal(x.grad, 2.0 * module.linear.weight.sum(0).expand(2, 4).detach())


def test_no_release_without_a_graph():
    module = _Linear()
    release_deferred_wgrads_after(module, lambda: module.events.append("released"))

    with torch.no_grad():
        module(torch.randn(2, 4))
    module(torch.randn(2, 4, requires_grad=False)).sum().backward()

    assert module.events == ["module backward"]
