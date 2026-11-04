# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The shared mixin interface; layer behavior is covered by existing TE graph tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core.transformer.cuda_graphs as cuda_graphs
from megatron.core.extensions.te_cuda_graphs import TECudaGraphMixin
from megatron.core.transformer.module import MegatronModule


def _config(backend="transformer_engine"):
    return SimpleNamespace(
        cuda_graph_impl=backend,
        fine_grained_activation_offloading=False,
        delay_offload_until_cuda_graph=False,
        sequence_packing_scheduler=None,
    )


class _Module(TECudaGraphMixin, MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, hidden_states):
        return hidden_states * self.weight

    def is_graphable_for_scope(self, modules):
        return not modules or "attn" in modules


@pytest.mark.parametrize("backend", ["none", "local", "transformer_engine"])
def test_te_cuda_graph_mixin_dispatch(monkeypatch, backend):
    """A plain MegatronModule opts in only for TE, then dispatches by capture/training state."""
    module = _Module(_config(backend))
    events = []
    module.register_forward_pre_hook(lambda *_: events.append("eager"))
    x = torch.tensor(3.0)
    if backend != "transformer_engine":
        monkeypatch.setattr(
            cuda_graphs, "is_graph_capturing", Mock(side_effect=AssertionError("TE path entered"))
        )
        assert not hasattr(module, "te_cuda_graphs")
        with pytest.raises(ValueError, match="registration requires"):
            module.register_te_cuda_graph()
        torch.testing.assert_close(module(x), x * 2)
        assert events == ["eager"]
        return

    module.register_te_cuda_graph()
    assert TECudaGraphMixin.is_graphable_for_scope(module, ())
    assert not TECudaGraphMixin.is_graphable_for_scope(module, ("attn",))
    assert module.is_graphable_for_scope(()) and module.is_graphable_for_scope(("attn",))
    assert not module.is_graphable_for_scope(("mlp",))
    module.setup_manual_hooks(lambda: lambda _: events.append("manual"))
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)
    torch.testing.assert_close(module(x), x * 2)
    assert events == ["eager"]
    events.clear()
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", True)
    torch.testing.assert_close(module(x), x * 2)
    assert events == []  # Capture bypasses ordinary forward hooks.
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)

    def graph(index):
        def replay(hidden_states, is_first_microbatch, optional="omitted"):
            assert optional is None
            events.append((index, is_first_microbatch))
            return hidden_states + index

        return replay

    module.te_cuda_graphs = [graph(0), graph(1)]
    for microbatch in (0, 1, 2):
        events.clear()
        module.te_cuda_graph_current_microbatch = microbatch
        torch.testing.assert_close(module(hidden_states=x, optional=None), x + microbatch % 2)
        assert events == ["manual", (microbatch % 2, microbatch == 0)]
    other = _Module(_config())
    assert other.te_cuda_graphs == [] and other.te_cuda_graph_manual_hooks == []

    events.clear()
    module.eval()
    torch.testing.assert_close(module(x), x * 2)
    assert events == ["eager"]


@pytest.mark.parametrize("failure", [None, "graph", "flush", "after"])
def test_te_cuda_graph_mixin_hooks(monkeypatch, failure):
    """Adapters can customize capture and replay while sharing offload ordering and cleanup."""
    events = []

    def record(name):
        events.append(name)
        if name == failure:
            raise RuntimeError(name)

    def backward_record(hidden_states):
        record("backward")
        return hidden_states

    offload = SimpleNamespace(
        backward_record=backward_record,
        forward_record=lambda: record("forward"),
        enter_replay=lambda: record("enter"),
        flush_delayed_groups=lambda: record("flush"),
        exit_replay=lambda: record("exit"),
    )

    class Adapter(_Module):
        def capture_forward_hook(self, *args, **kwargs):
            record("capture")
            return self.forward(*args, **kwargs) + 1

        def before_replay_hook(self, args, kwargs):
            record("before")
            return (args[0] + 1,), kwargs, args[0]

        def after_replay_hook(self, outputs, state):
            record("after")
            return outputs + state

        def get_offload_interface(self):
            return offload

    module = Adapter(_config())
    module.offload_module_in_cuda_graph = True
    module.config.delay_offload_until_cuda_graph = True
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", True)
    for args, kwargs in [((torch.tensor(3.0),), {}), ((), {"hidden_states": torch.tensor(3.0)})]:
        events.clear()
        torch.testing.assert_close(module(*args, **kwargs), torch.tensor(7.0))
        assert events == ["backward", "capture", "forward"]

    def graph(hidden_states, **kwargs):
        record("graph")
        return hidden_states * module.weight + 1

    module.te_cuda_graphs = [graph]
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)
    events.clear()
    order = ["before", "enter", "graph", "flush", "after", "exit"]
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            module(torch.tensor(3.0))
        assert events == order[: order.index(failure) + 1] + ["exit"]
    else:
        for value in (3.0, 7.0):
            events.clear()
            x = torch.tensor(value, requires_grad=True)
            output = module(x)
            torch.testing.assert_close(output, x.detach() * 3 + 3)
            output.backward()
            torch.testing.assert_close(x.grad, torch.tensor(3.0))
            assert events == order
