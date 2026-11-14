# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The shared mixin interface; layer behavior is covered by existing TE graph tests."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core.transformer.cuda_graphs as cuda_graphs
from megatron.core.extensions.te_cuda_graphs import (
    TECudaGraphMixin,
    set_te_cuda_graph_current_microbatch,
)
from megatron.core.transformer.module import MegatronModule


def _config(backend="transformer_engine"):
    return SimpleNamespace(
        cuda_graph_impl=backend,
        fine_grained_activation_offloading=False,
        delay_offload_until_cuda_graph=False,
        sequence_packing_scheduler=None,
    )


class _Module(TECudaGraphMixin, MegatronModule):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, hidden_states, offset=0):
        return hidden_states * self.weight + offset

    def is_graphable_for_scope(self, modules):
        return not modules or "attn" in modules


@pytest.mark.parametrize("backend", ["none", "local", "transformer_engine"])
def test_te_cuda_graph_mixin_dispatch(monkeypatch, backend):
    """A plain MegatronModule opts in only for TE, then dispatches by capture/training state."""
    module = _Module(_config(backend))
    # The outer model can use eager/local execution while its vision encoder uses TE.
    vision_layer = _Module(_config())
    mtp_layer = _Module(_config())
    model = SimpleNamespace(
        config=module.config,
        decoder=SimpleNamespace(layers=[module]),
        mtp=SimpleNamespace(layers=[SimpleNamespace(mtp_model_layer=mtp_layer)]),
        vision_model=SimpleNamespace(decoder=SimpleNamespace(layers=[vision_layer])),
    )
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)
    x = torch.tensor(3.0)
    set_te_cuda_graph_current_microbatch(model, 1)
    assert vision_layer.te_cuda_graph_current_microbatch == 1
    assert mtp_layer.te_cuda_graph_current_microbatch == 1
    if backend == "transformer_engine":
        assert module.te_cuda_graph_current_microbatch == 1
    events = []
    module.register_forward_pre_hook(lambda *_: events.append("eager"))
    if backend != "transformer_engine":
        monkeypatch.setattr(
            cuda_graphs, "is_graph_capturing", Mock(side_effect=AssertionError("TE path entered"))
        )
        assert not hasattr(module, "te_cuda_graphs")
        assert not hasattr(module, "te_cuda_graph_current_microbatch")
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
    offset = torch.tensor(2.0)
    torch.testing.assert_close(module(x, offset), x * 2 + offset)
    torch.testing.assert_close(module(x, offset=offset), x * 2 + offset)
    assert events == []  # Capture bypasses ordinary forward hooks.
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)

    def graph(index):
        def replay(hidden_states, offset, is_first_microbatch):
            events.append((index, is_first_microbatch))
            return hidden_states + index + offset

        return replay

    module.te_cuda_graphs = [graph(0), graph(1)]
    for microbatch in (0, 1, 2):
        set_te_cuda_graph_current_microbatch(model, microbatch)
        for replay in (partial(module, x, offset), partial(module, hidden_states=x, offset=offset)):
            events.clear()
            torch.testing.assert_close(replay(), x + microbatch % 2 + offset)
            assert events == ["manual", (microbatch % 2, microbatch == 0)]
    with pytest.raises(AssertionError, match="only Tensor inputs"):
        module(x, 2.0)
    for method in (module.__call__, module._te_cuda_graph_capture, module._te_cuda_graph_replay):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            method(hidden_states=x, unknown_input=None)
    other = _Module(_config())
    assert other.te_cuda_graphs == [] and other.te_cuda_graph_manual_hooks == []

    events.clear()
    module.eval()
    torch.testing.assert_close(module(x), x * 2)
    assert events == ["eager"]


@pytest.mark.parametrize("failure", [None, "graph", "after"])
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
        cuda_graph_stream=Mock(),
        cuda_graph_event=Mock(),
        backward_record=backward_record,
        forward_record=lambda: record("forward"),
        enter_replay=lambda: record("enter"),
        flush_delayed_groups=lambda: record("flush"),
        exit_replay=lambda: record("exit"),
    )

    class Adapter(_Module):
        def _te_cuda_graph_capture_forward(self, hidden_states, offset=1):
            record("capture")
            return self.forward(hidden_states) + offset

        def _te_cuda_graph_prepare_replay(self, hidden_states):
            record("before")
            replay_graph, _ = super()._te_cuda_graph_prepare_replay(hidden_states + 1)
            return replay_graph, partial(self._te_cuda_graph_finish_replay, residual=hidden_states)

        def _te_cuda_graph_finish_replay(self, outputs, residual):
            record("after")
            return outputs + residual

        def get_offload_interface(self):
            return offload

    module = Adapter(_config())
    module.offload_module_in_cuda_graph = True
    module.config.fine_grained_activation_offloading = True
    module.config.delay_offload_until_cuda_graph = True
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", True)
    torch.testing.assert_close(
        module(torch.tensor(3.0), offset=torch.tensor(1.0)), torch.tensor(7.0)
    )
    assert events == ["backward", "capture", "forward"]

    def graph(hidden_states, is_first_microbatch, cuda_graph_stream, cuda_graph_event):
        assert cuda_graph_stream is offload.cuda_graph_stream.return_value
        assert cuda_graph_event is offload.cuda_graph_event.return_value
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
