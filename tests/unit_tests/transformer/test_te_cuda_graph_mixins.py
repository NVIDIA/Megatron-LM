# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The shared mixin interface; layer behavior is covered by existing TE graph tests."""

import pickle
import subprocess
import sys
from functools import partial
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core.transformer.cuda_graphs as cuda_graphs
from megatron.core.extensions.te_cuda_graphs import (
    TECudaGraphMixin,
    get_te_cuda_graph_module_class,
    set_te_cuda_graph_current_microbatch,
)
from megatron.core.pipeline_parallel import combined_1f1b, schedules
from megatron.core.pipeline_parallel.utils import AbstractSchedulePlan
from megatron.core.transformer.hyper_connection_transformer_layer import (
    HyperConnectionTransformerLayer,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec


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

    def forward(self, hidden_states):
        return hidden_states * self.weight

    def is_graphable_for_scope(self, modules):
        return not modules or "attn" in modules


@pytest.mark.parametrize("backend", ["none", "local", "transformer_engine"])
@pytest.mark.parametrize("combined", [False, True])
def test_te_cuda_graph_mixin_dispatch(monkeypatch, backend, combined):
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
        set_input_tensor=Mock(),
        build_schedule_plan=Mock(),
    )
    vision_layer.te_cuda_graphs = [
        lambda hidden_states, is_first_microbatch: hidden_states,
        lambda hidden_states, is_first_microbatch: hidden_states + 1,
    ]
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)
    x = torch.tensor(3.0)
    model.config.timers = None
    model.config.enable_autocast = False
    model.config.fp8 = None
    if combined:
        # Exercise the actual combined schedule, substituting only model computation.
        class Plan(AbstractSchedulePlan):
            """Run the vision layer in place of a full model chunk."""

            event = None

            @staticmethod
            def run(
                f_schedule_plan,
                b_schedule_plan,
                b_grad,
                pre_forward,
                pre_backward,
                post_forward,
                post_backward,
            ):
                """Replay the graph selected by the schedule's microbatch update."""
                return vision_layer(x)

        monkeypatch.setattr(
            schedules.parallel_state, "get_context_parallel_world_size", Mock(return_value=1)
        )
        monkeypatch.setattr(
            schedules.parallel_state, "is_pipeline_last_stage", Mock(return_value=False)
        )
        step = partial(
            combined_1f1b.combined_forward_backward_step,
            forward_step_func=lambda iterator, model, return_schedule_plan: (Plan(), None),
            f_model=model,
            b_model=None,
            b_input_tensor=None,
            b_output_tensor=None,
            b_output_tensor_grad=None,
        )
    else:
        step = partial(
            schedules.forward_step,
            forward_step_func=lambda iterator, model: (vision_layer(x), None),
            model=model,
            cp_group_size=1,
            is_last_stage=False,
        )
    for microbatch, expected_slot in ((0, 0), (1, 1), (None, 1)):
        output, *_ = step(
            data_iterator=None,
            num_microbatches=2,
            input_tensor=None,
            forward_data_store=[],
            config=model.config,
            current_microbatch=microbatch,
        )
        assert mtp_layer.te_cuda_graph_current_microbatch == expected_slot
        assert vision_layer.te_cuda_graph_current_microbatch == expected_slot
        if backend == "transformer_engine":
            assert module.te_cuda_graph_current_microbatch == expected_slot
        torch.testing.assert_close(output, x + expected_slot)
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
    assert events == []  # Capture bypasses ordinary forward hooks.
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", False)

    def graph(index):
        def replay(hidden_states, is_first_microbatch, offset):
            events.append((index, is_first_microbatch))
            return hidden_states + index + offset

        return replay

    module.te_cuda_graphs = [graph(0), graph(1)]
    offset = torch.tensor(2.0)
    for microbatch in (0, 1, 2):
        events.clear()
        set_te_cuda_graph_current_microbatch(model, microbatch)
        torch.testing.assert_close(
            module._te_cuda_graph_replay(hidden_states=x, offset=offset),
            x + microbatch % 2 + offset,
        )
        assert events == ["manual", (microbatch % 2, microbatch == 0)]
    for method in (module.__call__, module._te_cuda_graph_capture, module._te_cuda_graph_replay):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            method(hidden_states=x, unknown_input=None)
    other = _Module(_config())
    assert other.te_cuda_graphs == [] and other.te_cuda_graph_manual_hooks == []

    events.clear()
    module.eval()
    torch.testing.assert_close(module(x), x * 2)
    assert events == ["eager"]


def test_te_cuda_graph_class_serialization():
    """Generated classes retain overrides and reconstruct canonically in a fresh process."""
    original = HyperConnectionTransformerLayer
    original_bases = original.__bases__
    spec = ModuleSpec(module=original)
    adapted = get_te_cuda_graph_module_class(spec.module)
    assert get_te_cuda_graph_module_class(original) is adapted
    assert get_te_cuda_graph_module_class(adapted) is adapted
    assert get_te_cuda_graph_module_class(_Module) is _Module
    assert get_te_cuda_graph_module_class(MegatronModule) is MegatronModule
    assert adapted.__call__ is original.__call__
    assert adapted.get_layer_static_inputs is original.get_layer_static_inputs
    assert original.__bases__ == original_bases and spec.module is original
    payload = pickle.dumps(adapted)
    assert pickle.loads(payload) is adapted

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""
                import pickle
                import sys

                # Unpickle before explicitly importing or calling the class factory.
                restored = pickle.loads(sys.stdin.buffer.read())
                from megatron.core.extensions.te_cuda_graphs import get_te_cuda_graph_module_class
                from megatron.core.transformer.hyper_connection_transformer_layer import (
                    HyperConnectionTransformerLayer,
                )
                assert restored is get_te_cuda_graph_module_class(HyperConnectionTransformerLayer)
                """),
        ],
        input=payload,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


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
        def capture_forward_hook(self, hidden_states, offset=1):
            record("capture")
            return self.forward(hidden_states) + offset

        def before_replay_hook(self, hidden_states):
            record("before")
            replay_graph, _ = super().before_replay_hook(hidden_states + 1)
            return replay_graph, partial(self.after_replay_hook, residual=hidden_states)

        def after_replay_hook(self, outputs, residual):
            record("after")
            return outputs + residual

        def get_offload_interface(self):
            return offload

    module = Adapter(_config())
    module.offload_module_in_cuda_graph = True
    module.config.delay_offload_until_cuda_graph = True
    monkeypatch.setattr(cuda_graphs, "_IS_GRAPH_CAPTURING", True)
    for capture in (
        partial(module, torch.tensor(3.0)),
        partial(module, hidden_states=torch.tensor(3.0), offset=torch.tensor(1.0)),
        partial(module, torch.tensor(3.0), torch.tensor(1.0)),
    ):
        events.clear()
        torch.testing.assert_close(capture(), torch.tensor(7.0))
        assert events == ["backward", "capture", "forward"]

    def graph(hidden_states, is_first_microbatch):
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
