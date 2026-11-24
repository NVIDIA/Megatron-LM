# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU contract tests for L1 observers; not a TE/CUDA integration test."""

import ast
import contextlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.determinism import te
from megatron.core.determinism.context import (
    DeterminismContext,
    determinism_context,
    get_determinism_context,
)
from megatron.core.determinism.te import (
    bind_layer_trace_context,
    suspend_layer_trace,
    trace_transformer_layer_forward,
)
from megatron.core.determinism.trace import (
    TraceConfig,
    close_determinism_trace,
    get_determinism_trace,
    initialize_determinism_trace,
)


class Layer(torch.nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.config = SimpleNamespace(determinism_trace_layers=enabled)
        self.layer_number = 3

    @trace_transformer_layer_forward
    def forward(self, tensor, **kwargs):
        return tensor * tensor, kwargs.get("context")


@pytest.fixture
def trace(tmp_path):
    recorder = initialize_determinism_trace(TraceConfig(tmp_path, rank=0))
    yield recorder
    close_determinism_trace()


def records(trace):
    trace.flush()
    return [json.loads(line) for line in trace.output_path.read_text().splitlines()]


def isolated_function(relative_path, name, namespace, class_name=None):
    """Load the production body without initializing GPU-only parent modules."""
    root = Path(__file__).resolve().parents[3]
    path = root / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name is not None:
        body = next(
            node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    function = next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_disabled_observer_does_not_lookup_recorder_or_register_hooks(monkeypatch):
    lookup = Mock(side_effect=AssertionError("disabled observer accessed recorder"))
    monkeypatch.setattr(te, "get_determinism_trace", lookup)
    tensor = torch.tensor([2.0], requires_grad=True)
    output, _ = Layer(enabled=False)(tensor)
    output.sum().backward()
    assert tensor.grad.item() == 4
    assert not tensor._backward_hooks
    lookup.assert_not_called()


def test_enabled_without_recorder_is_noop():
    assert get_determinism_trace() is None
    tensor = torch.tensor([2.0], requires_grad=True)
    output, _ = Layer()(tensor)
    assert not tensor._backward_hooks
    assert not output._backward_hooks
    output.sum().backward()
    assert tensor.grad.item() == 4


def test_interleaved_backward_uses_forward_identity_and_preserves_gradients(trace):
    layer = Layer()
    inputs, outputs = [], []
    for microbatch in (0, 1):
        tensor = torch.tensor([2.0], requires_grad=True)
        with determinism_context(DeterminismContext(7, microbatch, "forward")):
            output, context = layer(tensor, context="unchanged")
        assert context == "unchanged"
        inputs.append(tensor)
        outputs.append(output)
    unrelated = DeterminismContext(8, 9, "forward")
    with determinism_context(unrelated):
        for output in reversed(outputs):
            output.sum().backward()
            assert get_determinism_context() is unrelated
    rows = records(trace)
    assert [(row["name"], row["phase"], row["microbatch"]) for row in rows] == [
        (f"te.layer.3.{boundary}", phase, microbatch)
        for phase, microbatches, boundaries in [
            ("forward", (0, 1), ("in", "out")),
            ("backward", (1, 0), ("out", "in")),
        ]
        for microbatch in microbatches
        for boundary in boundaries
    ]
    assert all(row["iteration"] == 7 for row in rows)
    assert all(row["tensor"]["mode"] == "metadata" for row in rows)
    for tensor in inputs:
        torch.testing.assert_close(tensor.grad, torch.tensor([4.0]))


@pytest.mark.parametrize("mode", ["summary", "sampled", "full"])
def test_observer_respects_capture_mode_and_iteration_selection(tmp_path, mode):
    trace = initialize_determinism_trace(
        TraceConfig(tmp_path, rank=0, mode=mode, iteration_spec="2")
    )
    try:
        layer = Layer()
        for iteration in (1, 2):
            with determinism_context(DeterminismContext(iteration, 0, "forward")):
                layer(torch.tensor([2.0], requires_grad=True))[0].sum().backward()
        rows = records(trace)
        assert len(rows) == 4
        assert all(row["iteration"] == 2 for row in rows)
        assert all(row["occurrence"] == 0 for row in rows)
        assert all(row["tensor"]["mode"] == mode for row in rows)
        assert all(row["tensor"]["value_observed"] for row in rows)
    finally:
        close_determinism_trace()


def test_missing_context_and_no_grad_forward(trace):
    with torch.no_grad():
        Layer()(torch.ones(1))
    rows = records(trace)
    assert len(rows) == 2
    assert rows[0]["metadata"]["missing_context"] == ["iteration", "microbatch"]
    assert rows[0]["metadata"]["cuda_graph_replay_observed"] is False
    assert all(row["iteration"] is None for row in rows)


@pytest.mark.parametrize("reentrant", [True, False])
def test_production_recompute_closure_restores_microbatch(trace, reentrant):
    layer = Layer()
    config = SimpleNamespace(
        enable_mhc_connections=False,
        distribute_saved_activations=False,
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    block = SimpleNamespace(
        config=config,
        layers=[layer],
        num_layers_per_pipeline_rank=1,
        pg_collection=SimpleNamespace(tp=None),
    )
    namespace = dict(
        nullcontext=contextlib.nullcontext,
        TransformerLayer=Layer,
        bind_layer_trace_context=bind_layer_trace_context,
        get_determinism_trace=get_determinism_trace,
        get_determinism_context=get_determinism_context,
        determinism_context=determinism_context,
        observe_layer_residuals=lambda *args: None,
        precision_aware_checkpoint=lambda function, config, group, *args, **kwargs: checkpoint(
            function, *args, use_reentrant=reentrant
        ),
    )
    forward = isolated_function("megatron/core/recompute.py", "checkpointed_forward", namespace)
    inputs, outputs = [], []
    for microbatch in (0, 1):
        tensor = torch.tensor([2.0], requires_grad=True)
        with determinism_context(DeterminismContext(7, microbatch, "forward")):
            outputs.append(forward(block, tensor, None, None, None, None, None, None, False))
        inputs.append(tensor)
    unrelated = DeterminismContext(8, 9, "forward")
    with determinism_context(unrelated):
        for output in reversed(outputs):
            output.sum().backward()
            assert get_determinism_context() is unrelated
    rows = records(trace)
    backward = [row for row in rows if row["phase"] == "backward"]
    assert [(row["microbatch"], row["name"]) for row in backward] == [
        (microbatch, f"te.layer.3.{boundary}")
        for microbatch in (1, 0)
        for boundary in ("out", "in")
    ]
    assert all(row["iteration"] == 7 for row in rows)
    for tensor in inputs:
        torch.testing.assert_close(tensor.grad, torch.tensor([4.0]))


@pytest.mark.parametrize("reentrant", [True, False])
def test_production_mtp_checkpoint_restores_microbatch(trace, reentrant):
    layer = Layer()
    mtp = SimpleNamespace(
        is_hybrid_mtp=False,
        config=SimpleNamespace(
            fp8=False,
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=False,
        ),
        tp_group=None,
        _proj_and_transformer_layer=lambda hidden_states, **kwargs: layer(hidden_states)[0],
    )
    forward = isolated_function(
        "megatron/core/transformer/multi_token_prediction.py",
        "_checkpointed_forward",
        dict(
            nullcontext=contextlib.nullcontext,
            bind_layer_trace_context=bind_layer_trace_context,
            precision_aware_checkpoint=lambda function, config, group, *args, **kwargs: checkpoint(
                function, *args, use_reentrant=reentrant
            ),
        ),
        class_name="MultiTokenPredictionLayer",
    )
    inputs, outputs = [], []
    for microbatch in (0, 1):
        tensor = torch.tensor([2.0], requires_grad=True)
        with determinism_context(DeterminismContext(7, microbatch, "forward")):
            outputs.append(forward(mtp, tensor, torch.zeros(1)))
        inputs.append(tensor)
    unrelated = DeterminismContext(8, 9, "forward")
    with determinism_context(unrelated):
        for output in reversed(outputs):
            output.sum().backward()
            assert get_determinism_context() is unrelated
    rows = records(trace)
    backward = [row for row in rows if row["phase"] == "backward"]
    assert [row["microbatch"] for row in backward] == [1, 1, 0, 0]
    assert [row["name"] for row in backward] == [
        f"te.layer.3.{boundary}" for _ in (1, 0) for boundary in ("out", "in")
    ]
    assert all(row["iteration"] == 7 for row in rows)
    for tensor in inputs:
        torch.testing.assert_close(tensor.grad, torch.tensor([4.0]))


@pytest.mark.parametrize("mode", ["summary", "sampled", "full"])
@pytest.mark.parametrize("non_loss", ["callback", "schedule"])
@pytest.mark.parametrize("fail", [False, True])
def test_production_evaluate_suspends_l1_and_restores(tmp_path, mode, non_loss, fail):
    trace = initialize_determinism_trace(
        TraceConfig(tmp_path, rank=0, mode=mode, iteration_spec="7")
    )
    try:
        layer = Layer()
        with determinism_context(DeterminismContext(7, 0, "forward")):
            layer(torch.tensor([2.0], requires_grad=True))[0].sum().backward()
        assert get_determinism_context().iteration is None
        before = records(trace)
        calls = []

        def schedule(**kwargs):
            assert not torch.is_grad_enabled()
            assert get_determinism_context().iteration is None
            output, _ = layer(torch.ones(1))
            torch.testing.assert_close(output, torch.ones(1))
            calls.append(kwargs.get("collect_non_loss_data", False))
            if fail:
                raise RuntimeError("evaluation failed")
            return []

        cfg = SimpleNamespace(
            validation=SimpleNamespace(
                eval_global_batch_size=1, eval_micro_batch_size=1, eval_iters=1
            ),
            train=SimpleNamespace(empty_unused_memory_level=0, exit_duration_in_mins=None),
        )
        args = SimpleNamespace(
            cuda_graph_impl="none",
            moe_expert_rank_capacity_factor=None,
            seq_length=1,
            decoder_seq_length=None,
        )
        timers = Mock()
        state = SimpleNamespace(consumed_valid_samples=0)
        evaluate = isolated_function(
            "megatron/training/training.py",
            "evaluate",
            dict(
                torch=torch,
                suspend_layer_trace=suspend_layer_trace,
                normalize_callbacks=lambda manager: Mock(),
                get_run_config=lambda: cfg,
                get_args=lambda: args,
                get_timers=lambda: timers,
                get_train_state=lambda: state,
                get_rerun_state_machine=lambda: Mock(),
                RerunMode=SimpleNamespace(DISABLED="disabled"),
                get_batch_parallel_size=lambda args: 1,
                get_forward_backward_func=lambda **kwargs: schedule,
                get_attr_wrapped_model=lambda *args: SimpleNamespace(dp_cp=None, pp=None),
                has_nvidia_modelopt=False,
                ft_integration=Mock(),
                _otel_managed_span=lambda *args, **kwargs: contextlib.nullcontext(),
                is_pp_last_stage=lambda group: False,
                is_last_rank=lambda: True,
            ),
        )
        kwargs = dict(
            forward_step_func=None,
            data_iterator=None,
            model=[layer],
            process_non_loss_data_func=object() if non_loss == "schedule" else None,
            non_loss_data_func=(lambda model: schedule()) if non_loss == "callback" else None,
            config=SimpleNamespace(timers=timers),
        )
        if fail:
            with pytest.raises(RuntimeError, match="evaluation failed"):
                evaluate(**kwargs)
        else:
            assert evaluate(**kwargs) == ({}, [], False)
            assert state.consumed_valid_samples == 1
            assert calls == [False, non_loss == "schedule"]
            assert layer.training
        assert records(trace) == before
        with determinism_context(DeterminismContext(7, 1, "forward")):
            layer(torch.tensor([2.0], requires_grad=True))[0].sum().backward()
        rows = records(trace)
        assert len(rows) == 8
        assert all(row["iteration"] == 7 for row in rows)
        assert all(row["tensor"]["mode"] == mode for row in rows)
        with pytest.raises(ValueError, match="iteration"):
            layer(torch.ones(1))
    finally:
        close_determinism_trace()


def test_production_forward_step_sets_microbatch_and_restores_on_failure(trace):
    namespace = dict(
        torch=torch,
        contextlib=contextlib,
        replace=replace,
        get_determinism_trace=get_determinism_trace,
        get_determinism_context=get_determinism_context,
        determinism_context=determinism_context,
        set_current_microbatch=lambda *args: None,
        get_attr_wrapped_model=lambda *args: lambda tensor: None,
        forward_step_calc_loss=lambda model, output, *args: (output, 0),
    )
    forward = isolated_function(
        "megatron/core/pipeline_parallel/schedules.py", "forward_step", namespace
    )
    config = SimpleNamespace(timers=None, enable_autocast=False)
    observed = []

    def step(*args):
        observed.append(get_determinism_context())
        return torch.ones(1), None

    parent = DeterminismContext(7, 99, "unspecified")
    with determinism_context(parent):
        forward(step, None, object(), 2, None, [], config, 1, current_microbatch=0)
        assert get_determinism_context() is parent
        forward(step, None, object(), 2, None, [], config, 1, current_microbatch=None)
        assert get_determinism_context() is parent
        with pytest.raises(RuntimeError, match="failed"):
            forward(
                Mock(side_effect=RuntimeError("failed")),
                None,
                object(),
                2,
                None,
                [],
                config,
                1,
                current_microbatch=1,
            )
        assert get_determinism_context() is parent
    assert observed == [DeterminismContext(7, 0, "forward"), DeterminismContext(7, None, "forward")]
