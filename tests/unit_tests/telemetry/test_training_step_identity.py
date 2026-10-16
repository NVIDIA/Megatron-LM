# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stable attempted-step and committed-checkpoint identities."""

import ast
import importlib
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_PATH = REPO_ROOT / "megatron/training/training.py"


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _function(path, name):
    return next(
        node for node in _tree(path).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _calls(node, name):
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate.func) == name
    ]


@pytest.fixture
def span_exporter():
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
    try:
        in_memory_export = importlib.import_module("opentelemetry.sdk.trace.export")
        exporter_type = in_memory_export.InMemorySpanExporter
    except AttributeError:
        in_memory_export = importlib.import_module(
            "opentelemetry.sdk.trace.export.in_memory_span_exporter"
        )
        exporter_type = in_memory_export.InMemorySpanExporter

    exporter = exporter_type()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    try:
        yield provider.get_tracer("megatron-trace-lifecycle-test"), exporter
    finally:
        provider.shutdown()


def test_current_iteration_identity_is_reused_across_the_loop_pass():
    train = _function(TRAINING_PATH, "train")
    loop = next(node for node in train.body if isinstance(node, ast.While))
    loop_source = ast.unparse(loop)

    assert "training_iteration_span(current_iteration, is_first_iteration)" in loop_source
    assert "SPAN_CUDA_GRAPH_CAPTURE" in loop_source
    assert "{_otel.TRAINING_STEP: current_iteration}" in loop_source
    assert (
        "training_log(loss_dict, total_loss_dict, learning_rate, current_iteration" in loop_source
    )
    assert (
        "post_training_step_callbacks(model, optimizer, opt_param_scheduler, current_iteration"
        in loop_source
    )
    assert (
        "checkpoint_and_decide_exit(model, optimizer, opt_param_scheduler, current_iteration"
        in loop_source
    )
    assert "completed_iterations += 1" in loop_source
    assert "train_state.iteration += 1" in loop_source
    assert "assert completed_iterations == current_iteration" in loop_source


@pytest.mark.parametrize(
    "step,is_first,log_params_norm,tensorboard_dir,expected",
    [
        (200001, True, True, None, True),
        (200002, False, True, None, False),
        (200010, False, True, None, True),
        (200005, False, True, "tensorboard", True),
        (200005, False, True, None, False),
        (200001, True, False, None, False),
    ],
)
def test_parameter_norm_reporting_uses_live_loop_state(
    step, is_first, log_params_norm, tensorboard_dir, expected
):
    """Exercise the production call with only the iteration state the loop owns."""
    train = _function(TRAINING_PATH, "train")
    call = _calls(train, "_should_compute_params_norm")[0]
    helper = _function(TRAINING_PATH, "_should_compute_params_norm")
    logger_config = types.SimpleNamespace(
        log_params_norm=log_params_norm,
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        tensorboard_log_interval=5,
    )
    namespace = {
        "args": types.SimpleNamespace(),
        "get_run_config": lambda: types.SimpleNamespace(logger=logger_config),
        "current_iteration": step,
        "is_first_iteration": is_first,
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), TRAINING_PATH, "exec"), namespace)
    assert eval(compile(ast.Expression(call), TRAINING_PATH, "eval"), namespace) is expected


def test_dummy_skip_uses_skipped_iteration_span_without_consuming_first_state():
    train = _function(TRAINING_PATH, "train")
    dummy_call = _calls(train, "dummy_train_step")[0]
    dummy_if = next(
        node
        for node in ast.walk(train)
        if isinstance(node, ast.If) and dummy_call in list(ast.walk(node))
    )
    source = ast.unparse(dummy_if)

    assert "training_iteration_span(current_iteration, False)" in source
    assert "set_attributes(_step_span, {_otel.TRAINING_ITERATION_SKIPPED: True})" in source
    assert "train_state.iteration += 1" in source
    assert "train_state.consumed_train_samples += batch_size" in source
    assert "train_state.skipped_train_samples += batch_size" in source
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "is_first_iteration"
            for target in node.targets
        )
        for node in ast.walk(dummy_if)
    )


def test_pre_and_post_iteration_checkpoints_use_saved_state_identity():
    train = _function(TRAINING_PATH, "train")
    save_calls = _calls(train, "save_checkpoint_and_time")

    assert len(save_calls) == 2
    saved_steps = {call.args[0].id for call in save_calls if isinstance(call.args[0], ast.Name)}
    assert saved_steps == {"completed_iterations"}
    dispatch = _calls(train, "checkpoint_and_decide_exit")[0]
    assert ast.unparse(dispatch.args[3]) == "current_iteration"
    helper = _function(TRAINING_PATH, "checkpoint_and_decide_exit")
    assert all(
        ast.unparse(call.args[0]) == "iteration"
        for call in _calls(helper, "save_checkpoint_and_time")
    )


@pytest.mark.parametrize("enabled", [False, True], ids=["telemetry-off", "telemetry-on"])
@pytest.mark.parametrize("diagnostic_exit", [False, True], ids=["committed", "diagnostic"])
@pytest.mark.parametrize("train_iters", [42, 100], ids=["final-attempt", "mid-run"])
def test_checkpoint_progress_at_train_step_boundary(
    monkeypatch, span_exporter, telemetry_module, enabled, diagnostic_exit, train_iters
):
    """Execute production branches, not a copied checkpoint decision or a GPU reload."""
    telemetry = telemetry_module
    tracer, exporter = span_exporter
    monkeypatch.setattr(telemetry, "is_enabled", lambda group: enabled)

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        if enabled:
            with span_exporter[0].start_as_current_span(name, attributes=attributes) as span:
                yield span
        else:
            yield None

    monkeypatch.setattr(telemetry, "managed_span", managed_span)
    args = types.SimpleNamespace(
        train_iters=train_iters, skip_train=False, consumed_train_samples=41 * 8
    )
    saved = []

    def save(progress, *unused_args, **unused_kwargs):
        with telemetry.checkpoint_exposed_save_span(progress):
            with telemetry.checkpoint_save_span(progress):
                saved.append((progress, args.consumed_train_samples))

    def train_step(*unused_args, **kwargs):
        assert kwargs["iteration"] == 41
        return {}, diagnostic_exit, diagnostic_exit, diagnostic_exit, 0, None, None, None, None

    train = _function(TRAINING_PATH, "train")
    loop = next(node for node in train.body if isinstance(node, ast.While))
    # Keep the real branch nesting, exit, progress increment and sample increment.
    # Omit unrelated scheduling and GPU work; external work is stubbed below.
    boundary = next(
        i
        for i, node in enumerate(loop.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.skip_train"
    )
    assert ast.unparse(loop.body[boundary + 1].test) == "should_checkpoint"
    assert ast.unparse(loop.body[boundary + 2].test) == "should_exit"
    increment = next(
        node
        for node in loop.body
        if isinstance(node, ast.AugAssign) and ast.unparse(node.target) == "completed_iterations"
    )
    samples = next(
        node
        for node in loop.body
        if isinstance(node, ast.AugAssign)
        and ast.unparse(node.target) == "args.consumed_train_samples"
    )
    dispatch = next(node for node in loop.body if _calls(node, "checkpoint_and_decide_exit"))
    selected = loop.body[:1] + loop.body[boundary : boundary + 3] + [increment, samples, dispatch]
    assert [node.lineno for node in selected] == sorted(node.lineno for node in selected)
    # One pass suffices: the original while condition is also evaluated after a
    # simulated reload below to check that the final attempt remains runnable.
    loop.body = selected + [ast.Break()]
    namespace = dict.fromkeys(
        [
            "model",
            "optimizer",
            "opt_param_scheduler",
            "checkpointing_context",
            "train_data_iterator",
            "forward_step_func",
            "config",
            "forward_backward_func",
            "pg_collection",
            "p2p_communicator",
            "tensor_metric_observer",
            "_maybe_raise_workload_exception",
        ]
    )
    namespace.update(
        args=args,
        completed_iterations=41,
        start_iteration=41,
        iteration_sequences=8,
        num_floating_point_operations_so_far=0,
        _finished_training=lambda completed: completed >= train_iters,
        _otel=telemetry,
        is_first_iteration=True,
        ft_integration=types.SimpleNamespace(
            on_training_step_start=lambda **_kwargs: None, on_training_step_end=lambda: None
        ),
        callback_manager=types.SimpleNamespace(
            callback_context=types.SimpleNamespace(), trigger=lambda *_args, **_kwargs: None
        ),
        set_log_quantization_types=lambda _enabled: None,
        train_step=train_step,
        save_checkpoint_and_time=save,
        # Exercise the real ordinary checkpoint dispatch helper too.
        get_args=lambda: types.SimpleNamespace(
            exit_signal_handler=False,
            save=True,
            save_interval=1,
            non_persistent_save_interval=None,
            exit_duration_in_mins=None,
            exit_interval=None,
            phase_transition_iterations=None,
        ),
        get_timers=lambda: None,
        in_phase_transition=lambda args: False,
    )
    helper = _function(TRAINING_PATH, "checkpoint_and_decide_exit")
    module = ast.fix_missing_locations(ast.Module(body=[helper, loop], type_ignores=[]))
    exec(compile(module, str(TRAINING_PATH), "exec"), namespace)

    expected = 41 if diagnostic_exit else 42
    assert saved == [(expected, expected * 8)]
    assert namespace["completed_iterations"] == expected
    assert args.consumed_train_samples == expected * 8
    assert namespace["current_iteration"] == 42
    namespace["completed_iterations"] = saved[0][0]
    runnable = eval(compile(ast.Expression(loop.test), str(TRAINING_PATH), "eval"), namespace)
    assert runnable == (expected < train_iters)
    if diagnostic_exit:
        assert runnable  # In particular, the final requested attempt is not lost.

    spans = list(exporter.get_finished_spans())
    if not enabled:
        assert not spans
    else:
        by_name = {span.name: span for span in spans}
        assert by_name[telemetry.SPAN_TRAINING_ITERATION].attributes[telemetry.TRAINING_STEP] == 42
        for name in (telemetry.SPAN_CHECKPOINT_EXPOSED_SAVE, telemetry.SPAN_CHECKPOINT_SAVE):
            assert by_name[name].attributes[telemetry.TRAINING_STEP] == expected
