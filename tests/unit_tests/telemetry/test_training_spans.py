# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical training spans and production call-site contracts."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"
TRAINING_PATH = REPO_ROOT / "megatron/training/training.py"
CHECKPOINTING_PATH = REPO_ROOT / "megatron/training/checkpointing.py"

EXPECTED_SPANS = {
    'SPAN_CHECKPOINT_SAVE_STATE_DICT': 'nv.dl.training.checkpoint.save.state_dict',
    'SPAN_CHECKPOINT_SAVE_IO_WRITE': 'nv.dl.training.checkpoint.save.io_write',
    'SPAN_CHECKPOINT_LOAD': 'nv.dl.training.checkpoint.load',
    'SPAN_CHECKPOINT_LOAD_IO_READ': 'nv.dl.training.checkpoint.load.io_read',
    'SPAN_CHECKPOINT_REPORT_MEMORY': 'nv.mlm.checkpoint.report_memory',
    'SPAN_CHECKPOINT_TIMERS_LOG': 'nv.mlm.checkpoint.timers_log',
    'SPAN_CHECKPOINT_FT_HEARTBEAT': 'nv.mlm.checkpoint.ft_heartbeat',
    'SPAN_CHECKPOINT_EXPOSED_SAVE': 'nv.dl.training.checkpoint.exposed_save',
    'SPAN_CHECKPOINT_SAVE': 'nv.dl.training.checkpoint.save',
    'SPAN_CHECKPOINT_SAVE_FINALIZE': 'nv.dl.training.checkpoint.save.finalize',
    'SPAN_TRAINING_ITERATION': 'nv.dl.training.iteration',
    'SPAN_TRAINING_FORWARD_BACKWARD': 'nv.dl.training.iteration.forward_backward',
    'SPAN_TRAINING_OPTIMIZER_STEP': 'nv.dl.training.iteration.optimizer_step',
    'SPAN_GPU_SNIFF_PERIODIC': 'nv.dl.resiliency.gpu_sniff.periodic',
    'SPAN_CUDA_GRAPH_CAPTURE': 'nv.mcore.cuda_graph.capture',
    'SPAN_MEMORY_RECLAIM': 'nv.mcore.memory.reclaim',
    'SPAN_TRAINING_ITERATION_REPORT': 'nv.mlm.train.iteration_report',
    'SPAN_TRAINING_PARAMS_NORM': 'nv.mlm.train.params_norm',
    'SPAN_TRAINING_LOG': 'nv.mlm.train.log',
    'SPAN_TRAINING_FORWARD_PRE_HOOK': 'nv.mlm.train.forward_pre_hook',
    'SPAN_TRAINING_EVALUATE': 'nv.dl.training.evaluate',
    'SPAN_TRAINING_EVALUATE_STEP': 'nv.dl.training.evaluate.step',
}

EXPECTED_ATTRIBUTES = {
    'TRAINING_STEP': 'nv.dl.training.step',
    'TRAINING_ITERATION_IS_FIRST': 'nv.dl.training.iteration.is_first',
    'TRAINING_ITERATION_SKIPPED': 'nv.dl.training.iteration.skipped',
    'TRAINING_OPTIMIZER_UPDATE_SUCCESSFUL': 'nv.dl.training.optimizer.update_successful',
    'TRAINING_EVALUATE_ITERATION': 'nv.dl.training.evaluate.iteration',
    'TRAINING_EVALUATE_ITERATION_COUNT': 'nv.dl.training.evaluate.iteration_count',
    'GPU_SNIFF_TAG': 'nv.dl.resiliency.gpu_sniff.tag',
    'MEMORY_RECLAIM_OPERATION': 'nv.mcore.memory.reclaim.operation',
}


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _function(path, name):
    return next(
        node for node in _tree(path).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _module_string_constants(path):
    constants = {}
    for node in _tree(path).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _otel_constant(node):
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_otel"
    ):
        return node.attr
    return None


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _calls(function, name):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node.func) == name
    ]


def _span_calls(path):
    calls = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        primitive = _call_name(node.func)
        if primitive in {"_otel.managed_span", "_otel.trace_fn"}:
            group = _otel_constant(node.args[0])
            span_name = _otel_constant(node.args[1])
        elif primitive == "_otel.training_iteration_span":
            group = "TRAIN"
            span_name = "SPAN_TRAINING_ITERATION"
        elif primitive == "_otel.span_cm":
            group = None
            span_name = _otel_constant(node.args[0])
        else:
            continue
        if span_name is not None:
            calls.append((span_name, group))
    return calls


def _load_production_telemetry():
    spec = importlib.util.spec_from_file_location("_export_probe_telemetry", TELEMETRY_PATH)
    telemetry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = telemetry
    spec.loader.exec_module(telemetry)
    return telemetry


def _run_exporter_probe():
    from nemo.lens import NemoLensConfig, set_enabled_span_groups, setup_telemetry
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    telemetry = _load_production_telemetry()
    exporter = InMemorySpanExporter()
    config = NemoLensConfig(
        enabled=True, traces_enabled=True, metrics_enabled=False, span_groups="profiling"
    )
    handle = setup_telemetry(
        config, resource_attributes={"nv.dl.rank": 0, "nv.dl.world_size": 1}, span_exporter=exporter
    )

    is_first_iteration = True
    with telemetry.training_iteration_span(37, is_first_iteration) as span:
        telemetry.set_attributes(span, {telemetry.TRAINING_ITERATION_SKIPPED: False})
    is_first_iteration = False
    with telemetry.training_iteration_span(38, is_first_iteration) as span:
        telemetry.set_attributes(span, {telemetry.TRAINING_ITERATION_SKIPPED: True})

    with telemetry.managed_span(telemetry.CKPT, telemetry.SPAN_CHECKPOINT_LOAD):
        pass
    with telemetry.memory_reclaim_span(
        telemetry.CKPT, 38, telemetry.MEMORY_RECLAIM_FREE_OVERLAP_BUFFERS
    ):
        pass
    with telemetry.managed_span(
        telemetry.JOB, telemetry.SPAN_CUDA_GRAPH_CAPTURE, **{telemetry.TRAINING_STEP: 39}
    ):
        pass
    with telemetry.gpu_sniff_span(telemetry.SPAN_GPU_SNIFF_PERIODIC, "iteration 40", 40):
        pass
    with telemetry.gpu_sniff_span("probe.sniff.untagged", "before training"):
        pass
    with telemetry.managed_span(telemetry.EVAL, telemetry.SPAN_TRAINING_EVALUATE):
        telemetry.set_current_evaluation_span_attributes(41, 2)

    with telemetry.checkpoint_exposed_save_span(42):
        with telemetry.checkpoint_save_span(42):
            pass
    failure = RuntimeError("checkpoint application failure")
    try:
        with telemetry.checkpoint_exposed_save_span(42):
            raise failure
    except RuntimeError as caught:
        assert caught is failure
    else:
        raise AssertionError("checkpoint context suppressed the application failure")
    with telemetry.managed_span(telemetry.JOB, "probe.after-checkpoint-failure"):
        pass
    callbacks = []
    finalize = lambda *, blocking: callbacks.append(blocking)
    with telemetry.managed_span(telemetry.CKPT, telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE):
        finalize(blocking=True)
    assert callbacks == [True]
    assert trace.get_tracer_provider().force_flush()
    finished_spans = exporter.get_finished_spans()

    spans = {span.name: span for span in finished_spans}
    iterations = [span for span in finished_spans if span.name == telemetry.SPAN_TRAINING_ITERATION]
    assert [dict(span.attributes) for span in iterations] == [
        {
            telemetry.TRAINING_STEP: 37,
            telemetry.TRAINING_ITERATION_IS_FIRST: True,
            telemetry.TRAINING_ITERATION_SKIPPED: False,
        },
        {
            telemetry.TRAINING_STEP: 38,
            telemetry.TRAINING_ITERATION_IS_FIRST: False,
            telemetry.TRAINING_ITERATION_SKIPPED: True,
        },
    ]
    assert dict(spans[telemetry.SPAN_CHECKPOINT_LOAD].attributes) == {}
    assert dict(spans[telemetry.SPAN_MEMORY_RECLAIM].attributes) == {
        telemetry.TRAINING_STEP: 38,
        telemetry.MEMORY_RECLAIM_OPERATION: telemetry.MEMORY_RECLAIM_FREE_OVERLAP_BUFFERS,
    }
    assert dict(spans[telemetry.SPAN_CUDA_GRAPH_CAPTURE].attributes) == {
        telemetry.TRAINING_STEP: 39
    }
    assert dict(spans[telemetry.SPAN_GPU_SNIFF_PERIODIC].attributes) == {
        telemetry.GPU_SNIFF_TAG: "iteration 40",
        telemetry.TRAINING_STEP: 40,
    }
    assert dict(spans["probe.sniff.untagged"].attributes) == {
        telemetry.GPU_SNIFF_TAG: "before training"
    }
    assert dict(spans[telemetry.SPAN_TRAINING_EVALUATE].attributes) == {
        telemetry.TRAINING_STEP: 41,
        telemetry.TRAINING_EVALUATE_ITERATION_COUNT: 2,
    }
    save = spans[telemetry.SPAN_CHECKPOINT_SAVE]
    exposed = next(
        span
        for span in finished_spans
        if span.name == telemetry.SPAN_CHECKPOINT_EXPOSED_SAVE
        and span.context.span_id == save.parent.span_id
    )
    assert dict(exposed.attributes) == {telemetry.TRAINING_STEP: 42}
    assert dict(save.attributes) == {telemetry.TRAINING_STEP: 42}
    assert save.parent.span_id == exposed.context.span_id
    assert dict(spans[telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE].attributes) == {}
    assert spans["probe.after-checkpoint-failure"].parent is None
    enabled_span_count = len(exporter.get_finished_spans())
    set_enabled_span_groups(frozenset())
    with telemetry.training_iteration_span(43, False):
        pass
    with telemetry.checkpoint_exposed_save_span(43):
        with telemetry.checkpoint_save_span(43):
            pass
    with telemetry.memory_reclaim_span(telemetry.CKPT, 43, "gc_collect"):
        pass
    with telemetry.managed_span(
        telemetry.JOB, telemetry.SPAN_CUDA_GRAPH_CAPTURE, **{telemetry.TRAINING_STEP: 43}
    ):
        pass
    with telemetry.gpu_sniff_span(telemetry.SPAN_GPU_SNIFF_PERIODIC, "disabled", 43):
        pass
    finalize = lambda *, blocking: callbacks.append(blocking)
    with telemetry.managed_span(telemetry.CKPT, telemetry.SPAN_CHECKPOINT_SAVE_FINALIZE):
        finalize(blocking=False)
    assert callbacks == [True, False]
    assert trace.get_tracer_provider().force_flush()
    assert len(exporter.get_finished_spans()) == enabled_span_count
    handle.shutdown()


def test_constants_match_canonical_names():
    constants = _module_string_constants(TELEMETRY_PATH)

    assert {name: constants[name] for name in EXPECTED_SPANS} == EXPECTED_SPANS
    assert {name: constants[name] for name in EXPECTED_ATTRIBUTES} == EXPECTED_ATTRIBUTES
    assert constants["MEMORY_RECLAIM_GC_COLLECT"] == "gc_collect"
    assert constants["MEMORY_RECLAIM_FREE_OVERLAP_BUFFERS"] == "free_overlap_buffers"


def test_renamed_call_sites_use_constants_and_expected_groups():
    calls = _span_calls(TRAINING_PATH) + _span_calls(CHECKPOINTING_PATH)
    grouped = set(calls)

    expected_grouped_calls = {
        ("SPAN_CHECKPOINT_SAVE_STATE_DICT", "CKPT"),
        ("SPAN_CHECKPOINT_SAVE_IO_WRITE", "CKPT"),
        ("SPAN_CHECKPOINT_LOAD", "CKPT"),
        ("SPAN_CHECKPOINT_LOAD_IO_READ", "CKPT"),
        ("SPAN_CHECKPOINT_REPORT_MEMORY", "CKPT"),
        ("SPAN_CHECKPOINT_TIMERS_LOG", "CKPT"),
        ("SPAN_CHECKPOINT_FT_HEARTBEAT", "CKPT"),
        ("SPAN_TRAINING_PARAMS_NORM", "TRAIN"),
        ("SPAN_TRAINING_LOG", "TRAIN"),
        ("SPAN_TRAINING_FORWARD_PRE_HOOK", "TRAIN"),
        ("SPAN_TRAINING_EVALUATE", "EVAL"),
        ("SPAN_TRAINING_EVALUATE_STEP", "EVAL"),
    }
    assert expected_grouped_calls <= grouped

    source = TRAINING_PATH.read_text() + CHECKPOINTING_PATH.read_text()
    executable_strings = {
        node.value
        for path in (TRAINING_PATH, CHECKPOINTING_PATH)
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    helper_owned = {
        "SPAN_TRAINING_FORWARD_BACKWARD",
        "SPAN_TRAINING_OPTIMIZER_STEP",
        "SPAN_TRAINING_ITERATION_REPORT",
        "SPAN_TRAINING_ITERATION",
        "SPAN_CUDA_GRAPH_CAPTURE",
        "SPAN_MEMORY_RECLAIM",
        "SPAN_GPU_SNIFF_PERIODIC",
        "SPAN_CHECKPOINT_EXPOSED_SAVE",
        "SPAN_CHECKPOINT_SAVE",
        "SPAN_CHECKPOINT_SAVE_FINALIZE",
    }
    for name, value in EXPECTED_SPANS.items():
        if name in helper_owned:
            continue
        assert f"_otel.{name}" in source
        assert value not in executable_strings


def test_training_loop_owns_first_iteration_state():
    function = _function(TRAINING_PATH, "train")
    loop = next(node for node in function.body if isinstance(node, ast.While))
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "is_first_iteration"
            for target in node.targets
        )
    ]
    initialization = next(
        node
        for node in assignments
        if isinstance(node.value, ast.Constant) and node.value.value is True
    )
    transition = next(
        node
        for node in assignments
        if isinstance(node.value, ast.Constant) and node.value.value is False
    )
    iteration_call = next(
        call
        for call in _calls(function, "_otel.training_iteration_span")
        if isinstance(call.args[1], ast.Name)
    )
    assert len(iteration_call.args) == 2
    assert isinstance(iteration_call.args[0], ast.Name)
    assert iteration_call.args[0].id == "current_iteration"
    assert iteration_call.args[1].id == "is_first_iteration"
    skipped_call = next(
        call
        for call in _calls(function, "_otel.set_attributes")
        if isinstance(call.args[1], ast.Dict)
        and any(isinstance(value, ast.Call) for value in call.args[1].values)
    )
    assert isinstance(skipped_call.args[0], ast.Name) and skipped_call.args[0].id == "_step_span"
    assert _otel_constant(skipped_call.args[1].keys[0]) == "TRAINING_ITERATION_SKIPPED"
    bool_call = skipped_call.args[1].values[0]
    assert _call_name(bool_call.func) == "bool"
    assert isinstance(bool_call.args[0], ast.Name) and bool_call.args[0].id == "skipped_iter"
    assert initialization.lineno < iteration_call.lineno < transition.lineno
    norm_call = _calls(function, "_should_compute_params_norm")[0]
    assert ast.unparse(norm_call.args[2]) == "is_first_iteration"
    log_call = _calls(function, "training_log")[0]
    log_first = next(
        keyword.value for keyword in log_call.keywords if keyword.arg == "is_first_iteration"
    )
    assert ast.unparse(log_first) == "is_first_iteration"
    assert log_call.lineno < transition.lineno
    assert transition in loop.body
    assert "megatron.train.first_iteration" not in TRAINING_PATH.read_text()


def test_production_helpers_export_span_contracts():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--exporter-probe"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_evaluation_attributes_do_not_mutate_job_parent_when_eval_disabled(
    monkeypatch, telemetry_module
):
    from opentelemetry import trace

    telemetry = telemetry_module

    class RecordingParent:
        def __init__(self):
            self.attributes = {}

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            self.attributes[key] = value

    parent = RecordingParent()
    lookups = []

    def current_span():
        lookups.append(True)
        return parent

    monkeypatch.setattr(telemetry, "is_enabled", lambda group: group == telemetry.JOB)
    monkeypatch.setattr(trace, "get_current_span", current_span)

    assert telemetry.is_enabled(telemetry.JOB) is True
    assert telemetry.is_enabled(telemetry.EVAL) is False
    telemetry.set_current_evaluation_span_attributes(41, 2)

    assert lookups == []
    assert parent.attributes == {}


def test_memory_cuda_and_checkpoint_context_call_sites_use_production_helpers():
    save_function = _function(TRAINING_PATH, "save_checkpoint_and_time")
    callback_function = _function(TRAINING_PATH, "post_training_step_callbacks")
    train_function = _function(TRAINING_PATH, "train")

    memory_calls = _calls(save_function, "_otel.memory_reclaim_span") + _calls(
        callback_function, "_otel.memory_reclaim_span"
    )
    assert len(memory_calls) == 3
    assert [_otel_constant(call.args[0]) for call in memory_calls].count("CKPT") == 2
    assert [_otel_constant(call.args[0]) for call in memory_calls].count("TRAIN") == 1
    assert [_otel_constant(call.args[2]) for call in memory_calls].count(
        "MEMORY_RECLAIM_GC_COLLECT"
    ) == 2
    assert [_otel_constant(call.args[2]) for call in memory_calls].count(
        "MEMORY_RECLAIM_FREE_OVERLAP_BUFFERS"
    ) == 1
    assert [call.args[1].id for call in memory_calls] == ["iteration", "iteration", "training_step"]

    cuda_calls = [
        call
        for call in _calls(train_function, "_otel.managed_span")
        if _otel_constant(call.args[1]) == "SPAN_CUDA_GRAPH_CAPTURE"
    ]
    assert len(cuda_calls) == 1
    assert _otel_constant(cuda_calls[0].args[0]) == "JOB"
    assert ast.unparse(cuda_calls[0].keywords[0].value) == (
        "{_otel.TRAINING_STEP: current_iteration}"
    )

    outer_save = next(
        node
        for node in save_function.body
        if isinstance(node, ast.With)
        and _call_name(node.items[0].context_expr.func) == "_otel.checkpoint_exposed_save_span"
    )
    assert ast.unparse(outer_save.items[0].context_expr.args[0]) == "iteration"
    body_source = ast.unparse(ast.Module(body=outer_save.body, type_ignores=[]))
    for operation in (
        "force_param_sync",
        "timers('interval-time').stop()",
        "save_checkpoint",
        "timers.log([timer_key])",
        "energy_monitor.resume()",
        "timers('interval-time', log_level=0).start(barrier=True)",
    ):
        assert operation in body_source


def test_evaluate_uses_outer_training_step_and_canonical_attributes():
    evaluate = _function(TRAINING_PATH, "evaluate")
    wrapper = _function(TRAINING_PATH, "evaluate_and_print_results")

    assert any(argument.arg == "training_step" for argument in evaluate.args.args)
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
        and node.attr == "curr_iteration"
        for node in ast.walk(evaluate)
    )

    attribute_calls = _calls(evaluate, "_otel.set_current_evaluation_span_attributes")
    assert len(attribute_calls) == 2
    assert all(
        isinstance(call.args[0], ast.Name) and call.args[0].id == "training_step"
        for call in attribute_calls
    )

    evaluate_call = next(call for call in _calls(wrapper, "evaluate"))
    assert isinstance(evaluate_call.args[5], ast.Name)
    assert evaluate_call.args[5].id == "iteration"


def test_gpu_sniff_call_sites_distinguish_periodic_and_startup_steps():
    sniff = _function(TRAINING_PATH, "_run_gpu_sniff_test")
    helper_calls = _calls(sniff, "_otel.gpu_sniff_span")
    assert len(helper_calls) == 1
    assert [argument.id for argument in helper_calls[0].args] == [
        "span_name",
        "tag",
        "training_step",
    ]

    callbacks = _function(TRAINING_PATH, "post_training_step_callbacks")
    periodic = _calls(callbacks, "_run_gpu_sniff_test")
    assert len(periodic) == 1
    step_keyword = next(
        keyword for keyword in periodic[0].keywords if keyword.arg == "training_step"
    )
    assert isinstance(step_keyword.value, ast.Name)
    assert step_keyword.value.id == "training_step"

    train = _function(TRAINING_PATH, "train")
    startup = next(
        call
        for call in _calls(train, "_run_gpu_sniff_test")
        if any(keyword.arg == "span_name" for keyword in call.keywords)
    )
    assert not any(keyword.arg == "training_step" for keyword in startup.keywords)


if __name__ == "__main__":
    assert sys.argv[1:] == ["--exporter-probe"]
    _run_exporter_probe()
