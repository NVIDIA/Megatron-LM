# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical MCore call sites and gradient-sync callback transport."""

import ast
from contextlib import contextmanager
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

REPO_ROOT = Path(__file__).resolve().parents[3]


TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"


SCHEDULES_PATH = REPO_ROOT / "megatron/core/pipeline_parallel/schedules.py"


P2P_PATH = REPO_ROOT / "megatron/core/pipeline_parallel/p2p_communication.py"


DDP_PATH = REPO_ROOT / "megatron/core/distributed/distributed_data_parallel.py"


TRANSFORMER_LAYER_PATH = REPO_ROOT / "megatron/core/transformer/transformer_layer.py"


MAMBA_LAYER_PATH = REPO_ROOT / "megatron/core/ssm/mamba_layer.py"


FULLY_PARALLEL_PATH = REPO_ROOT / "megatron/core/dist_checkpointing/strategies/fully_parallel.py"


TORCH_CHECKPOINT_PATH = REPO_ROOT / "megatron/core/dist_checkpointing/strategies/torch.py"


EXPECTED_MCORE_SPANS = {
    "SPAN_CHECKPOINT_LOAD_READ_METADATA": "nv.mcore.checkpoint.load.read_metadata",
    "SPAN_CHECKPOINT_LOAD_MCORE_TO_PYT": "nv.mcore.checkpoint.load.mcore_to_pyt",
    "SPAN_CHECKPOINT_LOAD_FULLY_PARALLEL_PLAN": "nv.mcore.checkpoint.load.fully_parallel_plan",
    "SPAN_CHECKPOINT_LOAD_IO": "nv.mcore.checkpoint.load.io",
    "SPAN_CHECKPOINT_LOAD_EXCHANGE": "nv.mcore.checkpoint.load.exchange",
    "SPAN_MICROBATCH_FORWARD": "nv.mcore.microbatch.forward",
    "SPAN_MICROBATCH_BACKWARD": "nv.mcore.microbatch.backward",
    "SPAN_P2P_RECV_FORWARD": "nv.mcore.p2p.recv_forward",
    "SPAN_P2P_RECV_BACKWARD": "nv.mcore.p2p.recv_backward",
    "SPAN_P2P_SEND_FORWARD": "nv.mcore.p2p.send_forward",
    "SPAN_P2P_SEND_BACKWARD": "nv.mcore.p2p.send_backward",
    "SPAN_GRAD_SYNC_START": "nv.mcore.grad_sync.start",
    "SPAN_GRAD_SYNC_FINISH": "nv.mcore.grad_sync.finish",
    "SPAN_LAYER_FORWARD": "nv.mcore.layer.forward",
    "SPAN_LAYER_SELF_ATTENTION": "nv.mcore.layer.self_attention",
    "SPAN_LAYER_MLP": "nv.mcore.layer.mlp",
    "SPAN_LAYER_MAMBA": "nv.mcore.layer.mamba",
}


EXPECTED_MCORE_ATTRIBUTES = {
    "GRAD_SYNC_START_SITE": "nv.mcore.grad_sync.start.site",
    "LAYER_NUMBER": "nv.mcore.layer.number",
}


EXPECTED_GRAD_SYNC_SITES = {
    "GRAD_SYNC_SITE_INTERLEAVED_BACKWARD": "interleaved_backward",
    "GRAD_SYNC_SITE_INTERLEAVED_COOLDOWN": "interleaved_cooldown",
    "GRAD_SYNC_SITE_NON_INTERLEAVED_COOLDOWN": "non_interleaved_cooldown",
}


MCORE_CALLSITE_PATHS = (
    SCHEDULES_PATH,
    P2P_PATH,
    DDP_PATH,
    TRANSFORMER_LAYER_PATH,
    MAMBA_LAYER_PATH,
    FULLY_PARALLEL_PATH,
    TORCH_CHECKPOINT_PATH,
)


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _nested_function(path, name):
    return next(
        node
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.FunctionDef) and node.name == name
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


def _managed_span_constant(call):
    primitive = _call_name(call.func)
    if primitive == "_otel.managed_span":
        return _otel_constant(call.args[1])
    return None


def _contains_otel_constant(node, name):
    return any(_otel_constant(candidate) == name for candidate in ast.walk(node))


def _grad_sync_blocks():
    tree = _tree(SCHEDULES_PATH)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        context = node.items[0].context_expr
        if not isinstance(context, ast.Call) or _call_name(context.func) != "_otel.managed_span":
            continue
        if _otel_constant(context.args[1]) != "SPAN_GRAD_SYNC_START":
            continue
        parent = parents[node]
        statement_list = next(
            value
            for _field, value in ast.iter_fields(parent)
            if isinstance(value, list) and node in value
        )
        index = statement_list.index(node)
        blocks.append((node, statement_list[index - 2 : index + 1], parents))
    return sorted(blocks, key=lambda item: item[0].lineno)


def test_mcore_constants_match_canonical_names():
    constants = _module_string_constants(TELEMETRY_PATH)

    assert {name: constants[name] for name in EXPECTED_MCORE_SPANS} == EXPECTED_MCORE_SPANS
    assert {
        name: constants[name] for name in EXPECTED_MCORE_ATTRIBUTES
    } == EXPECTED_MCORE_ATTRIBUTES
    assert {name: constants[name] for name in EXPECTED_GRAD_SYNC_SITES} == EXPECTED_GRAD_SYNC_SITES


def test_mcore_call_sites_use_constants_and_detail_group():
    calls = [call for path in MCORE_CALLSITE_PATHS for call in _span_calls(path)]
    expected_counts = {
        "SPAN_CHECKPOINT_LOAD_READ_METADATA": 1,
        "SPAN_CHECKPOINT_LOAD_MCORE_TO_PYT": 1,
        "SPAN_CHECKPOINT_LOAD_FULLY_PARALLEL_PLAN": 1,
        "SPAN_CHECKPOINT_LOAD_IO": 1,
        "SPAN_CHECKPOINT_LOAD_EXCHANGE": 1,
        "SPAN_MICROBATCH_FORWARD": 1,
        "SPAN_MICROBATCH_BACKWARD": 1,
        "SPAN_P2P_RECV_FORWARD": 1,
        "SPAN_P2P_RECV_BACKWARD": 1,
        "SPAN_P2P_SEND_FORWARD": 1,
        "SPAN_P2P_SEND_BACKWARD": 1,
        "SPAN_GRAD_SYNC_FINISH": 1,
        "SPAN_LAYER_FORWARD": 2,
        "SPAN_LAYER_SELF_ATTENTION": 2,
        "SPAN_LAYER_MLP": 1,
        "SPAN_LAYER_MAMBA": 1,
    }

    for span_name, count in expected_counts.items():
        assert calls.count((span_name, "DETAIL")) == count

    source = "".join(path.read_text() for path in MCORE_CALLSITE_PATHS)
    for name, value in EXPECTED_MCORE_SPANS.items():
        if name != "SPAN_GRAD_SYNC_START":
            assert f"_otel.{name}" in source
        assert value not in source


def test_layer_forward_spans_carry_canonical_layer_number():
    calls = []
    for path in (TRANSFORMER_LAYER_PATH, MAMBA_LAYER_PATH):
        calls.extend(
            node
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.Call) and _managed_span_constant(node) == "SPAN_LAYER_FORWARD"
        )

    assert len(calls) == 2
    for call in calls:
        attributes = next(keyword.value for keyword in call.keywords if keyword.arg is None)
        assert isinstance(attributes, ast.Dict)
        assert len(attributes.keys) == 1
        assert _otel_constant(attributes.keys[0]) == "LAYER_NUMBER"
        value = attributes.values[0]
        assert isinstance(value, ast.Attribute)
        assert isinstance(value.value, ast.Name) and value.value.id == "self"
        assert value.attr == "layer_number"


def test_schedule_owns_each_grad_sync_managed_block():
    blocks = _grad_sync_blocks()
    assert len(blocks) == 3
    sites = []
    for managed, statements, _parents in blocks:
        context = managed.items[0].context_expr
        assert _otel_constant(context.args[0]) == "DETAIL"
        assert _otel_constant(context.args[1]) == "SPAN_GRAD_SYNC_START"
        attributes = next(keyword.value for keyword in context.keywords if keyword.arg is None)
        assert _otel_constant(attributes.keys[0]) == "GRAD_SYNC_START_SITE"
        sites.append(_otel_constant(attributes.values[0]))
        assert [ast.unparse(statement.targets[0]) for statement in statements[:2]] == [
            "_grad_sync_func",
            "_grad_sync_params",
        ]
        invocation = managed.body[0].value
        assert ast.unparse(invocation) == "_grad_sync_func(_grad_sync_params)"
        guard = managed
        while not (
            isinstance(guard, ast.If)
            and "config.grad_sync_func is not None" in ast.unparse(guard.test)
        ):
            guard = _parents[guard]
    assert set(sites) == set(EXPECTED_GRAD_SYNC_SITES)

    cooldown, _statements, parents = next(
        block
        for block in blocks
        if _otel_constant(block[0].items[0].context_expr.keywords[0].value.values[0])
        == "GRAD_SYNC_SITE_INTERLEAVED_COOLDOWN"
    )
    ancestor = parents[cooldown]
    while not isinstance(ancestor, ast.For):
        ancestor = parents[ancestor]
    assert isinstance(ancestor.target, ast.Name) and ancestor.target.id == "model_chunk_id"

    ddp_start = _nested_function(DDP_PATH, "start_grad_sync")
    assert not any(
        _contains_otel_constant(node, "SPAN_GRAD_SYNC_START") for node in ddp_start.decorator_list
    )


@pytest.mark.parametrize("detail_enabled", [False, True])
@pytest.mark.parametrize("training_parent", [False, True])
def test_grad_sync_dispatch_preserves_callbacks_and_parenting(
    monkeypatch, telemetry_module, detail_enabled, training_parent
):
    telemetry = telemetry_module
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("mcore-spans")
    blocks = _grad_sync_blocks()
    events = []
    callback_arguments = []

    @contextmanager
    def managed(group, name, tracer=None, **attributes):
        events.append("span-enter")
        if group == telemetry.DETAIL and detail_enabled:
            with provider.get_tracer("mcore-spans").start_as_current_span(
                name, attributes=attributes
            ) as span:
                yield span
        else:
            yield None
        events.append("span-exit")

    monkeypatch.setattr(telemetry, "_managed_span", managed)

    def callback(parameters):
        events.append("callback")
        callback_arguments.append(parameters)

    class CallbackList:
        def __getitem__(self, index):
            return callback

    class Config:
        def __init__(self, indexed):
            self.indexed = indexed

        @property
        def grad_sync_func(self):
            events.append("callback-lookup")
            return CallbackList() if self.indexed else callback

    class Model:
        def __init__(self):
            self.parameters_result = iter((object(),))

        def parameters(self):
            events.append("parameters")
            return self.parameters_result

    @contextmanager
    def parent_context():
        if training_parent:
            with tracer.start_as_current_span(telemetry.SPAN_TRAINING_FORWARD_BACKWARD):
                yield
        else:
            yield

    try:
        with parent_context():
            for index, (_managed, statements, _parents) in enumerate(blocks):
                events.clear()
                model_chunk = Model()
                namespace = {
                    "_otel": telemetry,
                    "config": Config(indexed=index < 2),
                    "model": [model_chunk] if index < 2 else model_chunk,
                    "grad_sync_chunk_id": 0,
                    "model_chunk_id": 0,
                }
                exec(
                    compile(
                        ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                        SCHEDULES_PATH,
                        "exec",
                    ),
                    namespace,
                )
                assert events == [
                    "callback-lookup",
                    "parameters",
                    "span-enter",
                    "callback",
                    "span-exit",
                ]
                assert callback_arguments[-1] is model_chunk.parameters_result
        spans = exporter.get_finished_spans()
        dispatches = [span for span in spans if span.name == telemetry.SPAN_GRAD_SYNC_START]
        assert len(dispatches) == (3 if detail_enabled else 0)
        if detail_enabled:
            assert [span.attributes[telemetry.GRAD_SYNC_START_SITE] for span in dispatches] == list(
                EXPECTED_GRAD_SYNC_SITES.values()
            )
            if training_parent:
                parent = next(
                    span for span in spans if span.name == telemetry.SPAN_TRAINING_FORWARD_BACKWARD
                )
                assert all(span.parent.span_id == parent.context.span_id for span in dispatches)
            else:
                assert all(span.parent is None for span in dispatches)
    finally:
        provider.shutdown()
