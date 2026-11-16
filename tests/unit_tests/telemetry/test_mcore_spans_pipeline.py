# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical pipeline-parallel spans and gradient-sync callback transport."""

import ast
from contextlib import contextmanager

import pytest

from tests.unit_tests.telemetry.mcore_span_ast import (
    ACTIVATION_OFFLOAD_PATH,
    EXPECTED_GRAD_SYNC_SITES,
    P2P_PATH,
    SCHEDULES_PATH,
    assert_detail_span_counts,
    assert_uses_shim_not_legacy_fallbacks,
    call_name,
    nested_function,
    otel_constant,
    tree,
)

# Any active span stands in for the trainer's forward/backward parent here.
PARENT_SPAN = "test.grad_sync.parent"


def _grad_sync_blocks():
    schedules = tree(SCHEDULES_PATH)
    parents = {
        child: parent for parent in ast.walk(schedules) for child in ast.iter_child_nodes(parent)
    }
    blocks = []
    for node in ast.walk(schedules):
        if not isinstance(node, ast.With):
            continue
        context = node.items[0].context_expr
        if not isinstance(context, ast.Call) or call_name(context.func) != "_otel.managed_span":
            continue
        if otel_constant(context.args[1]) != "SPAN_GRAD_SYNC_START":
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


def test_pipeline_call_sites_use_constants_and_detail_group():
    assert_detail_span_counts(
        (SCHEDULES_PATH, P2P_PATH),
        {
            "SPAN_MICROBATCH_FORWARD": 1,
            "SPAN_MICROBATCH_BACKWARD": 1,
            "SPAN_P2P_RECV_FORWARD": 1,
            "SPAN_P2P_RECV_BACKWARD": 1,
            "SPAN_P2P_SEND_FORWARD": 1,
            "SPAN_P2P_SEND_BACKWARD": 1,
            "SPAN_GRAD_SYNC_START": 3,
        },
    )
    assert_uses_shim_not_legacy_fallbacks(SCHEDULES_PATH)
    assert_uses_shim_not_legacy_fallbacks(P2P_PATH)


def test_activation_offload_instrumentation_is_removed():
    source = ACTIVATION_OFFLOAD_PATH.read_text()
    assert "telemetry as _otel" not in source
    assert "nemo.lens" not in source
    assert "megatron.core.telemetry.fallbacks" not in source
    for name in ("offload", "reload"):
        assert not nested_function(ACTIVATION_OFFLOAD_PATH, name).decorator_list


def test_schedule_owns_each_grad_sync_managed_block():
    blocks = _grad_sync_blocks()
    assert len(blocks) == 3
    sites = []
    for managed, statements, _parents in blocks:
        context = managed.items[0].context_expr
        assert otel_constant(context.args[0]) == "DETAIL"
        assert otel_constant(context.args[1]) == "SPAN_GRAD_SYNC_START"
        attributes = next(keyword.value for keyword in context.keywords if keyword.arg is None)
        assert otel_constant(attributes.keys[0]) == "GRAD_SYNC_START_SITE"
        sites.append(otel_constant(attributes.values[0]))
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
        if otel_constant(block[0].items[0].context_expr.keywords[0].value.values[0])
        == "GRAD_SYNC_SITE_INTERLEAVED_COOLDOWN"
    )
    ancestor = parents[cooldown]
    while not isinstance(ancestor, ast.For):
        ancestor = parents[ancestor]
    assert isinstance(ancestor.target, ast.Name) and ancestor.target.id == "model_chunk_id"


@pytest.mark.parametrize("detail_enabled", [False, True])
@pytest.mark.parametrize("training_parent", [False, True])
def test_grad_sync_dispatch_preserves_callbacks_and_parenting(
    monkeypatch, telemetry_module, detail_enabled, training_parent
):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
    in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    telemetry = telemetry_module
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
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
            with tracer.start_as_current_span(PARENT_SPAN):
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
                parent = next(span for span in spans if span.name == PARENT_SPAN)
                assert all(span.parent.span_id == parent.context.span_id for span in dispatches)
            else:
                assert all(span.parent is None for span in dispatches)
    finally:
        provider.shutdown()
