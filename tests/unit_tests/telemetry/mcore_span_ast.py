# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AST helpers shared by the MCore call-site span tests."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

TELEMETRY_PATH = REPO_ROOT / "megatron/core/telemetry/telemetry.py"
SCHEDULES_PATH = REPO_ROOT / "megatron/core/pipeline_parallel/schedules.py"
P2P_PATH = REPO_ROOT / "megatron/core/pipeline_parallel/p2p_communication.py"
ACTIVATION_OFFLOAD_PATH = (
    REPO_ROOT / "megatron/core/pipeline_parallel/fine_grained_activation_offload.py"
)
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


def tree(path):
    """Parse a source file."""
    return ast.parse(path.read_text(), filename=str(path))


def nested_function(path, name):
    """Return the first function definition named ``name`` in ``path``."""
    return next(
        node
        for node in ast.walk(tree(path))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def module_string_constants(path):
    """Return the module-level string constants assigned in ``path``."""
    constants = {}
    for node in tree(path).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def otel_constant(node):
    """Return ``NAME`` for an ``_otel.NAME`` attribute node, else ``None``."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_otel"
    ):
        return node.attr
    return None


def call_name(node):
    """Return a dotted name for simple ``name`` or ``module.name`` call targets."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def span_calls(path):
    """Return ``(span_constant, group_constant)`` pairs for shim span calls in ``path``."""
    calls = []
    for node in ast.walk(tree(path)):
        if not isinstance(node, ast.Call):
            continue
        primitive = call_name(node.func)
        if primitive in {"_otel.managed_span", "_otel.trace_fn"}:
            group = otel_constant(node.args[0])
            span_name = otel_constant(node.args[1])
        elif primitive == "_otel.span_cm":
            group = None
            span_name = otel_constant(node.args[0])
        else:
            continue
        if span_name is not None:
            calls.append((span_name, group))
    return calls


def managed_span_constant(call):
    """Return the span constant of an ``_otel.managed_span`` call, else ``None``."""
    if call_name(call.func) == "_otel.managed_span":
        return otel_constant(call.args[1])
    return None


def contains_otel_constant(node, name):
    """Return whether ``_otel.<name>`` appears anywhere beneath ``node``."""
    return any(otel_constant(candidate) == name for candidate in ast.walk(node))


def assert_detail_span_counts(paths, expected_counts):
    """Check shim span usage and literal-name absence for a set of call-site files."""
    calls = [call for path in paths for call in span_calls(path)]
    for span_name, count in expected_counts.items():
        assert calls.count((span_name, "DETAIL")) == count, span_name

    source = "".join(path.read_text() for path in paths)
    for name in expected_counts:
        assert f"_otel.{name}" in source
    for value in EXPECTED_MCORE_SPANS.values():
        assert value not in source


def assert_layer_forward_number(path):
    """Check that the layer-forward span carries only the canonical layer number."""
    calls = [
        node
        for node in ast.walk(tree(path))
        if isinstance(node, ast.Call) and managed_span_constant(node) == "SPAN_LAYER_FORWARD"
    ]
    assert len(calls) == 1
    attributes = next(keyword.value for keyword in calls[0].keywords if keyword.arg is None)
    assert isinstance(attributes, ast.Dict)
    assert len(attributes.keys) == 1
    assert otel_constant(attributes.keys[0]) == "LAYER_NUMBER"
    value = attributes.values[0]
    assert isinstance(value, ast.Attribute)
    assert isinstance(value.value, ast.Name) and value.value.id == "self"
    assert value.attr == "layer_number"


def assert_uses_shim_not_legacy_fallbacks(path):
    """Check that a migrated module imports the shim instead of Lens helpers."""
    source = path.read_text()
    assert "telemetry as _otel" in source
    assert "nemo.lens" not in source
    assert "megatron.core.telemetry.fallbacks" not in source
