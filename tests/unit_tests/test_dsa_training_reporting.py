# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Execute the train loop's reporting block without constructing a model."""

import ast
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _reporting_code():
    source = Path(__file__).resolve().parents[2] / "megatron/training/training.py"
    tree = ast.parse(source.read_text())
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        )

    loop = next(
        node
        for node in ast.walk(train)
        if isinstance(node, ast.While) and any(assigns(stmt, "_report_span") for stmt in node.body)
    )
    start = next(i for i, node in enumerate(loop.body) if assigns(node, "_report_span"))
    end = next(i for i in range(start, len(loop.body)) if assigns(loop.body[i], "is_first_iteration"))
    log = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training_log")
    extra_kwargs = {arg.arg for arg in log.args.args} & {
        "model", "callback_manager", "packed_sequence_stats"
    }
    code = compile(ast.Module(body=loop.body[start : end + 1], type_ignores=[]), str(source), "exec")
    return code, extra_kwargs


@pytest.mark.parametrize("telemetry_enabled", [False, True])
@pytest.mark.parametrize("cached_norms", ["available", "empty", "legacy_optimizer"])
def test_reports_once_with_preclip_norms_and_original_context(monkeypatch, telemetry_enabled, cached_norms):
    """One update must contribute once to metrics and use the saved pre-clip norms."""
    calls = []
    reductions = []
    scopes = []
    telemetry = []
    norm_fallbacks = []
    parameter_norms = []
    zero_counts = []
    pg_collection = object()
    model_pg_collection = object()
    model = object()
    optimizer = SimpleNamespace(is_stub_optimizer=False, param_groups=[{}])
    optimizer.get_loss_scale = lambda: SimpleNamespace(item=lambda: 1.0)
    if cached_norms != "legacy_optimizer":
        optimizer.get_last_dsa_split_grad_norms = lambda: (
            (7.0, 11.0) if cached_norms == "available" else None
        )

    def fallback(*args):
        norm_fallbacks.append(args)
        return 0.7, 1.1

    def parameter_norm(*args, **kwargs):
        parameter_norms.append((args, kwargs))
        return 13.0

    def zeros(*args):
        zero_counts.append(args)
        return 2, 3

    def reduce_stat(value):
        reductions.append(value)
        return value

    def report(*args, **kwargs):
        calls.append((args, kwargs))
        # Model the stateful accumulation performed on every training_log call.
        args[1]["updates"] = args[1].get("updates", 0) + 1
        args[1]["loss"] = args[1].get("loss", 0) + args[0]["lm loss"]
        return False

    @contextmanager
    def managed_span(group, name, **kwargs):
        scopes.append((group, name, kwargs))
        yield

    span = SimpleNamespace(end=lambda: telemetry.append("end"))
    tracer = SimpleNamespace(start_span=lambda name: span)
    context = SimpleNamespace(
        attach=lambda context: "token", detach=lambda token: telemetry.append(("detach", token))
    )
    trace = SimpleNamespace(set_span_in_context=lambda span: "context")
    monkeypatch.setitem(sys.modules, "opentelemetry", SimpleNamespace(context=context, trace=trace))
    scope = {
        "args": SimpleNamespace(tensorboard_log_interval=1, log_params_norm=True),
        "_otel_sg_enabled": lambda group: telemetry_enabled,
        "get_telemetry": lambda: SimpleNamespace(tracer=tracer),
        "_otel_mark_goodput": lambda span: telemetry.append("goodput"),
        "_otel_managed_span": managed_span,
        "optimizer": optimizer,
        "model": model,
        "callback_manager": object(),
        "packed_sequence_stats": object(),
        "pg_collection": pg_collection,
        "model_pg_collection": model_pg_collection,
        "_should_compute_params_norm": lambda *args: True,
        "calc_params_l2_norm": parameter_norm,
        "calc_dsa_split_grad_norms": fallback,
        "calc_dsa_split_grad_num_zeros": zeros,
        "reduce_max_stat_across_model_parallel_group": reduce_stat,
        "get_canonical_lr_for_logging": lambda groups: 0.01,
        "get_indexer_lr_for_logging": lambda groups: 0.02,
        "get_tensorboard_writer": lambda: None,
        "get_wandb_writer": lambda: None,
        "training_log": report,
        "loss_dict": {"lm loss": 2.0},
        "total_loss_dict": {},
        "report_memory_flag": True,
        "skipped_iter": 0,
        "grad_norm": 17.0,
        "num_zeros_in_grad": 5,
        "max_attention_logit": None,
        "is_first_iteration": True,
        "seqlen_squared_sum_in_batch": 1234,
        "total_real_tokens_in_batch": 100,
    }
    code, extra_kwargs = _reporting_code()
    for iteration in (1, 2, 3):
        scope["iteration"] = iteration
        exec(code, scope)

    assert len(calls) == 3
    assert scope["total_loss_dict"] == {"updates": 3, "loss": 6.0}
    assert len(parameter_norms) == len(zero_counts) == 3
    assert all(kwargs == {"pg_collection": pg_collection} for _, kwargs in parameter_norms)
    expected_norms = (11.0, 7.0) if cached_norms == "available" else (1.1, 0.7)
    assert len(norm_fallbacks) == (0 if cached_norms == "available" else 3)
    for iteration, (args, kwargs) in enumerate(calls, 1):
        assert args[4] == iteration
        assert args[9:11] == expected_norms
        assert kwargs == {
            "pg_collection": model_pg_collection,
            "is_first_iteration": iteration == 1,
            "seqlen_squared_sum_in_batch": 1234,
            "total_real_tokens_in_batch": 100,
            **{name: scope[name] for name in extra_kwargs},
        }
    assert [name for _, name, _ in scopes].count("megatron.train.log") == 3
    assert [name for _, name, _ in scopes].count("megatron.train.params_norm") == 3
    assert telemetry.count("end") == (3 if telemetry_enabled else 0)
    assert telemetry.count(("detach", "token")) == (3 if telemetry_enabled else 0)
    assert not scope["is_first_iteration"]
