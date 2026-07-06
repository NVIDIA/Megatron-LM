# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reconcile PyTorch profiler Chrome traces against theoretical FLOPs entries."""

from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TraceGemmEvent:
    """A GEMM-like profiler event with an inferred local ``(m, n, k)`` shape."""

    name: str
    shape: tuple[int, int, int]
    count: int
    estimated_flops: int


@dataclass
class TraceReconciliationResult:
    """Trace/theory reconciliation result for one profiled rank."""

    rank: int
    profile_window: dict[str, int | None]
    trace_path: str
    theoretical_path: str
    te_attention_selected: str | None
    te_fused_sub_backend: int | None
    analytical_gemm_operators: int
    trace_gemm_events: int
    trace_gemm_unique_shapes: int
    matched: int
    unmatched_analytical: list[dict[str, Any]]
    unmatched_trace: list[dict[str, Any]]
    flops_budget: dict[str, float | int]
    warnings: list[str]


def get_torch_profile_dir(args: Any) -> Path:
    """Return the PyTorch profiler Chrome trace directory for the current args."""

    if getattr(args, "report_theoretical_flops", False):
        return Path(getattr(args, "theoretical_flops_output_dir", "./flops_analysis")) / "torch_profile"
    return Path(f"{args.tensorboard_dir}/../torch_profile")


def get_torch_profile_trace_path(args: Any, rank: int) -> Path:
    """Return the Chrome trace path for ``rank``."""

    return get_torch_profile_dir(args) / f"rank-{rank}.json.gz"


def reconcile_trace_vs_theory(
    args: Any,
    *,
    rank: int,
    trace_path: str | os.PathLike[str] | None = None,
    theoretical_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    shape_tolerance: float = 0.01,
) -> TraceReconciliationResult:
    """Compare GEMM shapes in one Chrome trace against ``theoretical_flops.json``."""

    output_dir = Path(getattr(args, "theoretical_flops_output_dir", "./flops_analysis"))
    trace = Path(trace_path) if trace_path is not None else get_torch_profile_trace_path(args, rank)
    theory = Path(theoretical_path) if theoretical_path is not None else output_dir / "theoretical_flops.json"
    output = Path(output_path) if output_path is not None else output_dir / f"reconciliation_rank{rank}.json"

    with theory.open("r", encoding="utf-8") as f:
        theoretical_payload = json.load(f)

    analytical_entries = [
        entry
        for entry in theoretical_payload.get("entries", [])
        if entry.get("op_type") in {"gemm", "grouped_gemm"} and _parse_mnk_shape(entry.get("shape", "")) is not None
    ]
    trace_events = parse_chrome_trace_gemm_events(trace)
    unique_trace_events = _unique_trace_events(trace_events)

    unmatched_analytical = []
    matched = 0
    for entry in analytical_entries:
        analytical_shape = _parse_mnk_shape(entry["shape"])
        assert analytical_shape is not None
        if any(_shapes_match(analytical_shape, event.shape, shape_tolerance) for event in unique_trace_events):
            matched += 1
            continue
        unmatched_analytical.append(_unmatched_analytical_payload(entry, analytical_shape))

    unmatched_trace = []
    for event in unique_trace_events:
        if any(
            _shapes_match(_parse_mnk_shape(entry["shape"]), event.shape, shape_tolerance)
            for entry in analytical_entries
        ):
            continue
        unmatched_trace.append(
            {
                "name": event.name,
                "shape": list(event.shape),
                "count": event.count,
                "estimated_flops": event.estimated_flops,
            }
        )

    analytical_gemm_flops = int(sum(entry.get("global_flops", 0) for entry in analytical_entries))
    trace_estimated_flops = int(sum(event.estimated_flops for event in trace_events))
    warnings = _build_warnings(theoretical_payload, unmatched_analytical)
    runtime_context = theoretical_payload.get("runtime_context", {})
    result = TraceReconciliationResult(
        rank=rank,
        profile_window={
            "start": getattr(args, "profile_step_start", None),
            "end": getattr(args, "profile_step_end", None),
        },
        trace_path=str(trace),
        theoretical_path=str(theory),
        te_attention_selected=runtime_context.get("te_selected_backend"),
        te_fused_sub_backend=runtime_context.get("te_fused_sub_backend"),
        analytical_gemm_operators=len(analytical_entries),
        trace_gemm_events=len(trace_events),
        trace_gemm_unique_shapes=len(unique_trace_events),
        matched=matched,
        unmatched_analytical=unmatched_analytical,
        unmatched_trace=unmatched_trace,
        flops_budget={
            "analytical_gemm_flops": analytical_gemm_flops,
            "analytical_gemm_tflops": analytical_gemm_flops / 1e12,
            "trace_estimated_flops": trace_estimated_flops,
            "trace_estimated_tflops": trace_estimated_flops / 1e12,
            "analytical_total_flops": int(theoretical_payload.get("reference_total_flops", 0)),
            "analytical_total_tflops": int(theoretical_payload.get("reference_total_flops", 0)) / 1e12,
        },
        warnings=warnings,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, sort_keys=True)
        f.write("\n")
    return result


def parse_chrome_trace_gemm_events(trace_path: str | os.PathLike[str]) -> list[TraceGemmEvent]:
    """Parse GEMM-like events from a PyTorch profiler Chrome trace."""

    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    events = []
    for event in payload.get("traceEvents", []):
        name = str(event.get("name", ""))
        if not _is_gemm_event_name(name):
            continue
        shape = _extract_event_mnk_shape(event)
        if shape is None:
            continue
        events.append(
            TraceGemmEvent(
                name=name,
                shape=shape,
                count=1,
                estimated_flops=2 * shape[0] * shape[1] * shape[2],
            )
        )
    return events


def format_reconciliation_summary(result: TraceReconciliationResult) -> str:
    """Format a human-readable reconciliation summary."""

    lines = [
        "### TRACE RECONCILIATION START ###",
        f"  profile window: iterations [{result.profile_window['start']}, {result.profile_window['end']}), "
        f"rank {result.rank}",
        f"  trace source: {result.trace_path}",
        f"  te_attention_selected: {_format_te_backend(result)}",
        f"  analytical gemm operators: {result.analytical_gemm_operators}",
        f"  trace gemm events (unique shapes): {result.trace_gemm_unique_shapes}",
        "  matched: "
        f"{result.matched}, unmatched_analytical: {len(result.unmatched_analytical)}, "
        f"unmatched_trace: {len(result.unmatched_trace)}",
    ]
    if result.unmatched_analytical:
        lines.append("")
        lines.append("  unmatched_analytical:")
        for entry in result.unmatched_analytical[:10]:
            lines.append(
                f"    {entry['submodule']} | {entry['operator']} | {tuple(entry['shape'])} - "
                "no trace event with shape within 1% tol"
            )
            if entry.get("hint"):
                lines.append(f"      hint: {entry['hint']}")
    if result.warnings:
        lines.append("")
        lines.append("  warnings:")
        for warning in result.warnings:
            lines.append(f"    {warning}")
    budget = result.flops_budget
    lines.extend(
        [
            "",
            "  flops budget:",
            f"    analytical_gemm_tflops:  {budget['analytical_gemm_tflops']:.6f}",
            f"    trace_estimated_tflops:  {budget['trace_estimated_tflops']:.6f}",
            f"    analytical_total_tflops: {budget['analytical_total_tflops']:.6f}",
            "### TRACE RECONCILIATION END ###",
        ]
    )
    return "\n".join(lines)


def _unique_trace_events(events: list[TraceGemmEvent]) -> list[TraceGemmEvent]:
    counts = Counter((event.name, event.shape) for event in events)
    return [
        TraceGemmEvent(
            name=name,
            shape=shape,
            count=count,
            estimated_flops=count * 2 * shape[0] * shape[1] * shape[2],
        )
        for (name, shape), count in counts.items()
    ]


def _is_gemm_event_name(name: str) -> bool:
    lower_name = name.lower()
    return (
        "aten::mm" in lower_name
        or "aten::addmm" in lower_name
        or "gemm" in lower_name
        or "groupedgemm" in lower_name
    )


def _extract_event_mnk_shape(event: dict[str, Any]) -> tuple[int, int, int] | None:
    args = event.get("args", {})
    direct_shape = _shape_from_direct_mnk(args)
    if direct_shape is not None:
        return direct_shape

    for key in ("Input Dims", "Input Shapes", "input_dims", "input_shapes"):
        if key not in args:
            continue
        shape = _shape_from_input_dims(args[key])
        if shape is not None:
            return shape
    return None


def _shape_from_direct_mnk(args: dict[str, Any]) -> tuple[int, int, int] | None:
    lower_args = {str(key).lower(): value for key, value in args.items()}
    if {"m", "n", "k"}.issubset(lower_args):
        try:
            return int(lower_args["m"]), int(lower_args["n"]), int(lower_args["k"])
        except (TypeError, ValueError):
            return None
    for key in ("mnk", "shape", "gemm_shape"):
        value = lower_args.get(key)
        if isinstance(value, str):
            parsed = _parse_mnk_shape(value)
            if parsed is not None:
                return parsed
    return None


def _shape_from_input_dims(input_dims: Any) -> tuple[int, int, int] | None:
    matrix_shapes = []
    for dims in input_dims if isinstance(input_dims, list) else []:
        if not isinstance(dims, list) or len(dims) != 2:
            continue
        try:
            matrix_shapes.append((int(dims[0]), int(dims[1])))
        except (TypeError, ValueError):
            continue
    for left_index, left in enumerate(matrix_shapes):
        for right_index, right in enumerate(matrix_shapes):
            if left_index == right_index:
                continue
            if left[1] == right[0]:
                return left[0], right[1], left[1]
    return None


def _parse_mnk_shape(shape_text: str | None) -> tuple[int, int, int] | None:
    if not shape_text:
        return None
    match = re.search(r"\((?:m,\s*n,\s*k)?\)=?\((\d+),\s*(\d+),\s*(\d+)\)", shape_text)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match = re.search(r"m\s*=\s*(\d+).*n\s*=\s*(\d+).*k\s*=\s*(\d+)", shape_text)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def _shapes_match(
    analytical_shape: tuple[int, int, int] | None,
    trace_shape: tuple[int, int, int],
    tolerance: float,
) -> bool:
    if analytical_shape is None:
        return False
    return all(
        abs(expected - actual) <= max(1, int(expected * tolerance))
        for expected, actual in zip(analytical_shape, trace_shape)
    )


def _unmatched_analytical_payload(
    entry: dict[str, Any],
    analytical_shape: tuple[int, int, int],
) -> dict[str, Any]:
    hint = None
    if entry.get("submodule") == "mlp" and entry.get("operator") in {"fc1", "fc2"}:
        hint = "may be fused via TE gemm+activation kernel"
    return {
        "group": entry.get("group"),
        "submodule": entry.get("submodule"),
        "operator": entry.get("operator"),
        "precision": entry.get("precision"),
        "shape": list(analytical_shape),
        "global_flops": entry.get("global_flops"),
        "hint": hint,
    }


def _build_warnings(
    theoretical_payload: dict[str, Any],
    unmatched_analytical: list[dict[str, Any]],
) -> list[str]:
    runtime_context = theoretical_payload.get("runtime_context", {})
    selected_backend = runtime_context.get("te_selected_backend")
    warnings = []
    if selected_backend in {"FusedAttention", "UnfusedDotProductAttention"}:
        warnings.append(
            "TE attention backend may change how core attention appears in the trace; "
            "GEMM shape matching excludes core_attention entries."
        )
    if any(entry.get("hint") for entry in unmatched_analytical):
        warnings.append("Some unmatched analytical GEMMs may be expected when Transformer Engine fuses kernels.")
    return warnings


def _format_te_backend(result: TraceReconciliationResult) -> str:
    if result.te_attention_selected is None:
        return "unknown"
    if result.te_fused_sub_backend is None:
        return result.te_attention_selected
    return f"{result.te_attention_selected} (sub-backend {result.te_fused_sub_backend})"
