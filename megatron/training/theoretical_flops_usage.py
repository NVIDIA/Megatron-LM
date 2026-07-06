# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Operator-level theoretical FLOPs reporting for dense Transformer training."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class OpFlopEntry:
    """One operator-level contribution to the global-batch theoretical FLOPs."""

    group: str
    submodule: str
    operator: str
    op_type: str
    precision: str
    shape: str
    count: int
    per_call_flops: int
    global_flops: int
    per_layer_tflops: float
    global_tflops: float
    sequence_sharded: bool


@dataclass
class TeAttentionRuntimeContext:
    """Runtime context that affects TE attention trace interpretation."""

    attention_backend_cli: str
    transformer_impl: str
    nvte_flash_attn: str
    nvte_fused_attn: str
    nvte_unfused_attn: str
    nvte_debug: str
    nvte_debug_level: str
    te_available_backends: str | None = None
    te_selected_backend: str | None = None
    te_fused_sub_backend: int | None = None
    capture_step: int | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None


@dataclass
class TheoreticalFlopsReport:
    """The complete theoretical FLOPs report written to JSON and stdout."""

    entries: list[OpFlopEntry]
    reference_total_flops: int
    computed_total_flops: int
    relative_error: float
    global_batch_size: int
    num_microbatches: int
    runtime_context: TeAttentionRuntimeContext


class TeAttentionLogCapture(logging.Handler):
    """Capture TE DotProductAttention backend-selection log records."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._records: list[str] = []
        logging.getLogger().addHandler(self)

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "DotProductAttention" in message:
            self._records.append(message)

    def maybe_capture(self, capture_step: int) -> dict[str, Any] | None:
        """Return parsed runtime-context updates once TE selection is visible."""

        parsed = parse_te_attention_runtime_context("\n".join(self._records))
        if parsed["te_selected_backend"] is None and parsed["te_available_backends"] is None:
            return None
        parsed["capture_step"] = capture_step
        return parsed

    def close(self) -> None:
        logging.getLogger().removeHandler(self)
        super().close()


def set_te_attention_debug_env_if_needed(args: Any) -> None:
    """Enable TE backend-selection debug logging before model construction."""

    if not getattr(args, "report_theoretical_flops", False):
        return
    if not getattr(args, "capture_te_attention_backend", True):
        return
    if getattr(args, "transformer_impl", None) != "transformer_engine":
        return
    os.environ.setdefault("NVTE_DEBUG", "1")
    os.environ.setdefault("NVTE_DEBUG_LEVEL", "2")


def parse_te_attention_runtime_context(log_text: str) -> dict[str, Any]:
    """Parse TE DotProductAttention backend-selection log text.

    Args:
        log_text: Text containing TE debug log lines.

    Returns:
        A partial runtime-context update with available and selected backends.
    """

    available = None
    selected = None
    fused_sub_backend = None
    for line in log_text.splitlines():
        if "DotProductAttention" not in line:
            continue
        available_match = re.search(r"Available backends\s*=\s*(.+)$", line)
        if available_match:
            available = available_match.group(1).strip()
        selected_match = re.search(r"Selected backend\s*=\s*([A-Za-z0-9_]+)", line)
        if selected_match:
            selected = selected_match.group(1).strip()
        fused_match = re.search(r"sub-backend\s*([0-9]+)", line)
        if fused_match:
            fused_sub_backend = int(fused_match.group(1))
    return {
        "te_available_backends": available,
        "te_selected_backend": selected,
        "te_fused_sub_backend": fused_sub_backend,
    }


def build_runtime_context(args: Any) -> TeAttentionRuntimeContext:
    """Build startup runtime context for the theoretical FLOPs JSON."""

    git_commit, git_dirty = _git_context()
    return TeAttentionRuntimeContext(
        attention_backend_cli=_stringify_arg_value(getattr(args, "attention_backend", "auto")),
        transformer_impl=str(getattr(args, "transformer_impl", "")),
        nvte_flash_attn=os.environ.get("NVTE_FLASH_ATTN", ""),
        nvte_fused_attn=os.environ.get("NVTE_FUSED_ATTN", ""),
        nvte_unfused_attn=os.environ.get("NVTE_UNFUSED_ATTN", ""),
        nvte_debug=os.environ.get("NVTE_DEBUG", ""),
        nvte_debug_level=os.environ.get("NVTE_DEBUG_LEVEL", ""),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def build_theoretical_flops_report(
    args: Any,
    num_microbatches: int | None = None,
    global_batch_size: int | None = None,
) -> TheoreticalFlopsReport:
    """Build a dense GQA/MHA operator-level theoretical FLOPs report.

    Args:
        args: Megatron training args.
        num_microbatches: Number of microbatches in one global batch.
        global_batch_size: Optional explicit global batch size override.

    Returns:
        A report whose integer entry sum exactly matches
        ``num_floating_point_operations`` for the same global batch.
    """

    _validate_dense_transformer_m1_args(args)
    if num_microbatches is None:
        if global_batch_size is None:
            num_microbatches = int(getattr(args, "num_microbatches", 1))
        else:
            dp_size = _get_data_parallel_size(args)
            num_microbatches = global_batch_size // (args.micro_batch_size * dp_size)
    if global_batch_size is None:
        dp_size = _get_data_parallel_size(args)
        global_batch_size = int(args.micro_batch_size * dp_size * num_microbatches)

    precision = _effective_gemm_precision(args)
    total_tokens = int(global_batch_size * args.seq_length)
    seqlen_squared_sum = int(global_batch_size * args.seq_length * args.seq_length)
    query_projection_size = int(args.kv_channels * args.num_attention_heads)
    key_projection_size = int(args.kv_channels * _num_query_groups(args))
    value_projection_size = key_projection_size
    gate_projection_size = query_projection_size if getattr(args, "attention_output_gate", False) else 0
    ffn_hidden_size = int(args.ffn_hidden_size)
    fc1_multiplier = 2 if getattr(args, "swiglu", False) else 1
    fbw = 3
    num_layers = int(args.num_layers)
    entries = [
        _gemm_entry(
            args,
            group="transformer_layer:dense:gqa",
            submodule="attention",
            operator="qkv_projection",
            precision=precision,
            n=query_projection_size + key_projection_size + value_projection_size + gate_projection_size,
            k=int(args.hidden_size),
            tp_split="n",
            sequence_sharded=False,
            per_call_flops=2
            * total_tokens
            * int(args.hidden_size)
            * (query_projection_size + key_projection_size + value_projection_size + gate_projection_size),
            count=fbw,
            layer_count=num_layers,
        ),
        _gemm_entry(
            args,
            group="transformer_layer:dense:gqa",
            submodule="attention",
            operator="output_projection",
            precision=precision,
            n=int(args.hidden_size),
            k=query_projection_size,
            tp_split="k",
            sequence_sharded=True,
            per_call_flops=2 * total_tokens * query_projection_size * int(args.hidden_size),
            count=fbw,
            layer_count=num_layers,
        ),
        _core_attention_entry(
            args,
            precision=_attention_precision(args),
            seqlen_squared_sum=seqlen_squared_sum,
            query_projection_size=query_projection_size,
            count=fbw,
            layer_count=num_layers,
        ),
        _gemm_entry(
            args,
            group="transformer_layer:dense:gqa",
            submodule="mlp",
            operator="fc1",
            precision=precision,
            n=fc1_multiplier * ffn_hidden_size,
            k=int(args.hidden_size),
            tp_split="n",
            sequence_sharded=False,
            per_call_flops=2 * total_tokens * int(args.hidden_size) * fc1_multiplier * ffn_hidden_size,
            count=fbw,
            layer_count=num_layers,
        ),
        _gemm_entry(
            args,
            group="transformer_layer:dense:gqa",
            submodule="mlp",
            operator="fc2",
            precision=precision,
            n=int(args.hidden_size),
            k=ffn_hidden_size,
            tp_split="k",
            sequence_sharded=True,
            per_call_flops=2 * total_tokens * int(args.hidden_size) * ffn_hidden_size,
            count=fbw,
            layer_count=num_layers,
        ),
        _gemm_entry(
            args,
            group="output_head",
            submodule="output_head",
            operator="logits",
            precision=precision,
            n=int(args.padded_vocab_size),
            k=int(args.hidden_size),
            tp_split="n",
            sequence_sharded=False,
            per_call_flops=2 * total_tokens * int(args.hidden_size) * int(args.padded_vocab_size),
            count=fbw,
            layer_count=1,
            per_layer_denominator=num_layers,
        ),
    ]

    computed_total = sum(entry.global_flops for entry in entries)
    from megatron.training.training import num_floating_point_operations

    reference_total_raw = num_floating_point_operations(args, global_batch_size)
    reference_total = int(reference_total_raw)
    if reference_total != reference_total_raw:
        raise ValueError(
            "num_floating_point_operations returned a non-integral value: "
            f"{reference_total_raw}"
        )
    if computed_total != reference_total:
        raise ValueError(
            "Theoretical FLOPs entries do not match reference total: "
            f"computed={computed_total}, reference={reference_total}"
        )
    return TheoreticalFlopsReport(
        entries=entries,
        reference_total_flops=reference_total,
        computed_total_flops=computed_total,
        relative_error=0.0,
        global_batch_size=global_batch_size,
        num_microbatches=int(num_microbatches),
        runtime_context=build_runtime_context(args),
    )


def format_theoretical_flops_report(report: TheoreticalFlopsReport, verbose: bool = False) -> str:
    """Format a human-readable theoretical FLOPs report."""

    lines = [
        "### THEORETICAL FLOPS REPORT START ###",
        f"  global_batch_size: {report.global_batch_size}",
        f"  num_microbatches: {report.num_microbatches}",
        f"  reference_total_tflops: {report.reference_total_flops / 1e12:.6f}",
        "  grouped layers / pseudo-layers:",
    ]
    for entry in report.entries:
        if not verbose and entry.submodule == "output_head":
            continue
        lines.extend(
            [
                f"    {entry.submodule} | {entry.operator} | {entry.op_type} | "
                f"precision={entry.precision}",
                f"      shape={entry.shape}",
                f"      global_tflops={entry.global_tflops:.6f} "
                f"per_layer_tflops={entry.per_layer_tflops:.6f}",
            ]
        )
    lines.extend(
        [
            "  integer validation:",
            f"    computed_total_flops: {report.computed_total_flops}",
            f"    reference_total_flops: {report.reference_total_flops}",
            "### THEORETICAL FLOPS REPORT END ###",
            "### TE ATTENTION BACKEND ###",
            f"  cli: --attention-backend {report.runtime_context.attention_backend_cli} "
            f"(--transformer-impl {report.runtime_context.transformer_impl})",
            "  allowed (NVTE_*): "
            f"FLASH={report.runtime_context.nvte_flash_attn} "
            f"FUSED={report.runtime_context.nvte_fused_attn} "
            f"UNFUSED={report.runtime_context.nvte_unfused_attn}",
            "  debug: "
            f"NVTE_DEBUG={report.runtime_context.nvte_debug} "
            f"NVTE_DEBUG_LEVEL={report.runtime_context.nvte_debug_level}",
            "  resolved (first TE forward, rank 0):",
            f"    available: {report.runtime_context.te_available_backends}",
            f"    selected:  {report.runtime_context.te_selected_backend}",
            "### TE ATTENTION BACKEND END ###",
        ]
    )
    return "\n".join(lines)


def write_theoretical_flops_json(
    report: TheoreticalFlopsReport,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Write ``theoretical_flops.json`` and return its path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "theoretical_flops.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_report_to_dict(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return json_path


def update_theoretical_flops_json_runtime_context(
    args: Any,
    updates: dict[str, Any],
) -> Path:
    """Update ``runtime_context`` in an existing theoretical FLOPs JSON file."""

    json_path = Path(getattr(args, "theoretical_flops_output_dir", "./flops_analysis")) / (
        "theoretical_flops.json"
    )
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runtime_context = payload.setdefault("runtime_context", {})
    runtime_context.update(updates)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return json_path


def report_theoretical_flops(
    args: Any,
    num_microbatches: int | None = None,
    verbose: bool = False,
) -> TheoreticalFlopsReport:
    """Build, print, and write the M1 theoretical FLOPs report."""

    report = build_theoretical_flops_report(args, num_microbatches=num_microbatches)
    print(format_theoretical_flops_report(report, verbose=verbose), flush=True)
    write_theoretical_flops_json(report, getattr(args, "theoretical_flops_output_dir", "./flops_analysis"))
    return report


def _report_to_dict(report: TheoreticalFlopsReport) -> dict[str, Any]:
    return {
        "entries": [asdict(entry) for entry in report.entries],
        "reference_total_flops": report.reference_total_flops,
        "computed_total_flops": report.computed_total_flops,
        "relative_error": report.relative_error,
        "global_batch_size": report.global_batch_size,
        "num_microbatches": report.num_microbatches,
        "runtime_context": asdict(report.runtime_context),
    }


def _gemm_entry(
    args: Any,
    *,
    group: str,
    submodule: str,
    operator: str,
    precision: str,
    n: int,
    k: int,
    tp_split: str,
    sequence_sharded: bool,
    per_call_flops: int,
    count: int,
    layer_count: int,
    per_layer_denominator: int | None = None,
) -> OpFlopEntry:
    tp_size = _get_tensor_model_parallel_size(args)
    local_m = _local_m(args, sequence_sharded)
    local_n = _split_dim(n, tp_size) if tp_split == "n" else n
    local_k = _split_dim(k, tp_size) if tp_split == "k" else k
    global_flops = int(per_call_flops * count * layer_count)
    denominator = per_layer_denominator if per_layer_denominator is not None else layer_count
    return OpFlopEntry(
        group=group,
        submodule=submodule,
        operator=operator,
        op_type="gemm",
        precision=precision,
        shape=f"(m,n,k)=({local_m}, {local_n}, {local_k})",
        count=count,
        per_call_flops=int(per_call_flops),
        global_flops=global_flops,
        per_layer_tflops=global_flops / denominator / 1e12,
        global_tflops=global_flops / 1e12,
        sequence_sharded=sequence_sharded,
    )


def _core_attention_entry(
    args: Any,
    *,
    precision: str,
    seqlen_squared_sum: int,
    query_projection_size: int,
    count: int,
    layer_count: int,
) -> OpFlopEntry:
    local_seq = _local_seq(args)
    num_heads = int(args.num_attention_heads)
    kv_channels = int(args.kv_channels)
    per_call_flops = 2 * seqlen_squared_sum * query_projection_size
    global_flops = int(per_call_flops * count * layer_count)
    return OpFlopEntry(
        group="transformer_layer:dense:gqa",
        submodule="attention",
        operator="(gqa)core_attn(sbhd)",
        op_type="core_attention",
        precision=precision,
        shape=(
            f"(mbs={int(args.micro_batch_size)}, seq={local_seq}, h={num_heads}, "
            f"qk_d={kv_channels}, v_d={kv_channels}) (causal)"
        ),
        count=count,
        per_call_flops=int(per_call_flops),
        global_flops=global_flops,
        per_layer_tflops=global_flops / layer_count / 1e12,
        global_tflops=global_flops / 1e12,
        sequence_sharded=False,
    )


def _validate_dense_transformer_m1_args(args: Any) -> None:
    unsupported = []
    if getattr(args, "multi_latent_attention", False):
        unsupported.append("multi_latent_attention")
    if getattr(args, "num_experts", None) is not None:
        unsupported.append("num_experts")
    if getattr(args, "mtp_num_layers", None) is not None:
        unsupported.append("mtp_num_layers")
    if getattr(args, "hybrid_layer_pattern", None) is not None:
        unsupported.append("hybrid_layer_pattern")
    if getattr(args, "experimental_attention_variant", None) is not None:
        unsupported.append("experimental_attention_variant")
    if unsupported:
        raise NotImplementedError(
            "M1 theoretical FLOPs reporting supports dense GQA/MHA Transformer only; "
            f"unsupported args set: {', '.join(unsupported)}"
        )
    _require_divisible(int(args.seq_length), _get_context_parallel_size(args), "seq_length", "context_parallel_size")
    tp_size = _get_tensor_model_parallel_size(args)
    query_projection_size = int(args.kv_channels * args.num_attention_heads)
    key_projection_size = int(args.kv_channels * _num_query_groups(args))
    gate_projection_size = query_projection_size if getattr(args, "attention_output_gate", False) else 0
    for name, dim in (
        ("qkv_projection output", query_projection_size + 2 * key_projection_size + gate_projection_size),
        ("query projection", query_projection_size),
        ("fc1 output", int(args.ffn_hidden_size) * (2 if getattr(args, "swiglu", False) else 1)),
        ("fc2 input", int(args.ffn_hidden_size)),
        ("padded_vocab_size", int(args.padded_vocab_size)),
    ):
        _require_divisible(dim, tp_size, name, "tensor_model_parallel_size")


def _effective_gemm_precision(args: Any) -> str:
    fp8_format = getattr(args, "fp8_format", None)
    if fp8_format not in (None, "") and _fp8_supported():
        return "fp8"
    if getattr(args, "bf16", False):
        return "bf16"
    if getattr(args, "fp16", False):
        return "fp16"
    return "fp32"


def _attention_precision(args: Any) -> str:
    if getattr(args, "bf16", False):
        return "bf16"
    if getattr(args, "fp16", False):
        return "fp16"
    return "fp32"


def _fp8_supported() -> bool:
    try:
        import transformer_engine.pytorch.fp8 as te_fp8

        supported, _ = te_fp8.check_fp8_support()
        return bool(supported)
    except (ImportError, RuntimeError, AttributeError):
        return False


def _local_m(args: Any, sequence_sharded: bool) -> int:
    m = int(args.micro_batch_size) * _local_seq(args)
    if sequence_sharded and getattr(args, "sequence_parallel", False):
        m = _split_dim(m, _get_tensor_model_parallel_size(args))
    return m


def _local_seq(args: Any) -> int:
    return _split_dim(int(args.seq_length), _get_context_parallel_size(args))


def _split_dim(value: int, parts: int) -> int:
    _require_divisible(value, parts, "dimension", "parallel size")
    return value // parts


def _require_divisible(value: int, divisor: int, value_name: str, divisor_name: str) -> None:
    if divisor <= 0 or value % divisor != 0:
        raise ValueError(f"{value_name}={value} must be divisible by {divisor_name}={divisor}")


def _num_query_groups(args: Any) -> int:
    if getattr(args, "group_query_attention", False):
        return int(args.num_query_groups)
    return int(args.num_attention_heads)


def _stringify_arg_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _get_tensor_model_parallel_size(args: Any) -> int:
    return int(getattr(args, "tensor_model_parallel_size", 1))


def _get_context_parallel_size(args: Any) -> int:
    return int(getattr(args, "context_parallel_size", 1))


def _get_data_parallel_size(args: Any) -> int:
    if hasattr(args, "data_parallel_size"):
        return int(args.data_parallel_size)
    world_size = int(getattr(args, "world_size", 1))
    denominator = (
        _get_tensor_model_parallel_size(args)
        * int(getattr(args, "pipeline_model_parallel_size", 1))
        * _get_context_parallel_size(args)
    )
    return world_size // denominator


def _git_context() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) != 0
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None
