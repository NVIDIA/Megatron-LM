# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Benchmark result dataclasses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StepTrace:
    step: int
    loss: float
    grad_norm: float
    step_ms: float
    peak_mem_gb: float | None = None
    tflops_per_gpu: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "step": self.step,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "step_ms": self.step_ms,
        }
        if self.peak_mem_gb is not None:
            result["peak_mem_gb"] = self.peak_mem_gb
        if self.tflops_per_gpu is not None:
            result["tflops_per_gpu"] = self.tflops_per_gpu
        return result


@dataclass(slots=True)
class RunResult:
    backend: str
    model_name: str
    impl: str
    optimizer_backend: str
    tp: int
    etp: int | None
    ep: int
    pp: int
    vpp: int
    cp: int
    seq_len: int
    num_microbatches: int
    step_traces: list[StepTrace] = field(default_factory=list)
    avg_step_ms: float = 0.0
    peak_mem_gb: float = 0.0
    tok_per_s: float = 0.0
    tok_per_s_per_gpu: float = 0.0
    tflops_per_gpu: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "impl": self.impl,
            "optimizer_backend": self.optimizer_backend,
            "avg_step_ms": self.avg_step_ms,
            "tok_per_s": self.tok_per_s,
            "tok_per_s_per_gpu": self.tok_per_s_per_gpu,
            "peak_mem_gb": self.peak_mem_gb,
            "tflops_per_gpu": self.tflops_per_gpu,
            "steps_measured": len(self.step_traces),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_dict(),
            "result": {
                "backend": self.backend,
                "model_name": self.model_name,
                "impl": self.impl,
                "optimizer_backend": self.optimizer_backend,
                "tp": self.tp,
                "etp": self.etp,
                "ep": self.ep,
                "pp": self.pp,
                "vpp": self.vpp,
                "cp": self.cp,
                "seq_len": self.seq_len,
                "num_microbatches": self.num_microbatches,
                "step_traces": [trace.to_dict() for trace in self.step_traces],
                "avg_step_ms": self.avg_step_ms,
                "peak_mem_gb": self.peak_mem_gb,
                "tok_per_s": self.tok_per_s,
                "tok_per_s_per_gpu": self.tok_per_s_per_gpu,
                "tflops_per_gpu": self.tflops_per_gpu,
                "metadata": dict(self.metadata),
            },
        }


def load_result_artifact(path: str | Path) -> dict[str, Any]:
    """Load a benchmark JSON artifact from ``bench.py --output-json``."""
    with Path(path).open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark artifact must be a JSON object: {path}")
    return value


def result_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the summary block from a benchmark artifact."""
    summary = artifact.get("summary")
    if isinstance(summary, dict):
        return dict(summary)
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise ValueError("Benchmark artifact must contain `summary` or `result`.")
    return {
        "backend": result.get("backend"),
        "model_name": result.get("model_name"),
        "impl": result.get("impl"),
        "optimizer_backend": result.get("optimizer_backend"),
        "avg_step_ms": result.get("avg_step_ms"),
        "tok_per_s": result.get("tok_per_s"),
        "tok_per_s_per_gpu": result.get("tok_per_s_per_gpu"),
        "peak_mem_gb": result.get("peak_mem_gb"),
        "tflops_per_gpu": result.get("tflops_per_gpu"),
        "steps_measured": len(result.get("step_traces", [])),
    }


def compare_step_traces(
    baseline: dict[str, Any], candidate: dict[str, Any], *, atol: float = 1e-4, rtol: float = 1e-4
) -> dict[str, Any]:
    """Compare loss and grad-norm traces from two benchmark artifacts."""
    base_steps = baseline.get("result", {}).get("step_traces", [])
    cand_steps = candidate.get("result", {}).get("step_traces", [])
    sample_count = min(len(base_steps), len(cand_steps))
    max_loss_abs = 0.0
    max_grad_norm_abs = 0.0
    for idx in range(sample_count):
        base = base_steps[idx]
        cand = cand_steps[idx]
        max_loss_abs = max(max_loss_abs, abs(float(base["loss"]) - float(cand["loss"])))
        max_grad_norm_abs = max(
            max_grad_norm_abs, abs(float(base["grad_norm"]) - float(cand["grad_norm"]))
        )

    lengths_match = sample_count == len(base_steps) == len(cand_steps)
    loss_ref_max = max([abs(float(step["loss"])) for step in base_steps[:sample_count]] + [0.0])
    grad_norm_ref_max = max(
        [abs(float(step["grad_norm"])) for step in base_steps[:sample_count]] + [0.0]
    )
    loss_passed = lengths_match and max_loss_abs <= atol + rtol * loss_ref_max
    grad_norm_passed = lengths_match and max_grad_norm_abs <= atol + rtol * grad_norm_ref_max

    return {
        "samples": sample_count,
        "atol": atol,
        "rtol": rtol,
        "passed": loss_passed and grad_norm_passed,
        "loss_passed": loss_passed,
        "grad_norm_passed": grad_norm_passed,
        "max_loss_abs": max_loss_abs,
        "max_grad_norm_abs": max_grad_norm_abs,
    }


def compare_correctness_artifacts(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Strict bitwise comparison for deterministic correctness artifacts."""
    base_steps = baseline.get("steps", [])
    cand_steps = candidate.get("steps", [])
    sample_count = min(len(base_steps), len(cand_steps))
    lengths_match = sample_count == len(base_steps) == len(cand_steps)

    max_loss_abs = 0.0
    max_grad_norm_abs = 0.0
    mismatches: list[dict[str, Any]] = []

    def _tensor_fingerprint_matches(base: Any, cand: Any) -> bool:
        if base == cand:
            return True
        if not isinstance(base, dict) or not isinstance(cand, dict):
            return False
        if base.get("shape") != cand.get("shape"):
            return False
        base_bf16 = base.get("sha256_as_bf16")
        cand_bf16 = cand.get("sha256_as_bf16")
        return bool(base_bf16 and base_bf16 == cand_bf16)

    base_eval = baseline.get("eval_logits")
    cand_eval = candidate.get("eval_logits")
    if base_eval is not None or cand_eval is not None:
        if not _tensor_fingerprint_matches(base_eval, cand_eval):
            mismatches.append({"field": "eval_logits"})

    for idx in range(sample_count):
        base = base_steps[idx]
        cand = cand_steps[idx]
        loss_abs = abs(float(base["loss"]["value"]) - float(cand["loss"]["value"]))
        grad_abs = abs(float(base["grad_norm"]["value"]) - float(cand["grad_norm"]["value"]))
        if math.isfinite(loss_abs):
            max_loss_abs = max(max_loss_abs, loss_abs)
        if math.isfinite(grad_abs):
            max_grad_norm_abs = max(max_grad_norm_abs, grad_abs)

        for field in ("loss", "grad_norm", "post_step_weights", "update_successful", "num_zeros"):
            if base.get(field) != cand.get(field):
                mismatches.append({"step": idx, "field": field})
        if not _tensor_fingerprint_matches(base.get("logits"), cand.get("logits")):
            mismatches.append({"step": idx, "field": "logits"})

    if not lengths_match:
        mismatches.append(
            {
                "field": "steps",
                "baseline_count": len(base_steps),
                "candidate_count": len(cand_steps),
            }
        )

    return {
        "samples": sample_count,
        "passed": lengths_match and not mismatches,
        "max_loss_abs": max_loss_abs,
        "max_grad_norm_abs": max_grad_norm_abs,
        "tensor_fingerprint_rule": "raw_sha256_or_bf16_canonical_sha256",
        "mismatches": mismatches,
    }


__all__ = [
    "RunResult",
    "StepTrace",
    "compare_correctness_artifacts",
    "compare_step_traces",
    "load_result_artifact",
    "result_summary",
]
