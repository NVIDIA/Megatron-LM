"""Benchmark result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
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


__all__ = ["RunResult", "StepTrace"]
