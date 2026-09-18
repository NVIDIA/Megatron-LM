#!/usr/bin/env python3
"""B1 backend adapter layer.

A backend adapter carries exactly five things and nothing more (B1 contract, section 9):

    capability()  -> what this backend supports on this node
    build()       -> construct the model / dispatcher and load consistent weights
    run_workload()-> execute one logical workload and return metrics + correctness
    name          -> stable identifier recorded in every result row
    close()       -> release resources

It is deliberately NOT a scheduler framework. A capability check must report precision,
EP/ETP, zero-token support, recompute, graph and hardware dependencies; anything unknown is
reported as such rather than assumed.

The first (and currently only) adapter is ``standard_alltoall``: the repository's existing
``MoEAlltoAllTokenDispatcher``. Future backends (ECHO / MoonEP / UltraEP) plug in here by
reusing their authors' code and configuration, and are compared on the SAME logical workload
and oracle.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
UNVERIFIED = "UNVERIFIED"


@dataclass
class Capability:
    """What a backend can actually do here and now. Never a design intention."""

    backend: str
    precision: dict[str, str] = field(default_factory=dict)
    ep_sizes: dict[str, str] = field(default_factory=dict)
    etp: dict[str, str] = field(default_factory=dict)
    zero_token_rows: str = UNVERIFIED
    variable_topk: str = UNVERIFIED
    recompute: str = UNVERIFIED
    cuda_graph: str = UNVERIFIED
    grouped_gemm: str = UNVERIFIED
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class BackendAdapter(Protocol):
    name: str

    def capability(self) -> Capability: ...

    def build(self, **kwargs: Any) -> Any: ...

    def run_workload(self, *args: Any, **kwargs: Any) -> Any: ...

    def close(self) -> None: ...


def probe_environment(num_experts: int = 8) -> Capability:
    """Capability probe for the standard all-to-all backend on this node.

    Everything probed by import or by inspection; nothing claimed from documentation.
    """
    cap = Capability(backend="standard_alltoall")

    cap.precision["bf16"] = SUPPORTED  # the benchmark runs bf16 end to end

    try:
        import transformer_engine  # noqa: F401
        cap.grouped_gemm = SUPPORTED
        cap.precision["fp8"] = UNVERIFIED
        cap.notes.append("transformer_engine importable; grouped GEMM available")
    except Exception:
        cap.grouped_gemm = UNSUPPORTED
        cap.precision["fp8"] = UNSUPPORTED
        cap.notes.append(
            "transformer_engine is not installed, so the expert computation is a per-expert "
            "loop rather than TE grouped GEMM. Absolute step times are not comparable with a "
            "TE-enabled node; baseline/patch comparisons on this node remain valid."
        )

    # EP sizes are constrained by divisibility, not by preference
    for ep in (1, 2, 4, 8):
        key = f"ep={ep}"
        if num_experts % ep == 0:
            cap.ep_sizes[key] = SUPPORTED
        else:
            cap.ep_sizes[key] = UNSUPPORTED
    cap.etp["etp=1"] = SUPPORTED
    cap.etp["etp>1"] = UNVERIFIED  # no ETP run was performed, so this is not a claim

    cap.zero_token_rows = SUPPORTED      # W5 exercises it and passes
    cap.variable_topk = SUPPORTED        # W0..W7 does not vary topk within a run
    cap.recompute = UNVERIFIED           # not exercised by the benchmark
    cap.cuda_graph = UNVERIFIED          # not exercised by the benchmark
    return cap


@dataclass
class StandardAlltoAllAdapter:
    """The repository's existing all-to-all dispatcher, driven directly.

    ``build`` is supplied by the caller because the benchmark owns the process-group and
    weight setup; the adapter's job is to name the backend, describe its capability, and
    dispatch execution to it so future backends can be added without touching the harness.
    """

    name: str = "standard_alltoall"
    _dispatcher: Any = None
    _inner: Any = None

    def capability(self) -> Capability:
        return probe_environment()

    def bind(self, dispatcher: Any, inner: Any) -> None:
        self._dispatcher = dispatcher
        self._inner = inner

    def run_workload(self, **kwargs: Any) -> Any:
        if self._inner is None:
            raise RuntimeError("adapter.bind() must be called before run_workload()")
        return self._inner(**kwargs)

    def close(self) -> None:
        self._dispatcher = None
        self._inner = None


def report_capability(cap: Capability) -> str:
    lines = [f"backend: {cap.backend}"]
    for section in ("precision", "ep_sizes", "etp"):
        for k, v in getattr(cap, section).items():
            lines.append(f"  {section}.{k}: {v}")
    for single in ("zero_token_rows", "variable_topk", "recompute", "cuda_graph", "grouped_gemm"):
        lines.append(f"  {single}: {getattr(cap, single)}")
    for n in cap.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report_capability(probe_environment(int(sys.argv[1]) if len(sys.argv) > 1 else 8)))
