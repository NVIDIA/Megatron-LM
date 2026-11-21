#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Capability report for the all-to-all backend that ``benchmark.py`` drives.

``standard_alltoall`` is the repository's existing ``MoEAlltoAllTokenDispatcher``.
``probe_environment`` reports what that backend supports on this node: precision, EP and
ETP sizes, zero-token rows, recompute, CUDA graphs and grouped GEMM. Anything the benchmark
does not exercise is reported as ``UNVERIFIED`` rather than assumed.

The MoE-layer arms (HybridEP, MoonEP virtual experts and the MoEScheduler planners) live in
``arms.py`` and ``arm_adapter.py`` and are compared by ``compare_arms.py`` on the same
logical workloads.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

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


def probe_environment(num_experts: int = 8) -> Capability:
    """Capability probe for the standard all-to-all backend on this node.

    Everything comes from the benchmark's own configuration and from divisibility;
    nothing is claimed from documentation.
    """
    cap = Capability(backend="standard_alltoall")

    cap.precision["bf16"] = SUPPORTED  # the benchmark runs bf16 end to end

    # benchmark.py always builds its experts with moe_grouped_gemm=False.
    cap.grouped_gemm = UNSUPPORTED
    cap.precision["fp8"] = UNVERIFIED
    cap.notes.append("experts run as a per-expert loop (moe_grouped_gemm=False) in every mode")

    # EP sizes are constrained by divisibility, not by preference
    for ep in (1, 2, 4, 8):
        key = f"ep={ep}"
        if num_experts % ep == 0:
            cap.ep_sizes[key] = SUPPORTED
        else:
            cap.ep_sizes[key] = UNSUPPORTED
    cap.etp["etp=1"] = SUPPORTED
    # benchmark.py rejects --etp > 1: with the identity expert, combine would measure
    # topk * etp and fail the round-trip oracle.
    cap.etp["etp>1"] = UNSUPPORTED

    # W5 exercises empty destinations with nonempty sources; every workload uses fixed top-k.
    cap.recompute = UNVERIFIED  # not exercised by the benchmark
    cap.cuda_graph = UNVERIFIED  # not exercised by the benchmark
    return cap


@dataclass
class StandardAlltoAllAdapter:
    """Names the repository's existing all-to-all dispatcher and reports its capability."""

    name: str = "standard_alltoall"

    def capability(self, num_experts: int = 8) -> Capability:
        return probe_environment(num_experts)


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
