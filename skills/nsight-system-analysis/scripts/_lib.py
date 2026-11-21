# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for nsight-system-analysis scripts.

All functions here are deterministic, workload-agnostic, and operate on
nsys-exported sqlite files.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections.abc import Iterable
from contextlib import contextmanager

# ---------- sqlite ----------


@contextmanager
def open_sqlite(path: str):
    """Read-only sqlite connection."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        yield con
    finally:
        con.close()


def query_kernels(con: sqlite3.Connection, where: str = "1=1", params: tuple = ()):
    """Return list of (start_ns, end_ns, stream_id, demangled_name) tuples."""
    cur = con.execute(
        f"""
        SELECT k.start, k.end, k.streamId,
               COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE {where}
        ORDER BY k.start
        """,
        params,
    )
    return cur.fetchall()


def query_memcpy(con: sqlite3.Connection, where: str = "1=1", params: tuple = ()):
    """Return list of (start_ns, end_ns, stream_id) tuples for memcpys."""
    cur = con.execute(
        f"""
        SELECT start, end, streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE {where}
        ORDER BY start
        """,
        params,
    )
    return cur.fetchall()


# ---------- interval math ----------


def union_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping intervals. Returns sorted non-overlapping list."""
    ivals = sorted(intervals)
    if not ivals:
        return []
    out = [list(ivals[0])]
    for s, e in ivals[1:]:
        if s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def union_total_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Total ns covered by at least one interval."""
    return sum(e - s for s, e in union_intervals(intervals))


def subtract_union(
    a: Iterable[tuple[int, int]],
    b_union: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return union(a) minus b_union as a list of intervals.

    b_union must be already merged (non-overlapping, sorted). Result is sorted
    non-overlapping.
    """
    a_union = union_intervals(a)
    out = []
    for s, e in a_union:
        cursor = s
        for bs, be in b_union:
            if be <= cursor:
                continue
            if bs >= e:
                break
            if bs > cursor:
                out.append((cursor, min(bs, e)))
            cursor = max(cursor, be)
            if cursor >= e:
                break
        if cursor < e:
            out.append((cursor, e))
    return out


def clip(
    intervals: Iterable[tuple[int, int]],
    lo: int,
    hi: int,
) -> list[tuple[int, int]]:
    """Clip intervals to [lo, hi)."""
    out = []
    for s, e in intervals:
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2:
            out.append((s2, e2))
    return out


# ---------- taxonomy / YAML ----------


def load_taxonomy(path: str) -> dict[str, re.Pattern]:
    """Load a YAML taxonomy and compile its regexes.

    The YAML may have a top-level `Overall:` map or be a flat map; we accept
    either. Order is preserved (Python 3.7+ dict ordering).
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write(
            "ERROR: PyYAML required. Install with `pip install pyyaml` or run "
            "in an env that has it.\n"
        )
        sys.exit(2)
    with open(path) as f:
        data = yaml.safe_load(f)
    if (
        isinstance(data, dict)
        and "Overall" in data
        and isinstance(data["Overall"], dict)
    ):
        data = data["Overall"]
    if not isinstance(data, dict):
        sys.stderr.write(f"ERROR: taxonomy YAML at {path} is not a mapping.\n")
        sys.exit(2)
    return {name: re.compile(pattern) for name, pattern in data.items()}


def classify(name: str, taxonomy: dict[str, re.Pattern]) -> str | None:
    """First-match-wins classification. Returns category name or None."""
    for cat, rx in taxonomy.items():
        if rx.search(name):
            return cat
    return None


# Heuristic detection of "custom-fused" kernels — used by Step 4 to decide
# between op-group and module-slicing. The criteria are deliberately broad;
# false positives bias toward "use module-slicing", which is the safe default.

_CUSTOM_FUSED_HINTS = re.compile(
    r"_fused_[A-Za-z]|fused_.*kernel|"
    r"(?:qkv|rope|norm|adaln|gate|residual|silu|gelu|tanh|layernorm|rmsnorm)"
    r"_(?:.*_)?(?:qkv|rope|norm|adaln|gate|residual|silu|gelu|tanh|layernorm|rmsnorm|split|join|cat|sum|fwd|bwd)",
    re.IGNORECASE,
)


def looks_custom_fused(name: str) -> bool:
    """Heuristic: does this kernel name suggest fusion of 2+ ops?

    Used by the op-group-vs-module-slicing decision in Step 4. Matches kernel
    names like `_qkv_split_norm_rope_kernel`, `_fused_ln_adaln_fwd_kernel`,
    `triton_red_fused__to_copy_mul_native_layer_norm_*`, etc. Pure GEMM and
    pure MHA/SDPA names should not match.
    """
    if not name:
        return False
    # Triton fused-* kernels with 2+ op tokens
    if "triton_" in name and "fused_" in name:
        # Pull out the bit after `fused_` and look for 2+ op tokens
        tail = name.split("fused_", 1)[1]
        op_tokens = re.findall(
            r"(?:add|mul|sub|div|tanh|gelu|silu|relu|sigmoid|exp|log|"
            r"native_layer_norm|rms_norm|layer_norm|sum|view|cat|copy|clone|"
            r"rope|qkv|norm|adaln|gate)",
            tail,
        )
        if len(op_tokens) >= 2:
            return True
    return bool(_CUSTOM_FUSED_HINTS.search(name))


# ---------- output helpers ----------


def write_json(obj, fp=None):
    """Write JSON to stdout (default) or a file-like object."""
    fp = fp or sys.stdout
    json.dump(obj, fp, indent=2, sort_keys=False)
    fp.write("\n")


def err(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")


def die(msg: str, code: int = 1):
    err(f"ERROR: {msg}")
    sys.exit(code)
