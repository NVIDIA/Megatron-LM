#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fail on *new* reads of global process-group state in ``megatron/core``.

``megatron.core.parallel_state`` holds the process groups for a single, global parallel grid.
Reading it inside library code is being removed: for a model built on independent parallel grids
it returns the wrong grid, silently. See ``docs/developer/parallel-state-deprecation.md``.

This check is a ratchet, not a cleanup. Every existing violation is recorded in an allowlist so
the build stays green; the check fails only when a *new* one appears, or when the allowlist
claims a violation that no longer exists (so the allowlist shrinks as the migration lands).

Usage::

    python tools/check_process_group_usage.py            # check
    python tools/check_process_group_usage.py --update   # regenerate the allowlist
    python tools/check_process_group_usage.py --stats    # summarize without failing
"""

import argparse
import ast
import json
import pathlib
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "megatron" / "core"
ALLOWLIST = REPO_ROOT / "tools" / "process_group_usage_allowlist.json"

# Files permitted to read global state: the module itself, the collection that bridges to it,
# and the compatibility helper. Everything else in megatron/core is subject to the ratchet.
EXEMPT = {"megatron/core/parallel_state.py", "megatron/core/process_groups_config.py"}

# Accessors that read the global grid. Deliberately excludes initialize/destroy/is_initialized
# (the intended long-term surface) and the virtual-pipeline and memory-buffer globals, which have
# no replacement yet and are tracked separately.
DEPRECATED_PREFIXES = ("get_",)
DEPRECATED_SUFFIXES = ("_group", "_groups", "_rank", "_world_size", "_src_rank")
NOT_DEPRECATED = {
    "get_nccl_options",
    "get_all_ranks",
    "get_global_memory_buffer",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
}


def _is_deprecated_accessor(name: str) -> bool:
    if name in NOT_DEPRECATED:
        return False
    return name.startswith(DEPRECATED_PREFIXES) and name.endswith(DEPRECATED_SUFFIXES)


def _violations_in(path: pathlib.Path):
    """Yield (lineno, kind, detail) for global process-group reads in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return

    # Names imported directly from parallel_state, e.g. `from ..parallel_state import get_x_group`
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "parallel_state" in node.module:
            for alias in node.names:
                if _is_deprecated_accessor(alias.name):
                    imported.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # parallel_state.get_x_group(...) / mpu.get_x_group(...)
        if isinstance(func, ast.Attribute) and _is_deprecated_accessor(func.attr):
            base = func.value
            if isinstance(base, ast.Name) and base.id in ("parallel_state", "mpu", "ps"):
                yield node.lineno, "accessor", f"{base.id}.{func.attr}"
        # bare get_x_group(...) imported from parallel_state
        elif isinstance(func, ast.Name) and func.id in imported:
            yield node.lineno, "accessor", func.id
        # ProcessGroupCollection.use_mpu_process_groups(...)
        elif isinstance(func, ast.Attribute) and func.attr == "use_mpu_process_groups":
            yield node.lineno, "shim", "use_mpu_process_groups"


def scan():
    """Return {relative path: sorted list of "lineno:kind:detail"}."""
    found = {}
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in EXEMPT:
            continue
        hits = [f"{ln}:{kind}:{detail}" for ln, kind, detail in _violations_in(path)]
        if hits:
            found[rel] = sorted(set(hits))
    return found


def _load_allowlist():
    if not ALLOWLIST.exists():
        return {}
    return json.loads(ALLOWLIST.read_text(encoding="utf-8")).get("allowed", {})


def _counts(found):
    c = Counter()
    for hits in found.values():
        for h in hits:
            c[h.split(":", 2)[1]] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="regenerate the allowlist")
    ap.add_argument("--stats", action="store_true", help="summarize without failing")
    args = ap.parse_args()

    found = scan()
    counts = _counts(found)
    total = sum(counts.values())

    if args.update:
        ALLOWLIST.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Grandfathered reads of global process-group state in megatron/core. "
                        "This list must only shrink. Regenerate with "
                        "`python tools/check_process_group_usage.py --update` after REMOVING "
                        "usage -- never to silence a new violation."
                    ),
                    "total": total,
                    "allowed": found,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Wrote {ALLOWLIST.relative_to(REPO_ROOT)}: {total} grandfathered site(s) "
            f"across {len(found)} file(s)."
        )
        return 0

    if args.stats:
        print(f"{total} global process-group read(s) across {len(found)} file(s) in megatron/core")
        for kind, n in counts.most_common():
            print(f"  {kind:10} {n}")
        return 0

    allowed = _load_allowlist()

    added, removed = {}, {}
    for rel, hits in found.items():
        new = sorted(set(hits) - set(allowed.get(rel, [])))
        if new:
            added[rel] = new
    for rel, hits in allowed.items():
        gone = sorted(set(hits) - set(found.get(rel, [])))
        if gone:
            removed[rel] = gone

    if added:
        n = sum(len(v) for v in added.values())
        print(f"ERROR: {n} new read(s) of global process-group state in megatron/core:\n")
        for rel, hits in sorted(added.items()):
            for h in hits:
                ln, kind, detail = h.split(":", 2)
                print(f"  {rel}:{ln}  {detail}")
        print(
            "\nmegatron/core must not read process groups from parallel_state. Accept a "
            "ProcessGroupCollection or an explicit torch.distributed.ProcessGroup from the "
            "caller and pass it through.\n"
            "Note that ProcessGroupCollection.use_mpu_process_groups() is NOT a valid "
            "replacement -- it reads the same global state.\n"
            "See docs/developer/parallel-state-deprecation.md\n"
        )
        return 1

    if removed:
        n = sum(len(v) for v in removed.values())
        print(
            f"{n} allowlisted site(s) no longer exist -- nice. Refresh the allowlist:\n"
            f"  python tools/check_process_group_usage.py --update\n"
        )
        for rel, hits in sorted(removed.items()):
            for h in hits:
                print(f"  {rel}:{h.split(':', 2)[0]}")
        return 1

    print(f"OK: no new global process-group reads ({total} grandfathered).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
