"""Startup-phase wall-clock timing for throughput debugging.

Records a short sequence of named stamps (dist init, model build, iter 1/2
boundaries, ...) and flushes them into the run JSON as a top-level ``phases``
list. Each stamp also prints a one-line marker to stderr on rank 0 so the
timeline is visible live in the slurm log without opening the JSON.

Rank 0 only. No-op on other ranks.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_PHASES: list[dict[str, Any]] = []
_WRITER: Any = None
_INSTALLED = False
_TRAIN_STEP_CALLS = 0


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def stamp(name: str) -> None:
    """Record a named wall-clock stamp on rank 0 and flush to the writer."""
    if _rank() != 0:
        return
    now = time.time()
    prev = _PHASES[-1]["wall"] if _PHASES else now
    entry = {"name": name, "wall": now, "delta": now - prev}
    _PHASES.append(entry)
    print(
        f"[phase] {name} wall={now:.3f} (+{entry['delta']:.2f}s)",
        file=sys.stderr,
        flush=True,
    )
    if _WRITER is not None:
        try:
            _WRITER.set(phases=_PHASES)
        except Exception:
            pass


def install(writer: Any) -> None:
    """Attach the writer and monkey-patch ``torch.distributed.init_process_group``.

    Stamps around ``setup_model_and_optimizer`` and ``train_step`` are added by
    ``hooks.patch_*`` to avoid double-wrapping the same Megatron symbols.
    """
    global _WRITER, _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _WRITER = writer

    try:
        import torch.distributed as dist

        original_init = dist.init_process_group

        def wrapped_init(*args, **kwargs):
            stamp("before_dist_init")
            result = original_init(*args, **kwargs)
            stamp("after_dist_init")
            return result

        dist.init_process_group = wrapped_init
    except Exception:
        pass


def note_train_step_start() -> None:
    """Called from the patched ``train_step`` wrapper; stamps iter 1/2 starts."""
    global _TRAIN_STEP_CALLS
    _TRAIN_STEP_CALLS += 1
    if _TRAIN_STEP_CALLS <= 2:
        stamp(f"iter_{_TRAIN_STEP_CALLS}_start")


def note_train_step_end() -> None:
    """Called from the patched ``train_step`` wrapper; stamps iter 1/2 ends."""
    if _TRAIN_STEP_CALLS <= 2:
        stamp(f"iter_{_TRAIN_STEP_CALLS}_end")
