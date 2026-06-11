# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Lightweight, opt-in Perfetto/Chrome-trace markers for the checkpoint paths.

This is a source-level alternative to runtime monkey-patching: regions of
interest are annotated directly with ``trace_region``, which works both as a
function decorator (for whole functions) and as a ``with`` context manager
(for sub-function regions)::

    @trace_region("load")
    def load(...):
        ...

    with trace_region("load_common"):
        common_state_dict = load_common(checkpoint_dir)

When disabled the marker is a near-zero-cost passthrough (one dict lookup and
a generator enter/exit), so the annotations can stay in the source year-round.

Enabled via env vars (set by the launch script before ``python`` starts):

    CKPT_PERFETTO_TRACE=0|1      # default 0 -> all markers are no-ops
    CKPT_PERFETTO_OUT=<dir>      # output dir (falls back to LOG_DIR)
    CKPT_PERFETTO_RANKS="0 1 2"  # subset of ranks to trace; empty = all ranks

Each traced rank writes ``$CKPT_PERFETTO_OUT/perfetto_train_rank<R>.json`` in
the streaming Chrome-trace layout (one ``B``/``E`` event per line, no closing
``]}``) so the file stays valid even if the job is killed mid-write and so it
can be consumed by the same ``perfetto_merge.py`` tooling used for the
monkey-patch-based traces. Load the merged result in https://ui.perfetto.dev.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


class _Tracer:
    """Streaming Chrome-trace JSON writer (one ``B``/``E`` event per line).

    The trailing ``]}`` is intentionally never written: the Perfetto UI
    tolerates an unclosed array, and this keeps the file valid even if the
    process is killed (``scancel`` SIGKILL, OOM) mid-write.
    """

    def __init__(self, path):
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._pid = os.getpid()
        self._fh = open(path, "a", buffering=1)
        self._lock = threading.Lock()
        if self._fh.tell() == 0:
            self._fh.write('{"traceEvents":[\n')

    def _emit(self, event):
        line = json.dumps(event, default=str) + ",\n"
        with self._lock:
            self._fh.write(line)

    def _event(self, name, ph, args=None):
        event = {
            "name": name,
            "ph": ph,
            "ts": time.time_ns() / 1000.0,
            "pid": self._pid,
            "tid": threading.get_native_id(),
        }
        if args:
            event["args"] = args
        return event

    def begin(self, name, args=None):
        self._emit(self._event(name, "B", args))

    def end(self, name, args=None):
        self._emit(self._event(name, "E", args))


# Module-global tracer, lazily initialized on first use from the env vars.
_tracer: Optional[_Tracer] = None
_init_done = False
_init_lock = threading.Lock()


def _this_rank() -> int:
    """Read the current rank from whichever launcher env var is set."""
    for var in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "LOCAL_RANK"):
        val = os.environ.get(var, "").strip()
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _rank_enabled(rank: int) -> bool:
    raw = os.environ.get("CKPT_PERFETTO_RANKS", "").strip()
    if not raw:
        return True  # default: all ranks
    selected = set()
    for tok in raw.replace(",", " ").split():
        try:
            selected.add(int(tok))
        except ValueError:
            pass
    return rank in selected


def _get_tracer() -> Optional[_Tracer]:
    """Return the process tracer, initializing it once from the environment.

    Returns None (and stays None for the process lifetime) when tracing is
    disabled or this rank is not selected, so every marker becomes a cheap
    passthrough.
    """
    global _tracer, _init_done
    if _init_done:
        return _tracer
    with _init_lock:
        if _init_done:
            return _tracer
        _init_done = True
        if os.environ.get("CKPT_PERFETTO_TRACE", "0") == "1":
            rank = _this_rank()
            out_dir = (
                os.environ.get("CKPT_PERFETTO_OUT", "").strip()
                or os.environ.get("LOG_DIR", "").strip()
            )
            if _rank_enabled(rank) and out_dir:
                path = Path(out_dir) / f"perfetto_train_rank{rank}.json"
                _tracer = _Tracer(path)
        return _tracer


@contextmanager
def trace_region(name: str):
    """Mark a region (``with``) or whole function (``@``) on the Perfetto trace.

    ``contextlib.contextmanager`` returns an object that is both a context
    manager and a decorator, so a single helper covers both annotation styles.
    No-op when tracing is disabled or this rank is not selected.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield
        return
    tracer.begin(name)
    try:
        yield
    finally:
        tracer.end(name)
