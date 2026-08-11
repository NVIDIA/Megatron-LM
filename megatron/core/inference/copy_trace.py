# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""One-shot naming of the copies inside a single decode layer.

A per-kernel budget can say "there is one 2.7 us bf16 copy per layer" but not
which line of Python asks for it, because at decode every kernel is replayed
from a captured graph and has no launch-site backtrace left in the profile. This
mode runs one layer under a dispatch interceptor and prints each copy-like op
once, with its shape and the Megatron frames that requested it — enough to
decide whether the copy is removable or has to be fused.

Enable with ``MCORE_COPY_TRACE=1``; it disarms itself after the first layer, so
the cost is one layer's worth of Python dispatch on one step.
"""

# Reports go to stdout/stderr on purpose: this runs inside a multi-rank inference
# server whose logger configuration is the caller's, and a diagnostic must not
# depend on it.
# pylint: disable=bad-builtin

import os
import traceback
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ENABLED: bool = os.environ.get("MCORE_COPY_TRACE", "0") == "1"
SKIP: int = int(os.environ.get("MCORE_COPY_TRACE_SKIP", "500"))

# Ops that move bytes without changing them: the copies a fusion could remove.
_WATCHED = {"copy_", "_to_copy", "clone", "contiguous", "cat", "index_select", "gather"}

_armed: bool = ENABLED
_calls: int = 0
_seen: Dict[Tuple, int] = {}


def _describe(t: Any) -> str:
    if isinstance(t, torch.Tensor):
        layout = "" if t.is_contiguous() else " strided"
        return f"{tuple(t.shape)}{layout} {str(t.dtype).split('.')[-1]}"
    return type(t).__name__


def _callsite() -> str:
    """The innermost Megatron frames, skipping this module and torch internals."""
    frames = []
    for f in reversed(traceback.extract_stack()[:-2]):
        if "/torch/" in f.filename or f.filename.endswith("copy_trace.py"):
            continue
        frames.append(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}")
        if len(frames) == 3:
            break
    return " <- ".join(frames)


class _CopyTrace(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(getattr(func, "overloadpacket", None), "__name__", str(func))
        if name in _WATCHED:
            key = (name, tuple(_describe(a) for a in args[:2]), _callsite())
            _seen[key] = _seen.get(key, 0) + 1
        return func(*args, **kwargs)


def ready() -> bool:
    """True once enough layer forwards have gone by to be past one-time work.

    Arming on the *first* forward catches the lazy expert-weight consolidation and
    reports 32 weight copies per layer, which is one-time cost misread as per-step.
    Gating on ``inference_context.is_decode_only()`` instead looked more precise but
    never fired at all: by the time the layer runs under graph capture the context
    is not reaching it as a keyword argument. A call counter needs nothing from the
    caller — ``MCORE_COPY_TRACE_SKIP`` forwards (default 500, i.e. past 48 layers of
    first-touch several times over) are ignored, then the next one is traced.
    """
    global _calls
    if not _armed:
        return False
    _calls += 1
    return _calls > SKIP


def trace_one_layer() -> Optional[_CopyTrace]:
    """Return a mode to wrap one layer with, or None once already traced."""
    global _armed
    if not _armed:
        return None
    _armed = False
    return _CopyTrace()


def report() -> None:
    """Print what the traced layer copied, widest tensor first."""
    if not _seen:
        return
    print("===== MCORE_COPY_TRACE: copy-like ops in one decode layer =====", flush=True)
    for (name, operands, site), count in sorted(_seen.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}x {name:14s} {' | '.join(operands):48s} {site}", flush=True)
    print("=" * 62, flush=True)
    _seen.clear()
