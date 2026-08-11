# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""In-situ per-component timing inside the decode CUDA graph.

This exists because subtraction failed. Ablating a component and reading the change
in block time works only if the component's absence leaves everything else
unchanged, and for the MoE decode path it does not:
``multimem_all_gatherv_3tensor`` produces the routing map that sizes all downstream
grouped-GEMM work, so removing it silently changes how much work the experts do
(four failed attempts at exactly this are recorded in
``skills/run-qwen-model/EXPERIMENTS.md`` under "Measurement approaches that do not
work"). Nsys is equally unavailable here -- it costs ~28% end-to-end, and
that overhead is host-side, which desynchronizes the EP ranks and inflates the
spin-wait NVLS collectives by ~120x.

What is left is to time each component where it stands. An **external** event record
inside the capture (``cudaEventRecordWithFlags`` with ``cudaEventRecordExternal``)
becomes a graph node that re-records on every replay, so ``elapsed_time`` after a
replay yields that replay's duration for the enclosed region. The work descriptor,
the kernel sequence, and the cross-rank barrier order are all untouched; the only
addition is two event-record nodes.

The external flag is not optional and PyTorch does not expose it, hence the ctypes
call on the runtime. A plain ``torch.cuda.Event.record()`` during capture *appears*
to work -- it raises nothing and becomes a graph node -- but the event carries no
host-readable timestamp, and ``elapsed_time`` on it fails with
``cudaErrorInvalidValue``. That was established by standalone probes rather than
assumed: plain events fail, externally-flagged events work. The flagged record must
also target the *capturing* stream, which ``torch.cuda.graph()`` picks itself unless
told otherwise, so the stream is read back at record time rather than assumed --
getting that wrong returns ``cudaErrorIllegalState`` rather than anything
descriptive.

Each event pair carries a fixed cost of roughly 5-10 us, which would be a large
relative error on a 10-40 us component. It is calibrated rather than ignored: a
``_calib`` site wraps an empty region in the same graph, so its reading *is* the
per-pair overhead, and every other site is reported both raw and with that
subtracted. The probe also confirmed the decomposition is additive -- summed sites
reconstructed the whole graph within 1.3% -- which is what makes subtracting a
constant legitimate.

Two design points worth stating, because both were the cheap way out of a problem:

1. **Only a couple of occurrences per site are instrumented, not all 48 layers.**
   Every event record is a graph node, and ~600 of them would inflate the very
   block time being attributed. The layers are structurally identical, so sampling
   two of them and scaling by ``MCORE_INSITU_WRAP`` estimates the total; two rather
   than one so that a layer which turns out to be atypical is visible instead of
   silently wrong.
2. **The occurrence counter wraps at ``MCORE_INSITU_WRAP``** (default 48, the layer
   count) rather than tracking capture boundaries explicitly. Every full block pass
   calls each site exactly once per layer, so wrapping realigns occurrence indices
   across the several graphs the bucketed CUDA-graph pool captures, without needing
   a capture id that torch does not expose.

Numbers from this module are per-*replay* samples of one layer. Cross-check their
scaled sum against the whole-block figure from ``step_gpu_timing.py`` before
quoting any of them; if the parts do not reconstruct the whole, the sampling
assumption is broken and the components are meaningless.

Env-gated (``MCORE_INSITU_TIMING``); default off, zero cost when off.
"""

# Reports go to stderr on purpose: this runs inside a multi-rank inference server
# whose logger configuration is the caller's, and a diagnostic must not depend on it.
# pylint: disable=bad-builtin

import os
import statistics
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

ENABLED: bool = os.environ.get("MCORE_INSITU_TIMING", "0") == "1"
# Which per-layer occurrences to instrument. Two mid-stack layers by default: layer
# 0 can differ (no fused hand-off from a predecessor) and the last layer feeds
# final_layernorm rather than another block.
_OCC: Tuple[int, ...] = tuple(
    int(x) for x in os.environ.get("MCORE_INSITU_OCC", "8,24").split(",") if x.strip()
)
# Occurrences per full block pass; also the factor a single layer is scaled by.
WRAP: int = int(os.environ.get("MCORE_INSITU_WRAP", "48"))

_events: Dict[Tuple[str, int], Tuple["object", "object"]] = {}
_counts: Dict[str, int] = {}
_samples: Dict[Tuple[str, int], List[float]] = {}
_order: List[str] = []
_reports = 0
_first_error: Optional[str] = None


CALIB = "_calib"
_record_ext = None
_runtime_err: Optional[str] = None


def _external_record():
    """Resolve ``cudaEventRecordWithFlags`` once; None if it cannot be reached."""
    global _record_ext, _runtime_err
    if _record_ext is not None or _runtime_err is not None:
        return _record_ext
    import ctypes

    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        fn = lib.cudaEventRecordWithFlags
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        _record_ext = fn
        return fn
    _runtime_err = "could not load libcudart for cudaEventRecordWithFlags"
    print(f"[INSITU] disabled: {_runtime_err}", file=sys.stderr, flush=True)
    return None


def _rec(event, stream_ptr) -> bool:
    """External-record ``event`` on ``stream_ptr``. False on any failure, once-logged."""
    global _runtime_err
    import ctypes

    fn = _external_record()
    if fn is None:
        return False
    rc = fn(ctypes.c_void_p(event.cuda_event), stream_ptr, 1)  # cudaEventRecordExternal
    if rc != 0 and _runtime_err is None:
        _runtime_err = f"cudaEventRecordWithFlags returned {rc}"
        print(f"[INSITU] disabled: {_runtime_err}", file=sys.stderr, flush=True)
    return rc == 0


def _capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _ensure_events(name: str) -> None:
    """Allocate this site's event pairs, outside any capture.

    Creating a CUDA event while the stream is capturing is not reliably legal, and
    torch creates the underlying ``cudaEvent_t`` lazily on first record, so each
    event is also recorded once eagerly here -- otherwise ``cuda_event`` is null and
    the external record would target nothing.
    """
    import torch

    if name not in _order:
        _order.append(name)
    for occ in _OCC:
        if (name, occ) not in _events:
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for ev in pair:
                ev.record()
            _events[(name, occ)] = pair


@contextmanager
def site(name: str):
    """Time the enclosed region on every graph replay, for sampled occurrences.

    A no-op unless the gate is on. Outside capture this only allocates events; the
    occurrence counter is deliberately not advanced there, so eager steps (prefill,
    warmup) cannot shift the occurrence-to-layer mapping.
    """
    if not ENABLED:
        yield
        return
    if not _capturing():
        _ensure_events(name)
        yield
        return

    n = _counts.get(name, 0)
    _counts[name] = (n + 1) % WRAP
    pair = _events.get((name, n)) if n in _OCC else None
    if pair is None or _runtime_err is not None:
        yield
        return

    import ctypes

    import torch

    # The capturing stream, read rather than assumed: an external record on any
    # other stream is cudaErrorIllegalState, and torch.cuda.graph() chooses its own
    # side stream unless one is passed in.
    sp = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if not _rec(pair[0], sp):
        yield
        return
    try:
        yield
    finally:
        _rec(pair[1], sp)


def report(block_ms: Optional[float] = None) -> None:
    """Read every instrumented pair and print a scaled attribution.

    Caller must be at a point where the last replay has been submitted; this
    synchronizes once, which is why it belongs on a once-per-N-steps path rather
     than in the step itself.
    """
    global _reports
    if not ENABLED or not _events:
        return
    import torch

    global _first_error
    torch.cuda.synchronize()
    for key, (st, en) in _events.items():
        try:
            _samples.setdefault(key, []).append(st.elapsed_time(en))
        except Exception as ex:
            # A read before the graph's first replay legitimately fails, but so does
            # a broken premise, and the two are indistinguishable without the text.
            if _first_error is None:
                _first_error = f"{key}: {type(ex).__name__}: {ex}"
                print(
                    f"[INSITU] first elapsed_time failure -- {_first_error}",
                    file=sys.stderr,
                    flush=True,
                )

    def _med(name: str):
        """Median-of-medians across the sampled layers, plus the per-layer detail."""
        per_occ = []
        for occ in _OCC:
            vals = _samples.get((name, occ))
            if vals:
                per_occ.append((occ, statistics.median(vals[-64:])))
        if not per_occ:
            return None, []
        return statistics.median([v for _, v in per_occ]), per_occ

    # The empty-region site reads the per-pair event overhead directly, so every
    # other site is quoted net of it. Without this a 10 us component reads ~2x high.
    bias, _ = _med(CALIB)
    bias = bias or 0.0

    _reports += 1
    rank = int(os.environ.get("RANK", "0"))
    lines = [
        f"[INSITU r{rank}] report {_reports}: per-layer us (net of "
        f"{bias * 1000:.1f}us/pair event overhead), scaled x{WRAP}"
    ]
    total = 0.0
    for name in _order:
        if name == CALIB:
            continue
        med, per_occ = _med(name)
        if med is None:
            continue
        net = max(med - bias, 0.0)
        scaled = net * WRAP
        total += scaled
        detail = "  ".join(f"L{o}:{(v - bias) * 1000:6.1f}" for o, v in per_occ)
        spread = ""
        if len(per_occ) > 1:
            lo, hi = min(v - bias for _, v in per_occ), max(v - bias for _, v in per_occ)
            if lo > 0:
                spread = f"  spread {(hi - lo) / lo * 100:3.0f}%"
        lines.append(
            f"  {name:<18} {net * 1000:7.1f} us/layer -> {scaled:6.3f} ms/step  [{detail}]{spread}"
        )
    lines.append(f"  {'SUM of sites':<18} {'':>7}     -> {total:6.3f} ms/step")
    if block_ms:
        lines.append(
            f"  block total {block_ms:.3f} ms -> sites {100 * total / block_ms:4.1f}%, "
            f"{block_ms - total:6.3f} ms unattributed"
        )
    print("\n".join(lines), file=sys.stderr, flush=True)
