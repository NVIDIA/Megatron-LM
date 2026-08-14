# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU-side step accounting for dynamic inference decode, without a profiler.

Two questions this answers that nsys cannot on this workload:

1. **How much of a decode step is the transformer-block CUDA graph?** Everything
   else (embedding, final norm, logits GEMM, sampling, bookkeeping) is eager and
   host-driven, so the block-graph share decides whether the remaining gap to
   vLLM lives inside the model's GPU work or in exposed host/eager time.

2. **How skewed are the EP ranks?** The NVLS dispatch/combine collectives are
   spin-waits, so their cost is set by cross-rank arrival skew rather than by
   bytes moved. All four ranks are processes on one node, so ``perf_counter_ns``
   (CLOCK_MONOTONIC) is directly comparable between them and the spread of
   step-entry timestamps measures that skew directly.

Why not nsys: on this workload nsys costs ~28% end-to-end regardless of
``--cuda-graph-trace`` mode, and that overhead is host-side, which desynchronizes
the EP ranks and inflates exactly these spin-wait collectives (~120x observed).
Any absolute idle or barrier number read from such a trace is unusable.

Cost control: CUDA events are recorded into a ring of pairs and only read back
once the ring wraps, so a step never synchronizes on its own work. Env-gated
(``MCORE_INFER_STEP_GPU_TIMING``); default off.
"""

# Reports go to stderr on purpose: this runs inside a multi-rank inference server
# whose logger configuration is the caller's, and a diagnostic must not depend on it.
# pylint: disable=bad-builtin

import os
import statistics
import sys
import time

USE_STEP_GPU_TIMING: bool = os.environ.get("MCORE_INFER_STEP_GPU_TIMING", "0") == "1"
# Number of event pairs held before a readback. One readback per RING steps.
_RING: int = int(os.environ.get("MCORE_INFER_STEP_GPU_RING", "64"))
_REPORT_EVERY: int = int(os.environ.get("MCORE_INFER_STEP_GPU_REPORT_EVERY", "256"))
# Steps to discard before accumulating, so graph warmup and the first ramp steps
# do not pollute the medians.
_SKIP: int = int(os.environ.get("MCORE_INFER_STEP_GPU_SKIP", "64"))

_installed = False


class _Acc:
    def __init__(self):
        self.events = []  # [(start_event, end_event)] ring
        self.slot = 0
        self.calls = 0
        self.gpu_ms = []  # block-graph GPU durations
        self.period_ms = []  # host wall time between consecutive block entries
        self.entry_ns = []  # step-entry timestamps, for cross-rank skew
        self.last_entry = None
        self.pending = []  # slots with recorded-but-unread events


_acc = _Acc()


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _drain():
    """Read back every recorded pair in the ring. Called only when the ring wraps,
    so the work being queried is long finished and this does not stall the step."""
    if not _acc.pending:
        return
    # Only the newest pair could still be in flight; syncing on it is what makes
    # the readback safe without syncing the whole device.
    _acc.events[_acc.pending[-1]][1].synchronize()
    for s in _acc.pending:
        st, en = _acc.events[s]
        try:
            _acc.gpu_ms.append(st.elapsed_time(en))
        except Exception:
            pass
    _acc.pending.clear()


def _dump_entry_times():
    """Write this rank's step-entry timestamps so skew can be computed across ranks.

    Comparable across ranks because all four are processes on the same node.
    """
    d = os.environ.get("MCORE_INFER_STEP_GPU_DUMP", "")
    if not d or not _acc.entry_ns:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"entry_r{_rank()}.txt"), "w", encoding="utf-8") as f:
            for t in _acc.entry_ns:
                f.write(f"{t}\n")
    except Exception as e:
        print(f"[STEP_GPU] dump failed: {e}", file=sys.stderr, flush=True)


def _report():
    r = _rank()
    n = len(_acc.gpu_ms)
    if n < 8:
        return
    g = statistics.median(_acc.gpu_ms)
    lines = [f"[STEP_GPU r{r}] after {_acc.calls} block calls, {n} timed:"]
    lines.append(
        f"  block-graph GPU   median {g:7.3f} ms   "
        f"p10 {statistics.quantiles(_acc.gpu_ms, n=10)[0]:.3f} "
        f"p90 {statistics.quantiles(_acc.gpu_ms, n=10)[8]:.3f}"
    )
    if len(_acc.period_ms) >= 8:
        p = statistics.median(_acc.period_ms)
        lines.append(
            f"  step period (host) median {p:7.3f} ms   "
            f"p10 {statistics.quantiles(_acc.period_ms, n=10)[0]:.3f} "
            f"p90 {statistics.quantiles(_acc.period_ms, n=10)[8]:.3f}"
        )
        lines.append(
            f"  => block graph is {100 * g / p:5.1f}% of the step; "
            f"{p - g:6.3f} ms/step is outside it"
        )
    print("\n".join(lines), file=sys.stderr, flush=True)
    # Per-component in-situ attribution, printed against the block total it has to
    # reconcile with -- the parts are only meaningful if they reconstruct the whole.
    try:
        from megatron.core.inference import insitu_timing

        insitu_timing.report(block_ms=g)

        from megatron.core.inference.moe import expert_histogram

        expert_histogram.dump()
    except Exception as e:
        print(f"[STEP_GPU] insitu report failed: {e}", file=sys.stderr, flush=True)
    # Dump here rather than only at exit: the harness SIGTERMs the server, which
    # skips atexit handlers, so an exit-only dump would usually produce nothing.
    _dump_entry_times()


def install():
    """Wrap TransformerBlock.__call__ so the CUDA-graph path is timed."""
    global _installed
    if not USE_STEP_GPU_TIMING or _installed:
        return
    import atexit

    import torch

    from megatron.core.transformer.transformer_block import TransformerBlock

    orig_call = TransformerBlock.__call__
    if getattr(orig_call, "_step_gpu_timed", False):
        return

    for _ in range(_RING):
        _acc.events.append(
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        )

    def timed_call(self, *args, **kwargs):
        # Only the graph path is interesting; prefill and non-graph steps would
        # mix different work into the same medians.
        if not self._should_call_local_cudagraph(*args, **kwargs):
            return orig_call(self, *args, **kwargs)

        _acc.calls += 1
        if _acc.calls <= _SKIP:
            return orig_call(self, *args, **kwargs)

        now = time.perf_counter_ns()
        if _acc.last_entry is not None:
            _acc.period_ms.append((now - _acc.last_entry) / 1e6)
        _acc.last_entry = now
        _acc.entry_ns.append(now)

        s = _acc.slot
        if s == 0:
            _drain()
        st, en = _acc.events[s]
        st.record()
        try:
            return orig_call(self, *args, **kwargs)
        finally:
            en.record()
            _acc.pending.append(s)
            _acc.slot = (s + 1) % _RING
            if _REPORT_EVERY > 0 and _acc.calls % _REPORT_EVERY == 0:
                _drain()
                _report()

    timed_call._step_gpu_timed = True
    TransformerBlock.__call__ = timed_call
    atexit.register(_dump_entry_times)
    atexit.register(_report)
    _installed = True
    print(
        f"[STEP_GPU] installed (ring {_RING}, skip {_SKIP}, " f"report every {_REPORT_EVERY})",
        file=sys.stderr,
        flush=True,
    )
