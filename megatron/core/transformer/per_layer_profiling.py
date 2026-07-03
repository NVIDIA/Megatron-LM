# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-layer measured profiling for transformer layers.

Collects, per ``TransformerLayer`` on logged steps only:

* forward wall time, via ``torch.cuda.Event`` recorded on the hot path but
  resolved lazily in :meth:`PerLayerProfiler.flush` (called only at log time),
  so the training loop never force-synchronizes per layer;
* per-layer memory from a single ``torch.cuda.memory_stats()`` snapshot per
  hook: an allocated-memory delta (``allocated_bytes.all.current`` post minus
  pre, approximating retained activations), a reserved-pool delta
  (``reserved_bytes.all.current``, which -- unlike the allocated delta -- does
  not collapse under recompute), and a running allocated peak
  (``allocated_bytes.all.peak`` at the layer boundary);
* a MoE-vs-dense tag read once at attach time from ``is_moe_layer``.

Per-device global peaks are *measured* once per interval via
:meth:`start_step` / :meth:`end_step`, not reconstructed from per-layer
numbers: both the allocated peak (``allocated_bytes.all.peak``) and the
reserved peak (``reserved_bytes.all.peak`` -- the caching allocator's total
pool incl. fragmentation and comm buffers, i.e. the OOM-relevant ceiling).

Parallelism (all measurements are per-rank / per-device):

* TP: per-layer memory is the local ``1/TP`` shard; forward time includes the
  blocking TP all-reduce.
* PP: this rank holds only its own layers; local indices are mapped to global
  layer numbers before logging. A step runs several micro-batches (1F1B), so
  hooks fire multiple times and pending events accumulate in a list.
* EP: MoE layers process a router-dependent, per-step-varying token count;
  their figures fluctuate by design and are kept separate via the MoE tag.

Scope and known limitations:

* Retained-memory delta collapses under activation recompute / CPU offload
  (activations are discarded), so it no longer reflects true activation size.
  It is reported as-is with this caveat; the running peak remains valid.
  A theoretical-vs-measured reconciliation is a natural follow-up.
* Backward timing is not measured. ``register_full_backward_hook`` fires
  unreliably under custom autograd (recompute) and mutates the autograd graph
  in ways that break pipeline-parallel output deallocation. Correct per-layer
  backward timing needs a graph-preserving mechanism (tensor grad hooks or
  autograd node instrumentation) and is left as follow-up.
* Under pipeline parallelism only the first stage's layers are emitted
  (logging gated to global rank 0); full cross-stage coverage is follow-up.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from megatron.core.utils import unwrap_model


class _MemSnapshot(NamedTuple):
    """One consistent CUDA allocator snapshot (bytes) from torch.cuda.memory_stats().

    reserved_* is the caching allocator's total cudaMalloc pool (includes
    fragmentation and inactive splits) -- the OOM-relevant ceiling that the
    allocated_* figures do not capture.
    """

    allocated: int  # allocated_bytes.all.current  (== memory_allocated())
    reserved: int  # reserved_bytes.all.current   (== memory_reserved())
    allocated_peak: int  # allocated_bytes.all.peak     (== max_memory_allocated())
    reserved_peak: int  # reserved_bytes.all.peak      (== max_memory_reserved())


# A pending forward sample awaiting flush():
#   (start_event, end_event, snapshot_before, snapshot_after)
_PendingFwd = Tuple[
    Optional["torch.cuda.Event"], Optional["torch.cuda.Event"], _MemSnapshot, _MemSnapshot
]


@dataclasses.dataclass
class PerLayerProfileStats:
    """Accumulated measured stats for one transformer layer.

    Memory semantics (read before trusting the numbers):

    * ``fwd_mem_allocated_delta_bytes`` -- allocated delta across the layer's
      forward (post minus pre). Approximates memory *retained* for backward.
      Collapses toward zero under activation recompute and is misleading under
      CPU offload. NOT the peak and NOT transient/comm buffers.
    * ``fwd_mem_peak_after_bytes`` -- running ``max_memory_allocated`` observed
      when the layer's forward returns, i.e. the peak *up to and including*
      this layer on this device (monotonic across layers within a step).
    * ``fwd_mem_peak_rise_bytes`` -- how much this layer pushed the running
      peak up (peak_after - peak_before), >= 0. Attribution of the ceiling.
    * ``fwd_mem_reserved_delta_bytes`` -- reserved-pool delta across the
      layer's forward (post minus pre) from ``reserved_bytes.all.current``.
      Coarser than the allocated delta (the allocator grabs in large chunks)
      but, unlike the allocated delta, does NOT collapse under recompute.
    """

    layer_idx: int
    is_moe_layer: bool = False

    fwd_time_ms: List[float] = dataclasses.field(default_factory=list)
    fwd_mem_allocated_delta_bytes: List[int] = dataclasses.field(default_factory=list)
    fwd_mem_reserved_delta_bytes: List[int] = dataclasses.field(default_factory=list)
    fwd_mem_peak_after_bytes: List[int] = dataclasses.field(default_factory=list)
    fwd_mem_peak_rise_bytes: List[int] = dataclasses.field(default_factory=list)
    num_samples: int = 0

    # Pending, unresolved CUDA events. Lists (not single slots) so that
    # multiple micro-batches per step under PP accumulate instead of
    # overwriting. Resolved and cleared in PerLayerProfiler.flush().
    _pending_fwd: List[_PendingFwd] = dataclasses.field(
        default_factory=list, repr=False, compare=False
    )

    def record_fwd(
        self,
        time_ms: float,
        mem_delta_bytes: int,
        reserved_delta_bytes: int,
        peak_after_bytes: int,
        peak_rise_bytes: int,
    ) -> None:
        self.fwd_time_ms.append(time_ms)
        self.fwd_mem_allocated_delta_bytes.append(mem_delta_bytes)
        self.fwd_mem_reserved_delta_bytes.append(reserved_delta_bytes)
        self.fwd_mem_peak_after_bytes.append(peak_after_bytes)
        self.fwd_mem_peak_rise_bytes.append(peak_rise_bytes)
        self.num_samples += 1

    def reset(self) -> None:
        self.fwd_time_ms.clear()
        self.fwd_mem_allocated_delta_bytes.clear()
        self.fwd_mem_reserved_delta_bytes.clear()
        self.fwd_mem_peak_after_bytes.clear()
        self.fwd_mem_peak_rise_bytes.clear()
        self.num_samples = 0


class PerLayerProfiler:
    """Measured per-layer profiler using module hooks + deferred CUDA events."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._stats: Dict[int, PerLayerProfileStats] = {}
        self._handles: List["torch.utils.hooks.RemovableHandle"] = []
        self._layers: Dict[int, "torch.nn.Module"] = {}
        self._attached = False
        self._use_cuda = torch.cuda.is_available()
        # Per-device global peaks, measured (never reconstructed).
        self._global_peak_bytes: List[int] = []  # allocated peak
        self._global_reserved_peak_bytes: List[int] = []  # reserved peak (OOM ceiling)

    # ---- layer registration (construction time, no hooks) ---------------

    def register_layer(self, layer: "torch.nn.Module", layer_idx: int) -> None:
        """Record a layer and its MoE tag without installing hooks.

        Called once at block construction. Hooks are installed lazily by
        :meth:`start_step` only on steps that will be logged, so non-logged
        steps carry zero forward overhead. is_moe_layer is fixed at
        construction, so it is read here once, not per forward.
        """
        if not self.enabled:
            return
        is_moe = bool(getattr(layer, "is_moe_layer", False))
        self._stats[layer_idx] = PerLayerProfileStats(layer_idx=layer_idx, is_moe_layer=is_moe)
        self._layers[layer_idx] = layer

    # ---- hook attach / detach (per logged step) -------------------------

    def _attach_hooks(self) -> None:
        if self._attached or not self.enabled:
            return
        for layer_idx, layer in self._layers.items():
            self._handles.append(
                layer.register_forward_pre_hook(self._make_fwd_pre_hook(layer_idx))
            )
            self._handles.append(layer.register_forward_hook(self._make_fwd_post_hook(layer_idx)))
        self._attached = True

    def _detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._attached = False

    # ---- step boundary (global peak anchor) -----------------------------

    def start_step(self, should_profile: bool = True) -> None:
        """Call at the start of a step, before the first layer runs.

        When ``should_profile`` is True (i.e. this iteration will be logged),
        install hooks if not already installed and reset the CUDA peak tracker.
        When False, ensure hooks are removed so the step runs at zero overhead.
        Hooks must never call reset_peak_memory_stats(); resetting here (once
        per step) keeps the per-device global-peak measurement intact.
        """
        if not self.enabled:
            return
        if should_profile:
            self._attach_hooks()
            if self._use_cuda:
                torch.cuda.reset_peak_memory_stats()
        else:
            # Leaving a previously-logged step: drop hooks so subsequent
            # non-logged steps are overhead-free.
            if self._attached:
                self._detach_hooks()

    def end_step(self) -> None:
        """Call after the step's forward and backward complete.

        The global peak is read here so it includes the backward high-water
        mark (where real OOMs usually occur). No-op if hooks are not attached
        (this step is not being profiled).
        """
        if not self.enabled or not self._use_cuda or not self._attached:
            return
        stats = torch.cuda.memory_stats()
        self._global_peak_bytes.append(stats["allocated_bytes.all.peak"])
        self._global_reserved_peak_bytes.append(stats["reserved_bytes.all.peak"])

    @property
    def global_peak_bytes(self) -> List[int]:
        return self._global_peak_bytes

    @property
    def global_reserved_peak_bytes(self) -> List[int]:
        return self._global_reserved_peak_bytes

    @property
    def attached(self) -> bool:
        return self._attached

    # ---- helpers --------------------------------------------------------

    def _new_event(self) -> Optional["torch.cuda.Event"]:
        return torch.cuda.Event(enable_timing=True) if self._use_cuda else None

    def _mem_snapshot(self) -> _MemSnapshot:
        """One consistent allocator snapshot via torch.cuda.memory_stats().

        Replaces the separate memory_allocated()/max_memory_allocated() reads so
        every figure comes from the same snapshot and we additionally capture
        reserved bytes. Called only on logged steps (hooks are detached
        otherwise), so the memory_stats() dict construction is off the hot path.
        """
        if not self._use_cuda:
            return _MemSnapshot(0, 0, 0, 0)
        s = torch.cuda.memory_stats()
        return _MemSnapshot(
            s["allocated_bytes.all.current"],
            s["reserved_bytes.all.current"],
            s["allocated_bytes.all.peak"],
            s["reserved_bytes.all.peak"],
        )

    # ---- forward hooks --------------------------------------------------

    def _make_fwd_pre_hook(self, layer_idx: int):
        stats = self._stats[layer_idx]

        def hook(module, args):
            start = self._new_event()
            if start is not None:
                start.record()
            snap_before = self._mem_snapshot()
            # Park entry state in the last pending slot; completed by post hook.
            stats._pending_fwd.append((start, None, snap_before, snap_before))

        return hook

    def _make_fwd_post_hook(self, layer_idx: int):
        stats = self._stats[layer_idx]

        def hook(module, args, output):
            if not stats._pending_fwd:
                return
            end = self._new_event()
            if end is not None:
                end.record()
            snap_after = self._mem_snapshot()
            start, _, snap_before, _ = stats._pending_fwd[-1]
            stats._pending_fwd[-1] = (start, end, snap_before, snap_after)

        return hook

    # ---- deferred resolution -------------------------------------------

    def flush(self) -> None:
        """Resolve all pending CUDA events into recorded samples.

        Call only at log time. This is the single amortized synchronize point;
        the training hot path never blocks.
        """
        if not self.enabled:
            return
        if self._use_cuda:
            torch.cuda.synchronize()

        for stats in self._stats.values():
            for start, end, snap_before, snap_after in stats._pending_fwd:
                fwd_ms = start.elapsed_time(end) if start is not None and end is not None else 0.0
                mem_delta = snap_after.allocated - snap_before.allocated
                reserved_delta = snap_after.reserved - snap_before.reserved
                peak_rise = max(0, snap_after.allocated_peak - snap_before.allocated_peak)
                stats.record_fwd(
                    fwd_ms, mem_delta, reserved_delta, snap_after.allocated_peak, peak_rise
                )
            stats._pending_fwd.clear()

    # ---- access ---------------------------------------------------------

    def stats(self) -> Dict[int, PerLayerProfileStats]:
        return self._stats

    def reset(self) -> None:
        for s in self._stats.values():
            s.reset()
        self._global_peak_bytes.clear()
        self._global_reserved_peak_bytes.clear()


# ---------------------------------------------------------------------------
# Output layer
# ---------------------------------------------------------------------------


def per_layer_profiling_start_step(model, should_profile):
    """Start per-layer profiling for this step (call before forward).

    ``model`` is the list of model chunks. Single chunk only; interleaved
    virtual pipeline (len(model) > 1) is not yet supported and is skipped.
    """
    if len(model) != 1:
        return
    _model = unwrap_model(model[0])
    _dec = getattr(_model, "decoder", None)
    plp = getattr(_dec, "per_layer_profiler", None) if _dec is not None else None
    if plp is not None:
        plp.start_step(should_profile=should_profile)


def per_layer_profiling_end_step(model):
    """End per-layer profiling for this step (call after backward)."""
    if len(model) != 1:
        return
    _model = unwrap_model(model[0])
    _dec = getattr(_model, "decoder", None)
    plp = getattr(_dec, "per_layer_profiler", None) if _dec is not None else None
    if plp is not None:
        plp.end_step()


def _agg(samples: List[float]) -> Tuple[float, float]:
    """Return (mean, max) of a sample list; (0.0, 0.0) if empty."""
    if not samples:
        return 0.0, 0.0
    return sum(samples) / len(samples), max(samples)


def summarize_stats(profiler: "PerLayerProfiler", layer_offset: int = 0) -> Dict[str, object]:
    """Aggregate raw per-layer samples into mean/max, keyed by GLOBAL layer idx.

    Pure data (no I/O, no distributed calls), so it is unit-testable on CPU.

    ``layer_offset`` maps this rank's local layer indices to global ones under
    pipeline parallelism (pass the result of get_transformer_layer_offset() at
    the call site; 0 when PP is not used or in tests).

    Returned structure::

        {
          "global_peak_bytes": {"mean": float, "max": float},
          "layers": {
             <global_idx>: {
                "is_moe": bool,
                "num_samples": int,
                "fwd_time_ms":        {"mean": float, "max": float},
                "mem_delta_bytes":    {"mean": float, "max": float},
                "mem_delta_bytes":    {"mean": float, "max": float},
                "mem_reserved_delta_bytes":{"mean": float, "max": float},
                "mem_peak_after_bytes":{"mean": float, "max": float},
                "mem_peak_rise_bytes": {"mean": float, "max": float},
             }, ...
          },
        }
    """
    g_mean, g_max = _agg([float(x) for x in profiler.global_peak_bytes])
    gr_mean, gr_max = _agg([float(x) for x in profiler.global_reserved_peak_bytes])
    out: Dict[str, object] = {
        "global_peak_bytes": {"mean": g_mean, "max": g_max},
        "global_reserved_peak_bytes": {"mean": gr_mean, "max": gr_max},
        "layers": {},
    }
    layers: Dict[int, Dict[str, object]] = out["layers"]  # type: ignore[assignment]

    for local_idx, s in sorted(profiler.stats().items()):
        gidx = local_idx + layer_offset
        fwd_mean, fwd_max = _agg(s.fwd_time_ms)
        d_mean, d_max = _agg([float(x) for x in s.fwd_mem_allocated_delta_bytes])
        rd_mean, rd_max = _agg([float(x) for x in s.fwd_mem_reserved_delta_bytes])
        pa_mean, pa_max = _agg([float(x) for x in s.fwd_mem_peak_after_bytes])
        pr_mean, pr_max = _agg([float(x) for x in s.fwd_mem_peak_rise_bytes])
        layers[gidx] = {
            "is_moe": s.is_moe_layer,
            "num_samples": s.num_samples,
            "fwd_time_ms": {"mean": fwd_mean, "max": fwd_max},
            "mem_delta_bytes": {"mean": d_mean, "max": d_max},
            "mem_reserved_delta_bytes": {"mean": rd_mean, "max": rd_max},
            "mem_peak_after_bytes": {"mean": pa_mean, "max": pa_max},
            "mem_peak_rise_bytes": {"mean": pr_mean, "max": pr_max},
        }
    return out


def _mib(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


def log_per_layer_resource_usage(
    profiler: "PerLayerProfiler", layer_offset: int = 0, is_log_rank: bool = True
) -> Optional[Dict[str, object]]:
    """Produce a human-readable per-layer table and a structured summary.

    The summary dict is always returned (useful for TensorBoard/W&B and tests).
    The text table is printed only when ``is_log_rank`` is True; deciding rank
    membership is the caller's job (it needs distributed state), keeping this
    function free of distributed calls and unit-testable.

    Memory columns are per-rank / per-device; under TP they are the local
    shard. ``delta`` approximates retained activation and collapses under
    recompute -- read ``peak_after`` for the OOM-relevant figure.
    """
    summary = summarize_stats(profiler, layer_offset=layer_offset)
    if not is_log_rank:
        return summary

    layers: Dict[int, Dict[str, object]] = summary["layers"]  # type: ignore[assignment]
    gpeak: Dict[str, float] = summary["global_peak_bytes"]  # type: ignore[assignment]
    grpeak: Dict[str, float] = summary["global_reserved_peak_bytes"]  # type: ignore[assignment]

    header = (
        f"{'layer':>6} {'type':>5} {'n':>4} "
        f"{'fwd_ms(mean/max)':>20} "
        f"{'delta_MiB(mean/max)':>22} {'resv_d_MiB(mean/max)':>22} "
        f"{'peak_MiB(mean/max)':>22} {'rise_MiB(mean/max)':>22}"
    )
    lines = ["[per-layer resource usage] (per-rank; TP -> local shard)", header]
    for gidx in sorted(layers):
        r = layers[gidx]
        ft = r["fwd_time_ms"]
        dl = r["mem_delta_bytes"]
        rd = r["mem_reserved_delta_bytes"]
        pk = r["mem_peak_after_bytes"]
        rs = r["mem_peak_rise_bytes"]
        lines.append(
            f"{gidx:>6} {'MoE' if r['is_moe'] else 'dense':>5} {r['num_samples']:>4} "
            f"{ft['mean']:>9.3f}/{ft['max']:>9.3f} "
            f"{_mib(dl['mean']):>10.1f}/{_mib(dl['max']):>10.1f} "
            f"{_mib(rd['mean']):>10.1f}/{_mib(rd['max']):>10.1f} "
            f"{_mib(pk['mean']):>10.1f}/{_mib(pk['max']):>10.1f} "
            f"{_mib(rs['mean']):>10.1f}/{_mib(rs['max']):>10.1f}"
        )
    lines.append(
        f"[global per-device peak] "
        f"allocated mean={_mib(gpeak['mean']):.1f}/max={_mib(gpeak['max']):.1f} MiB  "
        f"reserved mean={_mib(grpeak['mean']):.1f}/max={_mib(grpeak['max']):.1f} MiB "
        f"(reserved = allocator pool incl. fragmentation; the OOM ceiling)"
    )
    print("\n".join(lines))
    return summary
