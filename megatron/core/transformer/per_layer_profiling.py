# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-layer measured profiling for transformer layers.

Recorded per ``TransformerLayer`` on logged steps only (zero overhead
otherwise) via ``torch.cuda.Event`` / ``torch.cuda.memory_stats()`` on the hot
path, but resolved lazily in :meth:`PerLayerProfiler.flush` at log time so the
training loop never force-synchronizes per layer:

* wall time -- forward from module pre/post hooks; backward from a viewless
  autograd marker (:class:`_LayerBoundaryMarker`) at each layer output and the
  block input, paired per layer in :meth:`PerLayerProfiler._on_bwd_marker`
  (avoids ``register_full_backward_hook``, which breaks PP output dealloc);
* memory (fwd and bwd) -- allocated delta, reserved delta, and a running
  allocated peak / peak-rise, all from one ``memory_stats()`` snapshot per
  boundary. The peak is the global monotonic ``allocated_bytes.all.peak``
  attributed to whichever layer lifted it, never a per-layer reset;
* a MoE-vs-dense tag, read once at attach time from ``is_moe_layer``.

Per-device global peaks (allocated and reserved) are measured once per interval
in :meth:`start_step`/:meth:`end_step`; the reserved peak is the allocator's
total pool (incl. fragmentation and comm buffers) -- the OOM-relevant ceiling.

Forward and backward are NOT symmetric -- read the numbers accordingly:

* Time: forward hooks *wrap* one layer, so it is that layer's forward wall time.
  Backward has no single-layer boundary, so it is the interval between adjacent
  markers -- a backward-*phase* wall time that also absorbs GPU idle (inflates
  and flattens it in launch-bound regimes), PP comm, and, under
  ``recompute_granularity='full'``, the layer's recomputed forward. Do NOT read
  ``bwd_ms`` as pure per-layer dgrad/wgrad. A per-``grad_fn`` timer is a follow-up.
* Memory is unified: both phases use the same boundary-snapshot difference.
  peak/peak-rise mean the same thing (global high-water attribution); delta
  differs only in sign by physics (fwd retains activations, bwd frees them).

Scope (all per-rank / per-device):

* TP: per-layer memory is the local ``1/TP`` shard; forward time includes the
  blocking TP all-reduce. PP: only the first stage's layers are emitted (logging
  gated to global rank 0; full cross-stage coverage is a follow-up). EP: MoE
  token counts vary per step, so those figures fluctuate by design.
* Allocated delta collapses under recompute / CPU offload (activations
  discarded) -- read the peak instead. Backward has no per-layer peak reset
  (a dedicated one is a follow-up).
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

# Same shape as _PendingFwd, but the two events bracket ONE layer's backward,
# paired by the profiler from the two adjacent boundary markers that fire around
# that layer (see PerLayerProfiler._on_bwd_marker).
_PendingBwd = _PendingFwd


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

    The ``bwd_*`` fields mirror the ``fwd_*`` ones for the backward phase, with
    two differences: (1) ``bwd_mem_allocated_delta_bytes`` is usually negative
    (backward frees activations while allocating grads); (2) backward samples do
    NOT bump ``num_samples`` (that counts forward passes) -- use
    ``len(bwd_time_ms)`` for the backward count. ``bwd_mem_peak_after_bytes`` /
    ``bwd_mem_peak_rise_bytes`` reuse the same monotonic global peak as forward
    (no per-layer reset), so a rise > 0 marks the backward layer that lifted the
    per-device high-water mark -- the figure that decides real OOMs under PP.
    """

    layer_idx: int
    is_moe_layer: bool = False

    fwd_time_ms: List[float] = dataclasses.field(default_factory=list)
    fwd_mem_allocated_delta_bytes: List[int] = dataclasses.field(default_factory=list)
    fwd_mem_reserved_delta_bytes: List[int] = dataclasses.field(default_factory=list)
    fwd_mem_peak_after_bytes: List[int] = dataclasses.field(default_factory=list)
    fwd_mem_peak_rise_bytes: List[int] = dataclasses.field(default_factory=list)

    bwd_time_ms: List[float] = dataclasses.field(default_factory=list)
    bwd_mem_allocated_delta_bytes: List[int] = dataclasses.field(default_factory=list)
    bwd_mem_reserved_delta_bytes: List[int] = dataclasses.field(default_factory=list)
    bwd_mem_peak_after_bytes: List[int] = dataclasses.field(default_factory=list)
    bwd_mem_peak_rise_bytes: List[int] = dataclasses.field(default_factory=list)

    num_samples: int = 0

    # Pending, unresolved CUDA events. Lists (not single slots) so that
    # multiple micro-batches per step under PP accumulate instead of
    # overwriting. Resolved and cleared in PerLayerProfiler.flush().
    _pending_fwd: List[_PendingFwd] = dataclasses.field(
        default_factory=list, repr=False, compare=False
    )
    _pending_bwd: List[_PendingBwd] = dataclasses.field(
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

    def record_bwd(
        self,
        time_ms: float,
        mem_delta_bytes: int,
        reserved_delta_bytes: int,
        peak_after_bytes: int,
        peak_rise_bytes: int,
    ) -> None:
        self.bwd_time_ms.append(time_ms)
        self.bwd_mem_allocated_delta_bytes.append(mem_delta_bytes)
        self.bwd_mem_reserved_delta_bytes.append(reserved_delta_bytes)
        self.bwd_mem_peak_after_bytes.append(peak_after_bytes)
        self.bwd_mem_peak_rise_bytes.append(peak_rise_bytes)

    def reset(self) -> None:
        self.fwd_time_ms.clear()
        self.fwd_mem_allocated_delta_bytes.clear()
        self.fwd_mem_reserved_delta_bytes.clear()
        self.fwd_mem_peak_after_bytes.clear()
        self.fwd_mem_peak_rise_bytes.clear()
        self.bwd_time_ms.clear()
        self.bwd_mem_allocated_delta_bytes.clear()
        self.bwd_mem_reserved_delta_bytes.clear()
        self.bwd_mem_peak_after_bytes.clear()
        self.bwd_mem_peak_rise_bytes.clear()
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
        # Transient state for pairing adjacent backward boundary markers into
        # per-layer intervals. (layer_idx, event, snapshot) of the most recent
        # marker whose backward fired. Reset each logged step and after flush.
        self._bwd_prev: Optional[Tuple[int, "torch.cuda.Event", _MemSnapshot]] = None

    # ---- layer registration (construction time, no hooks) ---------------

    def register_layer(self, layer: "torch.nn.Module", layer_idx: int) -> None:
        """Record a layer and its MoE tag without installing hooks.

        Called once at block construction. Hooks are installed lazily by
        :meth:`start_step` only on steps that will be logged, so non-logged
        steps carry zero forward overhead. is_moe_layer is fixed at
        construction, so it is read here once, not per forward.

        Also stashes back-references on the layer (``_per_layer_profiler`` and
        ``_per_layer_profiler_layer_idx``) so ``TransformerLayer.forward`` can
        insert the backward boundary marker without a second global registry.
        """
        if not self.enabled:
            return
        is_moe = bool(getattr(layer, "is_moe_layer", False))
        self._stats[layer_idx] = PerLayerProfileStats(layer_idx=layer_idx, is_moe_layer=is_moe)
        self._layers[layer_idx] = layer
        layer._per_layer_profiler = self
        layer._per_layer_profiler_layer_idx = layer_idx

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
        install hooks if not already installed, clear the backward marker-pairing
        state, and reset the CUDA peak tracker. When False, ensure hooks are
        removed so the step runs at zero overhead. Hooks must never call
        reset_peak_memory_stats(); resetting here (once per step) keeps the
        per-device global-peak measurement intact.
        """
        if not self.enabled:
            return
        if should_profile:
            self._attach_hooks()
            self._bwd_prev = None
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

    def _on_bwd_marker(self, layer_idx: int, evt: "torch.cuda.Event", snap: _MemSnapshot) -> None:
        """Pair this backward boundary marker with the previous one.

        Markers fire in strictly *decreasing* local layer order within one
        micro-batch's backward. Two adjacent markers bracket exactly one layer's
        backward: the interval from layer ``p``'s output marker to layer
        ``p-1``'s output marker is layer ``p``'s backward span. A jump *up* in
        layer_idx means a new backward pass began (next micro-batch under PP), so
        we start a fresh pairing instead of spanning across the boundary.
        """
        prev = self._bwd_prev
        if prev is not None:
            p_idx, p_evt, p_snap = prev
            if layer_idx < p_idx:
                # Close layer p_idx: p_evt (its output marker, earlier) ->
                # evt (deeper marker, later) == layer p_idx's backward.
                self._stats[p_idx]._pending_bwd.append((p_evt, evt, p_snap, snap))
            # else: layer_idx >= p_idx -> new pass; do not pair.
        self._bwd_prev = (layer_idx, evt, snap)

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
        """Resolve all pending forward and backward CUDA events into samples.

        Call only at log time. This is the single amortized synchronize point;
        the training hot path never blocks. Both the forward pending list
        (pre/post hook pairs) and the backward pending list (adjacent-marker
        pairs from :meth:`_on_bwd_marker`) are resolved and cleared here.
        """
        if not self.enabled:
            return
        if self._use_cuda:
            torch.cuda.synchronize()

        for stats in self._stats.values():
            # Forward pass
            for start, end, snap_before, snap_after in stats._pending_fwd:
                fwd_ms = start.elapsed_time(end) if start is not None and end is not None else 0.0
                mem_delta = snap_after.allocated - snap_before.allocated
                reserved_delta = snap_after.reserved - snap_before.reserved
                peak_rise = max(0, snap_after.allocated_peak - snap_before.allocated_peak)
                stats.record_fwd(
                    fwd_ms, mem_delta, reserved_delta, snap_after.allocated_peak, peak_rise
                )
            stats._pending_fwd.clear()
            # Backward pass
            for start, end, snap_before, snap_after in stats._pending_bwd:
                bwd_ms = start.elapsed_time(end) if start is not None and end is not None else 0.0
                mem_delta = snap_after.allocated - snap_before.allocated
                reserved_delta = snap_after.reserved - snap_before.reserved
                peak_rise = max(0, snap_after.allocated_peak - snap_before.allocated_peak)
                stats.record_bwd(
                    bwd_ms, mem_delta, reserved_delta, snap_after.allocated_peak, peak_rise
                )
            stats._pending_bwd.clear()

        self._bwd_prev = None

    # ---- access ---------------------------------------------------------

    def stats(self) -> Dict[int, PerLayerProfileStats]:
        return self._stats

    def reset(self) -> None:
        for s in self._stats.values():
            s.reset()
        self._global_peak_bytes.clear()
        self._global_reserved_peak_bytes.clear()


class _LayerBoundaryMarker(torch.autograd.Function):
    """Identity in both directions; its backward marks one layer's grad boundary.

    Zero-copy viewless output (``_base is None``) so it is safe under pipeline
    parallelism's ``deallocate_output_tensor`` assert and under in-place fusion
    ops -- unlike ``return x`` (a view) or ``return x.clone()`` (a full copy).
    Forward records nothing; backward records a single CUDA event + memory
    snapshot and hands them to the profiler, which pairs adjacent markers into
    per-layer intervals (see PerLayerProfiler._on_bwd_marker).
    """

    @staticmethod
    def forward(ctx, x, layer_idx, profiler):
        ctx.layer_idx = layer_idx
        ctx.profiler = profiler
        # Viewless output: a fresh tensor sharing x's storage, so out._base is
        # None. Same construction as megatron.core.utils.make_viewless_tensor,
        # inlined deliberately: (1) it uses no private symbol; (2) the public
        # make_viewless_tensor() would short-circuit and return x as-is when
        # x._base is None, so it would NOT install this Function's backward --
        # which is the whole point of the marker.
        out = torch.empty((1,), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        out.data = x.data
        return out

    @staticmethod
    def backward(ctx, grad_output):
        profiler = ctx.profiler
        if profiler is not None and profiler._use_cuda:
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            profiler._on_bwd_marker(ctx.layer_idx, evt, profiler._mem_snapshot())
        return grad_output, None, None


def mark_layer_boundary(x, layer_idx, profiler):
    """Insert a per-layer backward boundary marker at tensor ``x``."""
    return _LayerBoundaryMarker.apply(x, layer_idx, profiler)


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

    Backward columns mirror the forward ones (see :class:`PerLayerProfileStats`
    for their semantics); ``num_samples`` counts forward passes, ``bwd_num_samples``
    counts backward passes.

    Returned structure::

        {
          "global_peak_bytes":          {"mean": float, "max": float},
          "global_reserved_peak_bytes": {"mean": float, "max": float},
          "layers": {
             <global_idx>: {
                "is_moe": bool,
                "num_samples": int,
                "fwd_time_ms":               {"mean": float, "max": float},
                "mem_delta_bytes":           {"mean": float, "max": float},
                "mem_reserved_delta_bytes":  {"mean": float, "max": float},
                "mem_peak_after_bytes":      {"mean": float, "max": float},
                "mem_peak_rise_bytes":       {"mean": float, "max": float},
                "bwd_num_samples": int,
                "bwd_time_ms":                  {"mean": float, "max": float},
                "bwd_mem_delta_bytes":          {"mean": float, "max": float},
                "bwd_mem_reserved_delta_bytes": {"mean": float, "max": float},
                "bwd_mem_peak_after_bytes":     {"mean": float, "max": float},
                "bwd_mem_peak_rise_bytes":      {"mean": float, "max": float},
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
        # Forward pass
        fwd_mean, fwd_max = _agg(s.fwd_time_ms)
        d_mean, d_max = _agg([float(x) for x in s.fwd_mem_allocated_delta_bytes])
        rd_mean, rd_max = _agg([float(x) for x in s.fwd_mem_reserved_delta_bytes])
        pa_mean, pa_max = _agg([float(x) for x in s.fwd_mem_peak_after_bytes])
        pr_mean, pr_max = _agg([float(x) for x in s.fwd_mem_peak_rise_bytes])
        # Backward pass
        bwd_mean, bwd_max = _agg(s.bwd_time_ms)
        bd_mean, bd_max = _agg([float(x) for x in s.bwd_mem_allocated_delta_bytes])
        brd_mean, brd_max = _agg([float(x) for x in s.bwd_mem_reserved_delta_bytes])
        bpa_mean, bpa_max = _agg([float(x) for x in s.bwd_mem_peak_after_bytes])
        bpr_mean, bpr_max = _agg([float(x) for x in s.bwd_mem_peak_rise_bytes])

        layers[gidx] = {
            "is_moe": s.is_moe_layer,
            "num_samples": s.num_samples,
            "fwd_time_ms": {"mean": fwd_mean, "max": fwd_max},
            "mem_delta_bytes": {"mean": d_mean, "max": d_max},
            "mem_reserved_delta_bytes": {"mean": rd_mean, "max": rd_max},
            "mem_peak_after_bytes": {"mean": pa_mean, "max": pa_max},
            "mem_peak_rise_bytes": {"mean": pr_mean, "max": pr_max},
            "bwd_num_samples": len(s.bwd_time_ms),
            "bwd_time_ms": {"mean": bwd_mean, "max": bwd_max},
            "bwd_mem_delta_bytes": {"mean": bd_mean, "max": bd_max},
            "bwd_mem_reserved_delta_bytes": {"mean": brd_mean, "max": brd_max},
            "bwd_mem_peak_after_bytes": {"mean": bpa_mean, "max": bpa_max},
            "bwd_mem_peak_rise_bytes": {"mean": bpr_mean, "max": bpr_max},
        }
    return out


def _mib(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


def log_per_layer_resource_usage(
    profiler: "PerLayerProfiler", layer_offset: int = 0, is_log_rank: bool = True
) -> Optional[Dict[str, object]]:
    """Produce a human-readable per-layer table and a structured summary.

    The summary dict is always returned (useful for TensorBoard/W&B and tests).
    The text output is printed only when ``is_log_rank`` is True; deciding rank
    membership is the caller's job (it needs distributed state), keeping this
    function free of distributed calls and unit-testable. It prints two stacked
    sub-tables -- ``-- forward --`` and ``-- backward --`` -- sharing the layer
    index, plus a fwd-vs-full-step peak line and the global per-device peak.

    Memory columns are per-rank / per-device; under TP they are the local
    shard. ``delta`` approximates retained activation and collapses under
    recompute -- read ``peak_after`` for the OOM-relevant figure. In the
    backward sub-table ``delta`` is usually negative (activations freed).
    """
    summary = summarize_stats(profiler, layer_offset=layer_offset)
    if not is_log_rank:
        return summary

    layers: Dict[int, Dict[str, object]] = summary["layers"]  # type: ignore[assignment]
    gpeak: Dict[str, float] = summary["global_peak_bytes"]  # type: ignore[assignment]
    grpeak: Dict[str, float] = summary["global_reserved_peak_bytes"]  # type: ignore[assignment]

    def _sub_header(time_label: str) -> str:
        return (
            f"{'layer':>6} {'type':>5} {'n':>4} "
            f"{time_label:>20} "
            f"{'delta_MiB(mean/max)':>22} {'resv_d_MiB(mean/max)':>22} "
            f"{'peak_MiB(mean/max)':>22} {'rise_MiB(mean/max)':>22}"
        )

    def _fmt_row(gidx: int, r: Dict[str, object], t_key, d_key, rd_key, pk_key, rs_key) -> str:
        t, d, rdv, pkv, rsv = r[t_key], r[d_key], r[rd_key], r[pk_key], r[rs_key]
        return (
            f"{gidx:>6} {'MoE' if r['is_moe'] else 'dense':>5} {r['num_samples']:>4} "
            f"{t['mean']:>9.3f}/{t['max']:>9.3f} "
            f"{_mib(d['mean']):>10.1f}/{_mib(d['max']):>10.1f} "
            f"{_mib(rdv['mean']):>10.1f}/{_mib(rdv['max']):>10.1f} "
            f"{_mib(pkv['mean']):>10.1f}/{_mib(pkv['max']):>10.1f} "
            f"{_mib(rsv['mean']):>10.1f}/{_mib(rsv['max']):>10.1f}"
        )

    lines = ["[per-layer resource usage] (per-rank; TP -> local shard)", "-- forward --"]
    lines.append(_sub_header("fwd_ms(mean/max)"))
    for gidx in sorted(layers):
        lines.append(
            _fmt_row(
                gidx,
                layers[gidx],
                "fwd_time_ms",
                "mem_delta_bytes",
                "mem_reserved_delta_bytes",
                "mem_peak_after_bytes",
                "mem_peak_rise_bytes",
            )
        )
    lines.append(
        "-- backward (delta<0 = activations freed; peak/rise share the global monotonic peak) --"
    )
    lines.append(_sub_header("bwd_ms(mean/max)"))
    for gidx in sorted(layers):
        lines.append(
            _fmt_row(
                gidx,
                layers[gidx],
                "bwd_time_ms",
                "bwd_mem_delta_bytes",
                "bwd_mem_reserved_delta_bytes",
                "bwd_mem_peak_after_bytes",
                "bwd_mem_peak_rise_bytes",
            )
        )

    fwd_phase_peak = max((r["mem_peak_after_bytes"]["max"] for r in layers.values()), default=0.0)
    bwd_extra = gpeak["max"] - fwd_phase_peak
    lines.append(
        f"[fwd-vs-full peak] forward-phase max={_mib(fwd_phase_peak):.1f} MiB  "
        f"full-step(incl. bwd) max={_mib(gpeak['max']):.1f} MiB  "
        f"=> backward pushed peak +{_mib(bwd_extra):.1f} MiB"
        + ("  (backward is the high-water mark)" if bwd_extra > 0 else "")
    )
    lines.append(
        f"[global per-device peak] "
        f"allocated mean={_mib(gpeak['mean']):.1f}/max={_mib(gpeak['max']):.1f} MiB  "
        f"reserved mean={_mib(grpeak['mean']):.1f}/max={_mib(grpeak['max']):.1f} MiB "
        f"(reserved = allocator pool incl. fragmentation; the OOM ceiling)"
    )
    print("\n".join(lines))
    return summary
