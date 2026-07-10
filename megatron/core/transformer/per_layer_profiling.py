# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-layer forward/backward timing and memory profiling for transformer layers.

WHAT
    On logged steps only (zero overhead otherwise), records per
    ``TransformerLayer``: GPU wall time via ``torch.cuda.Event`` and per-layer
    memory from the CUDA allocator trace
    (``torch.cuda.memory._record_memory_history``), plus the device-global peak
    via ``torch.cuda.memory_stats()``. CUDA events are resolved lazily in
    :meth:`PerLayerProfiler.flush` at log time, so the training hot path never
    force-synchronizes per layer.

HOW TO USE
    Enable ``log_per_layer_profiling`` in the config. The training loop calls
    ``per_layer_profiling_start_step`` / ``per_layer_profiling_end_step`` each
    step and ``log_per_layer_resource_usage`` at each log interval (per-rank
    table on rank 0), then ``per_layer_profiling_final_report`` once after the
    loop -- an all-ranks gather that prints one cross-rank table so ALL pipeline
    stages are visible and TP shards / DP replicas are reduced by max (the worst
    GPU). Per-interval figures are per-rank / per-device (under TP the local
    ``1/TP`` shard). If the allocator trace is unavailable it degrades to a
    timing-only report.

FIELD SEMANTICS
    Time (CUDA events, always on)
    * ``fwd_ms`` -- the layer's forward wall time (hooks wrap the layer).
    * ``bwd_ms`` -- interval between adjacent backward markers: a backward-*phase*
      wall time that also absorbs GPU idle, PP comm, and (under
      ``recompute_granularity='full'``) the recomputed forward. NOT pure
      dgrad/wgrad; a per-``grad_fn`` timer is a follow-up.

    Per-layer memory (from the allocator trace)
    * ``peak`` -- true per-layer window high-water.
    * ``retained`` -- alloc'd in the window, freed later (activation held to
      backward; scales with PP depth; what recompute removes). NOTE: under
      activation recompute, ``retained`` over-reports -- the checkpoint frees the
      activations just AFTER the forward window, so they count as "freed later"
      even though they do not survive to backward. ``peak`` correctly reflects
      the recompute saving (it barely climbs across layers); read ``peak``, not
      ``retained``, under recompute.
    * ``transient`` -- alloc'd and freed within the window (workspace).
    * ``largest`` -- biggest single allocation (contiguity / fragmentation).
    * ``comm`` -- bytes on communication streams (exact stream ids required;
      0 until wired, never guessed).
    Without the trace, per-layer memory is not reported (there is no coarse
    memory_stats fallback -- the deltas were misleading); only time + the global
    peak are shown.

    Device-global (memory_stats, cheap, always on)
    * ``global_peak_bytes`` / ``global_reserved_peak_bytes`` -- per-device
      allocated / reserved high-water (reserved = allocator pool incl.
      fragmentation = the OOM ceiling).
    * ``reserved_overhead_bytes`` = reserved - allocated (fragmentation), from the
      trace snapshot when available.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from megatron.core.utils import unwrap_model

# A pending timing sample awaiting flush(): the two CUDA events bracket one
# forward (module pre/post hooks) or one backward (two adjacent boundary markers,
# paired in PerLayerProfiler._on_bwd_boundary).
_PendingEvents = Tuple[Optional["torch.cuda.Event"], Optional["torch.cuda.Event"]]


@dataclasses.dataclass
class _LayerTiming:
    """Per-layer GPU wall-time samples (one entry per profiled forward/backward).

    Timing only -- per-layer memory comes from the allocator trace
    (:class:`_MemTraceCollector`); the device-global peak is held on
    :class:`PerLayerProfiler`. ``time_ms`` / ``_pending`` are keyed by phase
    ("fwd" / "bwd"). ``num_samples`` counts forward passes; the backward count is
    ``len(time_ms["bwd"])``. ``_pending`` holds unresolved CUDA-event pairs,
    resolved and cleared in :meth:`PerLayerProfiler.flush` (lists, so multiple
    micro-batches per step under PP accumulate rather than overwrite).
    """

    layer_idx: int
    is_moe_layer: bool = False
    num_samples: int = 0
    time_ms: Dict[str, List[float]] = dataclasses.field(
        default_factory=lambda: {"fwd": [], "bwd": []}
    )
    _pending: Dict[str, List[_PendingEvents]] = dataclasses.field(
        default_factory=lambda: {"fwd": [], "bwd": []}, repr=False, compare=False
    )

    def record(self, phase: str, time_ms: float) -> None:
        self.time_ms[phase].append(time_ms)
        if phase == "fwd":
            self.num_samples += 1

    def reset(self) -> None:
        for samples in self.time_ms.values():
            samples.clear()
        for pending in self._pending.values():
            pending.clear()
        self.num_samples = 0


class _MemTraceCollector:
    """Optional per-layer memory-trace collector (``_record_memory_history``).

    A separate collaborator from the CUDA-event timing path (it does NOT time --
    ``time_us`` in the trace is host-launch time, not GPU compute time). On
    sampled steps it enables the allocator trace, drops a tiny **sentinel**
    tensor at each layer boundary (forward pre/post hook, backward marker) so the
    boundaries can be located in the trace by address, then at
    :meth:`end_step` snapshots, slices the trace into per-layer windows
    (:func:`build_trace_windows`) and derives the six metrics
    (:func:`summarize_trace_step`).

    Everything CUDA/private-API is guarded: a schema change or any failure
    degrades to skipping this step's sample, never interrupting training.
    Sentinels are held alive until end_step so their addresses are not reused
    and stay locatable; they are a few hundred bytes each (negligible vs the
    MiB-scale figures) and, being unfreed within a window, show up as a tiny
    constant in ``retained_bytes``.
    """

    def __init__(
        self, enabled: bool = False, stacks: str = "all", comm_stream_ids: Optional[set] = None
    ):
        # stacks="all" is the config proven by the PoC on torch 2.8; the trace is
        # sliced by sentinel address (not by frames), so frames are unused here
        # and could later be dropped ("python") to trim overhead.
        self.enabled = enabled and torch.cuda.is_available()
        self._stacks = stacks
        # Exact communication stream ids for comm_bytes attribution, supplied by
        # the caller from the comm setup (NOT guessed from "non-default stream",
        # which both over- and under-counts). None -> comm_bytes stays 0 until
        # the exact ids are wired in; never a heuristic.
        self._comm_stream_ids = comm_stream_ids
        self._recording = False
        self._base = 0
        # (layer_idx, kind, tensor) in host order; kind in {"pre","post","bwd"}.
        self._sentinels: List[Tuple[int, str, "torch.Tensor"]] = []
        # One summarize_trace_step() dict per profiled step (this interval).
        self._summaries: List[Dict[str, object]] = []

    def start_step(self, should_profile: bool) -> None:
        """Enable recording + capture the anchor on profiled steps; else ensure off."""
        if not self.enabled:
            return
        if not should_profile:
            self._stop_recording()
            return
        self._sentinels = []
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks=self._stacks, max_entries=1_000_000
            )
            torch.cuda.synchronize()
            # Anchor: allocated bytes resident before the trace begins (params,
            # persistent state) -- the trace omits their alloc events.
            self._base = torch.cuda.memory_allocated()
            self._recording = True
        except Exception:
            self._recording = False

    def _sentinel(self, layer_idx: int, kind: str) -> None:
        if not self._recording:
            return
        try:
            self._sentinels.append(
                (
                    layer_idx,
                    kind,
                    torch.empty(64, dtype=torch.uint8, device="cuda"),
                )  # viewless tensor
            )
        except Exception:
            pass

    def mark(self, kind: str, layer_idx: int) -> None:
        """Drop a boundary sentinel; ``kind`` is one of "pre" / "post" / "bwd"
        (the vocabulary :func:`build_trace_windows` pairs into windows)."""
        self._sentinel(layer_idx, kind)

    def end_step(self) -> None:
        """Snapshot, slice into per-layer windows, derive metrics, stop recording.

        Local layer indices are kept as-is (global relabeling happens at the
        reporting layer, as with ``summarize_timing``'s ``layer_offset``).
        """
        if not self.enabled or not self._recording:
            return
        try:
            torch.cuda.synchronize()
            snap = torch.cuda.memory._snapshot()
            dev = torch.cuda.current_device()
            events = self._device_events(snap, dev)
            # Each sentinel is still alive, so the last 'alloc' of its address is
            # itself (no later reuse) -- match on the most recent alloc index.
            last_alloc: Dict[int, int] = {}
            for i, e in enumerate(events):
                if e.get("action") == "alloc":
                    last_alloc[e.get("addr")] = i
            fwd_b: List[Tuple[int, str, int]] = []
            bwd_b: List[Tuple[int, int]] = []
            for layer_idx, kind, tensor in self._sentinels:
                eidx = last_alloc.get(tensor.data_ptr())
                if eidx is None:
                    continue
                if kind in ("pre", "post"):
                    fwd_b.append((layer_idx, kind, eidx))
                else:
                    bwd_b.append((layer_idx, eidx))
            fwd_b.sort(key=lambda b: b[2])
            bwd_b.sort(key=lambda b: b[1])
            windows = build_trace_windows(fwd_b, bwd_b)
            summary = summarize_trace_step(
                events,
                windows,
                self._base,
                comm_stream_ids=self._comm_stream_ids,
                segments=snap.get("segments"),
            )
            self._summaries.append(summary)
        except Exception:
            pass
        finally:
            self._stop_recording()

    def _stop_recording(self) -> None:
        if self._recording:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
        self._recording = False
        self._sentinels = []

    @staticmethod
    def _device_events(snap: Dict[str, object], dev: int) -> List[dict]:
        traces = snap.get("device_traces", []) or []
        if dev < len(traces):
            return traces[dev]
        flat: List[dict] = []
        for t in traces:
            flat.extend(t)
        return flat

    def summaries(self) -> List[Dict[str, object]]:
        return self._summaries

    def interval_summary(self, layer_offset: int = 0) -> Dict[str, object]:
        """Aggregate this interval's profiled steps to L2 and clear the buffer.

        Returns the interval's :func:`summarize_trace` output
        (global-indexed). The running per-rank global and cross-rank reduction
        are handled at the training/reporting layer over the unified report, so
        both timing and memory flow through one accumulator.
        """
        l2 = summarize_trace(self._summaries, layer_offset=layer_offset)
        self._summaries = []
        return l2

    def reset(self) -> None:
        self._summaries = []


class PerLayerProfiler:
    """Measured per-layer profiler using module hooks + deferred CUDA events."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._stats: Dict[int, _LayerTiming] = {}
        self._handles: List["torch.utils.hooks.RemovableHandle"] = []
        self._layers: Dict[int, "torch.nn.Module"] = {}
        self._attached = False
        self._use_cuda = torch.cuda.is_available()
        # Allocator-trace collector: the per-layer MEMORY source (timing stays on
        # the CUDA-event path). Enabled whenever profiling is; it self-guards and
        # degrades to timing-only if the private trace API is unavailable.
        self._mem_trace = _MemTraceCollector(enabled=enabled)
        # Running per-rank global (all log intervals) of the unified report,
        # reduced across ranks at end of training for the cross-stage table.
        # Persists across reset() (which only clears the current interval).
        self._report_global = _MemTraceGlobalAccumulator()
        # Per-device global peaks, measured (never reconstructed).
        self._global_peak_bytes: List[int] = []  # allocated peak
        self._global_reserved_peak_bytes: List[int] = []  # reserved peak (OOM ceiling)
        # Transient state for pairing adjacent backward boundary markers into
        # per-layer intervals. (layer_idx, event, snapshot) of the most recent
        # marker whose backward fired. Reset each logged step and after flush.
        self._bwd_prev: Optional[Tuple[int, Optional["torch.cuda.Event"]]] = None

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
        self._stats[layer_idx] = _LayerTiming(layer_idx=layer_idx, is_moe_layer=is_moe)
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
        self._mem_trace.start_step(should_profile)

    def end_step(self) -> None:
        """Call after the step's forward and backward complete.

        The global peak is read here so it includes the backward high-water
        mark (where real OOMs usually occur). No-op if hooks are not attached
        (this step is not being profiled).
        """
        # Resolve the allocator-trace sample first (self-guards when not
        # recording); it takes its own snapshot before this step's teardown.
        self._mem_trace.end_step()
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

    def _on_bwd_boundary(self, layer_idx: int, evt: Optional["torch.cuda.Event"]) -> None:
        """Pair this backward boundary marker with the previous one.

        Markers fire in strictly *decreasing* local layer order within one
        micro-batch's backward. Two adjacent markers bracket exactly one layer's
        backward: the interval from layer ``p``'s output marker to layer
        ``p-1``'s output marker is layer ``p``'s backward span. A jump *up* in
        layer_idx means a new backward pass began (next micro-batch under PP), so
        we start a fresh pairing instead of spanning across the boundary.
        """
        self._mem_trace.mark("bwd", layer_idx)
        prev = self._bwd_prev
        if prev is not None:
            p_idx, p_evt = prev
            if layer_idx < p_idx:
                # Close layer p_idx: p_evt (its output marker, earlier) ->
                # evt (deeper marker, later) == layer p_idx's backward.
                self._stats[p_idx]._pending["bwd"].append((p_evt, evt))
            # else: layer_idx >= p_idx -> new pass; do not pair.
        self._bwd_prev = (layer_idx, evt)

    # ---- forward hooks --------------------------------------------------

    def _make_fwd_pre_hook(self, layer_idx: int):
        stats = self._stats[layer_idx]

        def hook(module, args):
            start = self._new_event()
            if start is not None:
                start.record()
            # Park entry event in the last pending slot; completed by post hook.
            stats._pending["fwd"].append((start, None))
            self._mem_trace.mark("pre", layer_idx)

        return hook

    def _make_fwd_post_hook(self, layer_idx: int):
        stats = self._stats[layer_idx]

        def hook(module, args, output):
            if not stats._pending["fwd"]:
                return
            end = self._new_event()
            if end is not None:
                end.record()
            start, _ = stats._pending["fwd"][-1]
            stats._pending["fwd"][-1] = (start, end)
            self._mem_trace.mark("post", layer_idx)

        return hook

    # ---- deferred resolution -------------------------------------------

    def flush(self) -> None:
        """Resolve all pending forward and backward CUDA events into samples.

        Call only at log time. This is the single amortized synchronize point;
        the training hot path never blocks. Both the forward pending list
        (pre/post hook pairs) and the backward pending list (adjacent-marker
        pairs from :meth:`_on_bwd_boundary`) are resolved and cleared here.
        """
        if not self.enabled:
            return
        if self._use_cuda:
            torch.cuda.synchronize()

        for stats in self._stats.values():
            for phase in ("fwd", "bwd"):
                for start, end in stats._pending[phase]:
                    ms = start.elapsed_time(end) if start is not None and end is not None else 0.0
                    stats.record(phase, ms)
                stats._pending[phase].clear()

        self._bwd_prev = None

    # ---- access ---------------------------------------------------------

    def stats(self) -> Dict[int, _LayerTiming]:
        return self._stats

    def reset(self) -> None:
        for s in self._stats.values():
            s.reset()
        self._global_peak_bytes.clear()
        self._global_reserved_peak_bytes.clear()
        self._mem_trace.reset()

    @property
    def mem_trace(self) -> "_MemTraceCollector":
        return self._mem_trace

    def fold_report(self, report: Dict[str, object]) -> None:
        """Accumulate one interval's unified report into this rank's running global."""
        self._report_global.add(report)

    def global_report(self) -> Dict[str, object]:
        """This rank's global (all intervals), for the end-of-run cross-rank merge."""
        return self._report_global.result()


class _LayerBoundaryMarker(torch.autograd.Function):
    """Identity in both directions; its backward marks one layer's grad boundary.

    Zero-copy viewless output (``_base is None``) so it is safe under pipeline
    parallelism's ``deallocate_output_tensor`` assert and under in-place fusion
    ops -- unlike ``return x`` (a view) or ``return x.clone()`` (a full copy).
    Forward records nothing; backward records a single CUDA event and hands it
    to the profiler, which pairs adjacent markers into per-layer intervals (see
    PerLayerProfiler._on_bwd_boundary).
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
            profiler._on_bwd_boundary(ctx.layer_idx, evt)
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


def per_layer_profiling_final_report(model, group=None):
    """End-of-run cross-rank table (call once after the training loop).

    Gathers every rank's running global summary over ``group`` (default: the
    whole world) and prints one table on global rank 0, so under PP the layers of
    ALL stages are visible (each rank only holds its own stage) and TP shards / DP
    replicas are reduced by max. Uses ``torch.distributed.all_gather_object`` with
    the explicit ``group``; falls back to the single-process view if distributed
    is not initialized. No-op unless profiling is enabled on a single chunk.
    """
    if len(model) != 1:
        return
    _model = unwrap_model(model[0])
    _dec = getattr(_model, "decoder", None)
    plp = getattr(_dec, "per_layer_profiler", None) if _dec is not None else None
    if plp is None or not plp.enabled:
        return
    local = {"rank": 0, "layers": plp.global_report()["layers"]}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local["rank"] = torch.distributed.get_rank(group)
        gathered: List[Optional[dict]] = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(gathered, local, group=group)
        merged = merge_across_ranks([g for g in gathered if g])
        is_log_rank = torch.distributed.get_rank() == 0
    else:
        merged = merge_across_ranks([local])
        is_log_rank = True
    log_global_ranks_report(merged, is_log_rank=is_log_rank)


def _agg(samples: List[float]) -> Tuple[float, float]:
    """Return (mean, max) of a sample list; (0.0, 0.0) if empty."""
    if not samples:
        return 0.0, 0.0
    return sum(samples) / len(samples), max(samples)


def summarize_timing(profiler: "PerLayerProfiler", layer_offset: int = 0) -> Dict[str, object]:
    """Aggregate raw per-layer samples into mean/max, keyed by GLOBAL layer idx.

    Pure data (no I/O, no distributed calls), so it is unit-testable on CPU.

    ``layer_offset`` maps this rank's local layer indices to global ones under
    pipeline parallelism (pass the result of get_transformer_layer_offset() at
    the call site; 0 when PP is not used or in tests).

    ``num_samples`` counts forward passes, ``bwd_num_samples`` backward passes.
    Per-layer MEMORY is not here -- it comes from the allocator trace
    (:func:`summarize_trace`); this path keeps only timing plus the cheap,
    always-on device-global peak.

    Returned structure::

        {
          "global_peak_bytes":          {"mean": float, "max": float},
          "global_reserved_peak_bytes": {"mean": float, "max": float},
          "layers": {
             <global_idx>: {
                "is_moe": bool,
                "num_samples": int,      "fwd_time_ms": {"mean": float, "max": float},
                "bwd_num_samples": int,  "bwd_time_ms": {"mean": float, "max": float},
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
        fwd_mean, fwd_max = _agg(s.time_ms["fwd"])
        bwd_mean, bwd_max = _agg(s.time_ms["bwd"])
        layers[local_idx + layer_offset] = {
            "is_moe": s.is_moe_layer,
            "num_samples": s.num_samples,
            "fwd_time_ms": {"mean": fwd_mean, "max": fwd_max},
            "bwd_num_samples": len(s.time_ms["bwd"]),
            "bwd_time_ms": {"mean": bwd_mean, "max": bwd_max},
        }
    return out


# ---------------------------------------------------------------------------
# Memory-trace post-processing (torch.cuda.memory._record_memory_history)
#
# Pure, CUDA-free functions that turn ONE device's allocator trace
# (``device_traces`` from ``torch.cuda.memory._snapshot()``) plus per-layer
# boundary windows into the six partitioning-relevant metrics from
# per_layer_mem_trace_design.md: peak, retained activation, transient workspace,
# largest single alloc, comm buffer (by stream), and reserved fragmentation
# overhead. Side-effect-free so they unit-test on CPU with synthetic event
# dicts; the CUDA-touching collection (enabling recording, sentinel boundary
# markers, taking the snapshot) is a separate layer.
#
# Every field is read via ``.get()`` because this parses a private torch API
# whose event schema evolves additively across versions (e.g. ``time_us`` and
# ``compile_context`` were added after the feature shipped).
# ---------------------------------------------------------------------------


class _TraceWindow(NamedTuple):
    """One layer/phase slice of the trace, as inclusive event-index bounds.

    Built by the collector from sentinel boundary markers matched by address
    (see the PoC); the pure summarizer below only needs the bounds.
    """

    layer_idx: int
    phase: str  # "fwd" or "bwd"
    lo: int
    hi: int


def _allocated_curve(
    events: List[dict], base: int, free_action: str = "free_requested"
) -> List[int]:
    """Anchored running ``allocated_bytes.all.current`` after each event.

    ``+size`` on ``alloc``, ``-size`` on ``free_action``. ``base`` is
    ``memory_allocated()`` captured at record-start; the trace omits allocations
    made before recording was enabled (params, warmup grads), so without the
    anchor the curve is low by exactly that resident baseline (PoC: anchoring
    made the reconstructed peak match ``max_memory_allocated()`` to 0 MiB).
    ``free_requested`` (not ``free_completed``) matches the allocator's current
    counter; ``free_completed`` lags by stream completion.
    """
    cur = base
    curve: List[int] = []
    for e in events:
        action = e.get("action")
        size = e.get("size", 0) or 0
        if action == "alloc":
            cur += size
        elif action == free_action:
            cur -= size
        curve.append(cur)
    return curve


def _addr_lifetimes(
    events: List[dict], free_action: str = "free_requested"
) -> Dict[int, Optional[int]]:
    """Map each ``alloc`` event index -> its ``free_action`` event index (or None).

    Matches every allocation to the next free of the SAME address, so address
    reuse (freed then re-allocated) starts a fresh lifetime rather than
    aliasing the old one. Used to split a window's allocations into retained
    (freed later, e.g. activations held for backward) vs transient (freed within
    the same window, e.g. kernel workspace).
    """
    open_by_addr: Dict[int, int] = {}  # addr -> alloc event index currently live
    free_of: Dict[int, Optional[int]] = {}  # alloc_idx -> free_idx (or None)
    for i, e in enumerate(events):
        action = e.get("action")
        addr = e.get("addr")
        if action == "alloc":
            open_by_addr[addr] = i
            free_of[i] = None
        elif action == free_action and addr in open_by_addr:
            free_of[open_by_addr.pop(addr)] = i
    return free_of


def _reserved_overhead(segments: Optional[List[dict]]) -> Dict[str, Optional[int]]:
    """Reserved-pool vs live-allocated gap from a ``_snapshot()`` segment list.

    ``reserved`` = total cudaMalloc'd pool (sum of segment sizes); ``allocated``
    = bytes in active blocks; ``overhead`` = reserved - allocated, i.e. pool
    held by the allocator but not backing a live tensor (fragmentation, inactive
    splits, rounding) -- the gap between the allocated figure and the real
    per-device footprint that must fit in HBM. Returns Nones when no segment
    list is supplied.
    """
    if segments is None:
        return {"reserved_bytes": None, "allocated_bytes": None, "reserved_overhead_bytes": None}
    reserved = 0
    allocated = 0
    for seg in segments:
        reserved += seg.get("total_size", 0) or 0
        for blk in seg.get("blocks", []) or []:
            state = blk.get("state")
            if state in ("active_allocated", "active_pending_free"):
                allocated += blk.get("size", 0) or 0
    return {
        "reserved_bytes": reserved,
        "allocated_bytes": allocated,
        "reserved_overhead_bytes": reserved - allocated,
    }


def summarize_trace_step(
    events: List[dict],
    windows: List[_TraceWindow],
    base_allocated: int,
    comm_stream_ids: Optional[set] = None,
    segments: Optional[List[dict]] = None,
) -> Dict[str, object]:
    """Derive per-layer memory metrics from one device's allocator trace.

    Pure (no CUDA, no torch, no distributed) -- ``events`` is one device's
    ``device_traces`` list, ``windows`` the per-layer/phase event-index slices,
    ``base_allocated`` the record-start anchor, ``comm_stream_ids`` the set of
    stream ids treated as communication (all-reduce / P2P / all-gather buffers;
    ``None`` -> comm_bytes reported as 0), ``segments`` an optional snapshot
    segment list for the reserved-overhead figure.

    A given (layer, phase) can have MULTIPLE windows in one call -- one per
    micro-batch under PP (the sentinels for a whole profiled step, across all
    micro-batches, are collected before this is called). Those windows are
    aggregated into a ``{"mean": float, "max": float}`` cell per metric (mirrors
    how :class:`_LayerTiming` keeps one sample per micro-batch and
    :func:`summarize_timing` reduces them) -- NOT overwritten, which would
    silently keep only the last micro-batch and, since 1F1B's last micro-batch is
    typically in cooldown with fewer layers in flight, would systematically
    UNDER-report peak.

    Per window (keyed layer -> phase), each cell aggregates, in bytes:

    * ``peak_bytes``          -- max of the anchored allocated curve in the window
                                 (the OOM-relevant per-layer high-water mark);
    * ``retained_bytes``      -- alloc'd in the window but freed later (activation
                                 held past the window, e.g. for backward); this is
                                 what multiplies with in-flight microbatches / PP
                                 depth and is what recompute removes;
    * ``transient_bytes``     -- alloc'd AND freed within the window (workspace);
                                 sets a peak floor recompute cannot lower;
    * ``largest_alloc_bytes`` -- biggest single allocation (contiguity / frag);
    * ``comm_bytes``          -- alloc'd on a comm stream (unaffected by recompute).

    Plus device-global ``reserved_bytes`` / ``allocated_bytes`` /
    ``reserved_overhead_bytes`` (see :func:`_reserved_overhead`), one figure for
    the whole step (not per window).
    """
    curve = _allocated_curve(events, base_allocated)
    free_of = _addr_lifetimes(events)
    n = len(events)

    # (layer_idx, phase, metric) -> one raw value per window (per micro-batch).
    raw: Dict[Tuple[int, str, str], List[float]] = {}
    for w in windows:
        lo = max(0, w.lo)
        hi = min(n - 1, w.hi)
        peak = max(curve[lo : hi + 1], default=base_allocated)
        retained = 0
        transient = 0
        largest = 0
        comm = 0
        for i in range(lo, hi + 1):
            e = events[i]
            if e.get("action") != "alloc":
                continue
            size = e.get("size", 0) or 0
            if size > largest:
                largest = size
            if comm_stream_ids is not None and e.get("stream") in comm_stream_ids:
                comm += size
            free_idx = free_of.get(i)
            if free_idx is not None and lo <= free_idx <= hi:
                transient += size
            else:
                retained += size
        for metric, value in (
            ("peak_bytes", peak),
            ("retained_bytes", retained),
            ("transient_bytes", transient),
            ("largest_alloc_bytes", largest),
            ("comm_bytes", comm),
        ):
            raw.setdefault((w.layer_idx, w.phase, metric), []).append(float(value))

    layers: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for (gidx, phase, metric), values in raw.items():
        mean, mx = _agg(values)
        layers.setdefault(gidx, {}).setdefault(phase, {})[metric] = {"mean": mean, "max": mx}

    out: Dict[str, object] = {"layers": layers}
    out.update(_reserved_overhead(segments))
    return out


def summarize_trace(summaries: List[Dict[str, object]], layer_offset: int = 0) -> Dict[str, object]:
    """Aggregate per-step mem-trace summaries into per-rank ``{mean, max}`` cells (L2).

    ``summaries`` is a list of :func:`summarize_trace_step` outputs (one per
    profiled step this rank saw); each per-(layer, phase, metric) value there is
    ALREADY a ``{mean, max}`` cell (aggregated across that step's micro-batches).
    This combines those cells across steps via **mean-of-means** and
    **max-of-maxes** -- the same reduction convention used elsewhere in this
    module for combining nested cells (e.g. cross-rank / cross-interval). Local
    layer indices are relabeled to global via ``layer_offset`` (same convention
    as :func:`summarize_timing`). The result feeds :func:`build_per_layer_report`
    as the precise-memory side. Global ``reserved_overhead_bytes`` (a step-level
    scalar, not a per-window cell) is aggregated directly via ``{mean, max}``.
    This is per-rank only; cross-rank reduction over this shape happens later, in
    :func:`merge_across_ranks` (called from :func:`per_layer_profiling_final_report`).
    """
    # gidx -> phase -> metric -> list of (this_step_mean, this_step_max).
    cells: Dict[int, Dict[str, Dict[str, List[Tuple[float, float]]]]] = {}
    overhead: List[float] = []
    for summ in summaries:
        layers = summ.get("layers", {}) or {}
        for lidx, phases in layers.items():
            gidx = lidx + layer_offset if isinstance(lidx, int) and lidx >= 0 else lidx
            for phase, metrics in (phases or {}).items():
                for metric, cell in (metrics or {}).items():
                    if not isinstance(cell, dict) or "max" not in cell:
                        continue
                    cells.setdefault(gidx, {}).setdefault(phase, {}).setdefault(metric, []).append(
                        (float(cell.get("mean", cell["max"])), float(cell["max"]))
                    )
        ov = summ.get("reserved_overhead_bytes")
        if ov is not None:
            overhead.append(float(ov))

    layers_out: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for gidx, phases in cells.items():
        for phase, metrics in phases.items():
            for metric, pairs in metrics.items():
                mean_of_means, _ = _agg([m for m, _ in pairs])
                _, max_of_maxes = _agg([mx for _, mx in pairs])
                layers_out.setdefault(gidx, {}).setdefault(phase, {})[metric] = {
                    "mean": mean_of_means,
                    "max": max_of_maxes,
                }
    out: Dict[str, object] = {"layers": layers_out}
    if overhead:
        o_mean, o_max = _agg(overhead)
        out["reserved_overhead_bytes"] = {"mean": o_mean, "max": o_max}
    return out


def build_trace_windows(
    fwd_boundaries: List[Tuple[int, str, int]], bwd_boundaries: List[Tuple[int, int]]
) -> List[_TraceWindow]:
    """Pair sentinel boundaries into per-layer trace windows (pure, CPU-testable).

    Both inputs are in trace/host order (event-index order).

    * ``fwd_boundaries`` -- ``(layer_idx, kind, event_idx)`` with kind ``"pre"``
      / ``"post"``. A layer's forward pre-hook and post-hook bracket exactly that
      layer (layers are sequential siblings, never nested), so each ``pre`` is
      matched FIFO to the next ``post`` of the same layer -> ``(L, "fwd", ...)``.
      FIFO (not one-shot) so multiple micro-batches per step accumulate.
    * ``bwd_boundaries`` -- ``(layer_idx, event_idx)``. Backward markers fire in
      strictly decreasing layer order; two adjacent markers bracket one layer's
      backward, and a jump *up* starts a fresh pass -- identical to the pairing
      in :meth:`PerLayerProfiler._on_bwd_boundary`. The block-input marker
      (``layer_idx == -1``) only ever closes layer 0; it never becomes a window.
    """
    windows: List[_TraceWindow] = []
    open_pre: Dict[int, List[int]] = {}
    for layer_idx, kind, eidx in fwd_boundaries:
        if kind == "pre":
            open_pre.setdefault(layer_idx, []).append(eidx)
        elif kind == "post":
            queue = open_pre.get(layer_idx)
            if queue:
                pre_idx = queue.pop(0)
                windows.append(_TraceWindow(layer_idx, "fwd", pre_idx, eidx))
    prev: Optional[Tuple[int, int]] = None
    for layer_idx, eidx in bwd_boundaries:
        if prev is not None:
            p_layer, p_eidx = prev
            if layer_idx < p_layer:
                windows.append(_TraceWindow(p_layer, "bwd", p_eidx, eidx))
        prev = (layer_idx, eidx)
    return windows


def _mib(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


def _fmt_cell(cell: Optional[Dict[str, float]], as_mib: bool = True) -> str:
    """Format a ``{mean,max}`` cell as ``mean/max`` (MiB if requested); n/a if None."""
    if not cell:
        return f"{'n/a':>9}/{'n/a':>9}"
    scale = _mib if as_mib else (lambda x: x)
    return f"{scale(cell.get('mean', 0.0)):>9.1f}/{scale(cell.get('max', 0.0)):>9.1f}"


def build_per_layer_report(
    timing_summary: Dict[str, object], mem_summary: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Merge timing + (optional) trace memory into one per-layer view.

    Pure (no CUDA / dist). ``time_ms`` always comes from the CUDA-event
    ``timing_summary`` (:func:`summarize_timing`). Per-layer memory comes from the
    allocator trace (``mem_summary`` = :func:`summarize_trace` output) when it is
    present ("trace" source): true window ``peak``, lifetime-based ``retained``,
    plus ``transient`` / ``largest`` / ``comm``. Without the trace ("timing-only"
    source) the memory cells are ``None`` -- there is no coarse per-layer
    fallback (the memory_stats deltas were dropped as misleading). The cheap,
    always-on device-global peak rides along from ``timing_summary``. Both inputs
    are expected already relabeled to GLOBAL layer indices.

    Returns ``{"source": "trace"|"timing-only", "layers": {gidx: {"is_moe",
    "fwd": {...}, "bwd": {...}}}, "global_peak_bytes", "global_reserved_peak_bytes",
    "reserved_overhead_bytes"?}`` where each phase dict has ``time_ms`` (always)
    and ``peak``/``retained``/``transient``/``largest``/``comm`` (``{mean,max}``
    cells when the trace is on, else ``None``).
    """
    t_layers: Dict[int, Dict[str, object]] = timing_summary.get("layers", {}) or {}
    m_layers: Dict[int, Dict[str, object]] = (mem_summary or {}).get("layers", {}) or {}
    has_trace = bool(m_layers)

    layers: Dict[int, Dict[str, object]] = {}
    for gidx in set(t_layers) | set(m_layers):
        t = t_layers.get(gidx, {})
        m = m_layers.get(gidx, {})
        entry: Dict[str, object] = {"is_moe": bool(t.get("is_moe", False))}
        for phase in ("fwd", "bwd"):
            time_cell = t.get("fwd_time_ms" if phase == "fwd" else "bwd_time_ms") or {
                "mean": 0.0,
                "max": 0.0,
            }
            mp = m.get(phase) if has_trace else None
            if mp:
                mem_cells = {
                    "peak": mp.get("peak_bytes"),
                    "retained": mp.get("retained_bytes"),
                    "transient": mp.get("transient_bytes"),
                    "largest": mp.get("largest_alloc_bytes"),
                    "comm": mp.get("comm_bytes"),
                }
            else:
                mem_cells = {k: None for k in ("peak", "retained", "transient", "largest", "comm")}
            entry[phase] = {"time_ms": time_cell, **mem_cells}
        layers[gidx] = entry

    out: Dict[str, object] = {
        "source": "trace" if has_trace else "timing-only",
        "layers": layers,
        "global_peak_bytes": timing_summary.get("global_peak_bytes"),
        "global_reserved_peak_bytes": timing_summary.get("global_reserved_peak_bytes"),
    }
    if mem_summary and mem_summary.get("reserved_overhead_bytes") is not None:
        out["reserved_overhead_bytes"] = mem_summary["reserved_overhead_bytes"]
    return out


def _render_per_layer_report(
    report: Dict[str, object], is_log_rank: bool = True, title: str = "[per-layer report]"
) -> Dict[str, object]:
    """Print the unified per-layer table from :func:`build_per_layer_report`.

    The table is printed only when ``is_log_rank`` (rank membership is the
    caller's decision, keeping this dist-free and unit-testable); the ``report``
    is always returned. A banner declares the memory source (PRECISE trace vs
    COARSE memory_stats fallback) and the meaning of each column so the two
    peak / retained definitions are never conflated.
    """
    if not is_log_rank:
        return report

    source = report.get("source", "timing-only")
    layers: Dict[int, Dict[str, object]] = report.get("layers", {})  # type: ignore[assignment]
    precise = source == "trace"

    lines = [title]
    if precise:
        lines.append(
            "memory source = PRECISE trace  "
            "(peak=window high-water; retained=activation held to backward; "
            "transient=freed-in-window workspace; largest=biggest single alloc; "
            "comm=comm-stream bytes)"
        )
    else:
        lines.append(
            "memory source = TIMING-ONLY (allocator trace off/unavailable; "
            "per-layer memory not shown -- see the global peak below)"
        )

    # Every cell is mean/max (stated once in the caption); each _fmt_cell renders
    # "%9.1f/%9.1f" = 19 chars, so headers are 19-wide to sit over the pair.
    lines.append("cell = mean/max   (time in ms, memory in MiB)")

    def _sub(phase: str, time_label: str) -> None:
        header = f"{'layer':>6} {'type':>5} {time_label:>19} {'peak':>19} {'retained':>19}"
        if precise:
            header += f" {'transient':>19} {'largest':>19} {'comm':>19}"
        lines.append(header)
        for gidx in sorted(layers):
            p = layers[gidx].get(phase, {})
            row = (
                f"{gidx:>6} {'MoE' if layers[gidx].get('is_moe') else 'dense':>5} "
                f"{_fmt_cell(p.get('time_ms'), as_mib=False)} "
                f"{_fmt_cell(p.get('peak'))} {_fmt_cell(p.get('retained'))}"
            )
            if precise:
                row += (
                    f" {_fmt_cell(p.get('transient'))} "
                    f"{_fmt_cell(p.get('largest'))} {_fmt_cell(p.get('comm'))}"
                )
            lines.append(row)

    lines.append("-- forward --")
    _sub("fwd", "fwd_ms")
    lines.append("-- backward --")
    _sub("bwd", "bwd_ms")

    ov = report.get("reserved_overhead_bytes")
    if isinstance(ov, dict):
        lines.append(
            f"[reserved fragmentation overhead] mean={_mib(ov.get('mean', 0.0)):.1f}/"
            f"max={_mib(ov.get('max', 0.0)):.1f} MiB (reserved - allocated)"
        )
    gpeak = report.get("global_peak_bytes")
    grpeak = report.get("global_reserved_peak_bytes")
    if isinstance(gpeak, dict) and isinstance(grpeak, dict):
        lines.append(
            f"[global per-device peak] "
            f"allocated mean={_mib(gpeak.get('mean', 0.0)):.1f}/"
            f"max={_mib(gpeak.get('max', 0.0)):.1f} MiB  "
            f"reserved mean={_mib(grpeak.get('mean', 0.0)):.1f}/"
            f"max={_mib(grpeak.get('max', 0.0)):.1f} MiB "
            f"(reserved = allocator pool incl. fragmentation; the OOM ceiling)"
        )
    print("\n".join(lines))
    return report


def log_per_layer_resource_usage(
    profiler: "PerLayerProfiler", layer_offset: int = 0, is_log_rank: bool = True
) -> Optional[Dict[str, object]]:
    """Single per-layer logging entry point -- the training loop calls only this.

    Builds the unified report from the timing summary plus, when the profiler's
    memory-trace collector is active, the precise allocator-trace memory; then
    renders it (banner + per-layer table + device-global peak). Free of
    distributed calls -- rank membership is the caller's decision via
    ``is_log_rank`` -- so it stays unit-testable. Returns the report dict
    (useful for TensorBoard / W&B / tests).
    """
    mt = getattr(profiler, "mem_trace", None)
    trace = (
        mt.interval_summary(layer_offset=layer_offset)
        if mt is not None and getattr(mt, "enabled", False)
        else None
    )
    report = build_per_layer_report(summarize_timing(profiler, layer_offset=layer_offset), trace)
    # Every rank folds its own report into the running global (GLOBAL layer ids);
    # only the log rank prints the per-interval table.
    profiler.fold_report(report)
    return _render_per_layer_report(report, is_log_rank=is_log_rank)


# ---------------------------------------------------------------------------
# Cross-rank aggregation (end-of-run): per-rank running global -> one table.
# Needed under PP, where each rank only sees its own stage's layers; the gather
# unions all stages and reduces TP shards / DP replicas by max (the worst GPU).
# ---------------------------------------------------------------------------


class _MemTraceGlobalAccumulator:
    """Fold each interval's unified report into a running per-rank global.

    Keeps only ``(sum_of_interval_means, n, max)`` per (global layer, phase,
    metric) -- not raw samples -- so cost is constant over a long run. ``add``
    ignores ``None`` cells (timing-only phases). ``result`` emits the same shape
    :func:`build_per_layer_report` uses for its ``layers`` block, so it feeds
    straight into :func:`merge_across_ranks`.
    """

    def __init__(self) -> None:
        self._cells: Dict[Tuple[int, str, str], List[float]] = {}  # -> [sum, n, max]
        self._is_moe: Dict[int, bool] = {}

    def add(self, report: Dict[str, object]) -> None:
        for gidx, entry in (report.get("layers", {}) or {}).items():
            self._is_moe[gidx] = bool(entry.get("is_moe", self._is_moe.get(gidx, False)))
            for phase in ("fwd", "bwd"):
                for metric, cell in (entry.get(phase, {}) or {}).items():
                    if not isinstance(cell, dict) or "max" not in cell:
                        continue
                    acc = self._cells.setdefault((gidx, phase, metric), [0.0, 0.0, 0.0])
                    acc[0] += float(cell.get("mean", cell["max"]))
                    acc[1] += 1.0
                    acc[2] = max(acc[2], float(cell["max"]))

    def result(self) -> Dict[str, object]:
        layers: Dict[int, Dict[str, object]] = {}
        for (gidx, phase, metric), (s, n, mx) in self._cells.items():
            entry = layers.setdefault(gidx, {"is_moe": self._is_moe.get(gidx, False)})
            entry.setdefault(phase, {})[metric] = {"mean": s / n if n else 0.0, "max": mx}
        return {"layers": layers}


def merge_across_ranks(per_rank: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    """Reduce per-rank global summaries into one per-(global layer, phase) view.

    Pure (no CUDA / dist) -- the ``all_gather_object`` is done by the caller and
    the gathered per-rank dicts (``_MemTraceGlobalAccumulator.result`` output,
    each tagged with its ``rank``) are passed in. OOM / bottlenecks are per-GPU,
    so across every rank that hosts a global layer (TP shards and DP replicas
    alike) the binding figure is the worst GPU: ``max`` = max over ranks of each
    rank's ``max`` (with ``argmax_rank``); ``mean`` = mean of per-rank means;
    ``spread`` = max - mean. NOT summed across TP; DP never co-resides. Under PP
    this simply unions the disjoint per-stage layer sets.

    ``spread`` is genuine cross-rank imbalance only when ``count`` (the number of
    ranks reporting this layer) is > 1 (TP shards / DP replicas). When
    ``count == 1`` (a layer only one rank ever hosts, e.g. plain PP with no
    replication), ``spread`` collapses to that one rank's OWN max-minus-mean
    across its logged intervals -- i.e. its across-time variability (a warmup
    spike), not a cross-rank difference. Always check ``count`` before reading
    ``spread`` as "imbalance"; :func:`log_global_ranks_report` prints it inline.
    """
    collected: Dict[Tuple[int, str, str], List[Tuple[float, float, int]]] = {}
    is_moe: Dict[int, bool] = {}
    for entry in per_rank:
        rank = int(entry.get("rank", -1))
        for gidx, layer in (entry.get("layers", {}) or {}).items():
            is_moe[gidx] = bool(layer.get("is_moe", is_moe.get(gidx, False)))
            for phase in ("fwd", "bwd"):
                for metric, cell in (layer.get(phase, {}) or {}).items():
                    if not isinstance(cell, dict) or "max" not in cell:
                        continue
                    collected.setdefault((gidx, phase, metric), []).append(
                        (float(cell.get("mean", cell["max"])), float(cell["max"]), rank)
                    )

    out: Dict[int, Dict[str, object]] = {}
    for (gidx, phase, metric), triples in collected.items():
        # triples are (mean, max, rank); reduce on the max field.
        _, max_val, argmax_rank = max(triples, key=lambda t: t[1])
        n = len(triples)
        mean_val = sum(m for m, _, _ in triples) / n if n else 0.0
        entry = out.setdefault(gidx, {"is_moe": is_moe.get(gidx, False)})
        entry.setdefault(phase, {})[metric] = {
            "max": max_val,
            "mean": mean_val,
            "spread": max_val - mean_val,
            "count": n,
            "argmax_rank": argmax_rank,
        }
    return out


def log_global_ranks_report(
    merged: Dict[int, Dict[str, object]],
    is_log_rank: bool = True,
    title: str = "[per-layer GLOBAL (all steps) + RANKS AGGREGATED]",
) -> Dict[int, Dict[str, object]]:
    """Print the end-of-run cross-rank table from :func:`merge_across_ranks`.

    Each cell shows ``max`` (worst GPU -- the OOM / bottleneck figure), ``spread``
    (max - mean), the ``argmax_rank`` that held it, and ``n`` (how many ranks
    reported this layer). Read ``spread`` as cross-rank imbalance ONLY when
    ``n > 1`` (TP shards / DP replicas); at ``n == 1`` (e.g. a layer only one PP
    stage hosts) it is that one rank's own across-time variability, not a
    cross-rank difference -- ``n`` is printed precisely so this isn't guessed.
    Time is ms, memory MiB. Printed only on the log rank; ``merged`` is returned.
    """
    if not is_log_rank:
        return merged

    def _c(cell: Optional[Dict[str, float]], as_mib: bool) -> str:
        if not cell:
            return f"{'n/a':>26}"
        scale = _mib if as_mib else (lambda x: x)
        return (
            f"{scale(cell.get('max', 0.0)):>8.1f}"
            f"(s{scale(cell.get('spread', 0.0)):>7.1f}"
            f" r{int(cell.get('argmax_rank', -1)):>2}"
            f" n{int(cell.get('count', 0)):>2})"
        )

    lines = [
        title,
        "cell = max (s=spread, r=argmax_rank, n=#ranks reporting this layer);"
        "  time ms, memory MiB",
        "spread is cross-rank imbalance only when n>1; at n==1 it is that rank's"
        " own across-time variability, not a cross-rank difference",
    ]
    metrics = [("time_ms", False), ("peak", True), ("retained", True)]
    for phase, label in (("fwd", "-- forward --"), ("bwd", "-- backward --")):
        lines.append(label)
        lines.append(f"{'layer':>6} {'type':>5} " + " ".join(f"{m:>26}" for m, _ in metrics))
        for gidx in sorted(merged):
            cells = merged[gidx].get(phase, {})
            row = f"{gidx:>6} {'MoE' if merged[gidx].get('is_moe') else 'dense':>5} " + " ".join(
                _c(cells.get(m), mib) for m, mib in metrics
            )
            lines.append(row)
    print("\n".join(lines))
    return merged
