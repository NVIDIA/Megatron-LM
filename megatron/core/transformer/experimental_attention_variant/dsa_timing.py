# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Lightweight CUDA-event timing for the DSA attention forward pass.

Ported/adapted from the min-memory branch's _DSATimingProfiler. On main the DSA
forward lives in a single complex DSAttention.forward, so rather than editing it
we attach forward hooks to every DSAttention module and time the whole call with
CUDA events. Zero changes to main's DSA kernels.

Usage (from the training entrypoint, after the model is built):

    from megatron.core.transformer.experimental_attention_variant.dsa_timing import (
        attach_dsa_forward_timing,
    )
    timer = attach_dsa_forward_timing(model, profile_rank=0)
    ...
    # once per iteration, after the forward (or after the step):
    timer.log(iteration)
"""

import time
from typing import Dict, List, Optional, Tuple

import torch


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class _DSATimingProfiler:
    """Accumulates per-DSA-layer forward-pass CUDA timings and reports them.

    Median over a window of iterations to smooth out warmup/jitter, mirroring the
    min-memory profiler's reporting.
    """

    _MEDIAN_WINDOW = 5

    def __init__(self, enabled: bool, profile_rank: int, label: str = "") -> None:
        rank = _distributed_rank()
        self.enabled = bool(enabled) and (profile_rank < 0 or rank == profile_rank)
        self.rank = rank
        self.label = label
        # (name, start_event, end_event) collected during a forward pass.
        self._records: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        # id(module) -> pending start event, for pre/post hook pairing.
        self._pending: Dict[int, torch.cuda.Event] = {}
        # per-iteration total-forward history for median reporting.
        self._history: List[Dict[str, float]] = []
        # Number of DSA layers; when this many post-hooks have fired, one full
        # forward pass is complete and we auto-log. Set by attach_dsa_forward_timing.
        self.n_layers = 0
        self._auto_iter = 0

    # ---- forward hooks ----
    def pre_hook(self, module, args):
        if not self.enabled or not torch.cuda.is_available():
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._pending[id(module)] = ev

    def post_hook(self, module, args, output):
        if not self.enabled or not torch.cuda.is_available():
            return
        start = self._pending.pop(id(module), None)
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        name = f"layer{getattr(module, 'layer_number', '?')}"
        self._records.append((name, start, end))
        # One full forward pass is complete once every DSA layer has fired.
        if self.n_layers and len(self._records) >= self.n_layers:
            self._auto_iter += 1
            self.log(self._auto_iter)

    # ---- reporting ----
    def log(self, iteration: int) -> None:
        if not self.enabled or not self._records:
            return
        torch.cuda.synchronize()
        totals: Dict[str, float] = {}
        order: List[str] = []
        for name, start, end in self._records:
            ms = start.elapsed_time(end)
            if name not in totals:
                totals[name] = 0.0
                order.append(name)
            totals[name] += ms
        self._records = []

        dsa_total = sum(totals.values())
        totals["dsa_forward_total"] = dsa_total
        self._history.append(totals)

        label = f" {self.label}" if self.label else ""
        per_layer = " ".join(f"{n}={totals[n]:.3f}ms" for n in order)
        print(
            f"[rank{self.rank}] DSA forward iter{iteration}{label}: "
            f"total={dsa_total:.3f}ms | {per_layer}",
            flush=True,
        )

        if len(self._history) >= self._MEDIAN_WINDOW:
            meds = sorted(h["dsa_forward_total"] for h in self._history)
            median_total = meds[len(meds) // 2]
            print(
                f"[rank{self.rank}] DSA forward MEDIAN over {self._MEDIAN_WINDOW} "
                f"iters{label}: dsa_forward_total={median_total:.3f}ms",
                flush=True,
            )
            self._history = []


def attach_dsa_forward_timing(model, profile_rank: int = 0, label: str = "") -> _DSATimingProfiler:
    """Register forward hooks on every DSAttention module in `model`.

    Returns the profiler; call `.log(iteration)` once per iteration to report.
    `model` may be a single module or a list of pipeline chunks.
    """
    # Import here to avoid a hard dependency at module import time.
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

    profiler = _DSATimingProfiler(enabled=True, profile_rank=profile_rank, label=label)
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n_attached = 0
    for chunk in chunks:
        if chunk is None:
            continue
        for submodule in chunk.modules():
            if isinstance(submodule, DSAttention):
                submodule.register_forward_pre_hook(profiler.pre_hook)
                submodule.register_forward_hook(profiler.post_hook)
                n_attached += 1
    profiler.n_layers = n_attached  # enables auto-log once per full forward pass
    if profiler.enabled:
        print(
            f"[rank{profiler.rank}] DSA timing attached to {n_attached} DSAttention layer(s)",
            flush=True,
        )
    return profiler
