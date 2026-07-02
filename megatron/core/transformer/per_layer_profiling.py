# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-layer measured resource profiling for transformer layers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch

from megatron.core.utils import get_pg_rank


_BYTES_PER_MB = 1024.0 * 1024.0


@dataclass
class PerLayerProfileStats:
    """Accumulated measured resource usage for one transformer layer."""

    layer_number: int
    forward_calls: int = 0
    backward_calls: int = 0
    activation_memory_bytes: int = 0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0

    def reset(self) -> None:
        """Reset counters after a log interval is emitted."""
        self.forward_calls = 0
        self.backward_calls = 0
        self.activation_memory_bytes = 0
        self.forward_time_ms = 0.0
        self.backward_time_ms = 0.0


class PerLayerProfiler:
    """Collect rank-local memory and wall-clock timing with module hooks."""

    def __init__(self, layers: Iterable[torch.nn.Module], layer_offset: int = 0) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._stats_by_module: dict[torch.nn.Module, PerLayerProfileStats] = {}
        self._forward_start: dict[torch.nn.Module, tuple[float, int]] = {}
        self._backward_start: dict[torch.nn.Module, float] = {}

        for local_idx, layer in enumerate(layers):
            layer_number = getattr(layer, "layer_number", layer_offset + local_idx + 1)
            self._stats_by_module[layer] = PerLayerProfileStats(layer_number=layer_number)
            self._handles.extend(
                [
                    layer.register_forward_pre_hook(self._forward_pre_hook),
                    layer.register_forward_hook(self._forward_hook),
                    layer.register_full_backward_pre_hook(self._backward_pre_hook),
                    layer.register_full_backward_hook(self._backward_hook),
                ]
            )

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self, reset: bool = False) -> list[PerLayerProfileStats]:
        """Return accumulated stats sorted by layer number."""
        stats = sorted(self._stats_by_module.values(), key=lambda item: item.layer_number)
        snapshot = [
            PerLayerProfileStats(
                layer_number=item.layer_number,
                forward_calls=item.forward_calls,
                backward_calls=item.backward_calls,
                activation_memory_bytes=item.activation_memory_bytes,
                forward_time_ms=item.forward_time_ms,
                backward_time_ms=item.backward_time_ms,
            )
            for item in stats
        ]
        if reset:
            for item in stats:
                item.reset()
        return snapshot

    def _forward_pre_hook(self, module: torch.nn.Module, _inputs) -> None:
        self._forward_start[module] = (self._time(), self._memory_allocated())

    def _forward_hook(self, module: torch.nn.Module, _inputs, _outputs) -> None:
        start = self._forward_start.pop(module, None)
        if start is None:
            return
        start_time, start_memory = start
        stats = self._stats_by_module[module]
        stats.forward_calls += 1
        stats.forward_time_ms += (self._time() - start_time) * 1000.0
        stats.activation_memory_bytes += max(0, self._memory_allocated() - start_memory)

    def _backward_pre_hook(self, module: torch.nn.Module, _grad_output) -> None:
        self._backward_start[module] = self._time()

    def _backward_hook(self, module: torch.nn.Module, _grad_input, _grad_output) -> None:
        start_time = self._backward_start.pop(module, None)
        if start_time is None:
            return
        stats = self._stats_by_module[module]
        stats.backward_calls += 1
        stats.backward_time_ms += (self._time() - start_time) * 1000.0

    @staticmethod
    def _time() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _memory_allocated() -> int:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated()


def log_per_layer_resource_usage(
    model,
    iteration: int,
    process_group=None,
    *,
    reset: bool = True,
) -> None:
    """Print per-layer resource usage from all profiled modules in ``model``."""
    if model is None:
        return

    model_chunks = model if isinstance(model, list) else [model]
    profilers = []
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            profiler = getattr(module, "per_layer_profiler", None)
            if profiler is not None:
                profilers.append(profiler)

    if not profilers:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    is_log_rank = get_pg_rank(process_group) == 0 if process_group is not None else rank == 0
    if not is_log_rank:
        for profiler in profilers:
            profiler.summary(reset=reset)
        return

    print(f"[Rank {rank}] per-layer resource usage after {iteration} iterations:", flush=True)
    for profiler in profilers:
        for stats in profiler.summary(reset=reset):
            if stats.forward_calls == 0 and stats.backward_calls == 0:
                continue
            activation_mb = stats.activation_memory_bytes / max(1, stats.forward_calls) / _BYTES_PER_MB
            forward_ms = stats.forward_time_ms / max(1, stats.forward_calls)
            backward_ms = stats.backward_time_ms / max(1, stats.backward_calls)
            print(
                f"[Rank {rank}] layer {stats.layer_number:4d}"
                f" | activation allocated (MB): {activation_mb:.2f}"
                f" | forward time (ms): {forward_ms:.3f}"
                f" | backward time (ms): {backward_ms:.3f}"
                f" | forward calls: {stats.forward_calls}"
                f" | backward calls: {stats.backward_calls}",
                flush=True,
            )
