# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in per-engine-step batch-size tracer for the dynamic inference engine.

Records, every engine step (optionally strided), how many requests are being
processed versus waiting in the engine, to a per-rank JSONL file for offline
analysis (see ``megatron.rl.rl_profiling``). This is a pure systems-level metric;
the module deliberately does not import any training/RL code. Callers inject extra
per-step metrics (e.g. the RL in-flight rollout count) via ``register_callback``.

Each engine (one per data-parallel replica) writes its own file tagged with its
global rank and data-parallel rank, so the aggregate and per-DP batch-size plots
can be reconstructed from the set of files.
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional


class InferenceStepTracer:
    """Buffers and writes per-step engine batch-size records to a per-rank JSONL file."""

    def __init__(
        self,
        output_dir: str,
        rank: int,
        dp_rank: int,
        stride: int = 1,
        run_id: Optional[str] = None,
        flush_every: int = 200,
    ):
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.dp_rank = dp_rank
        self.stride = max(1, stride)
        self.flush_every = max(1, flush_every)
        self.run_id = run_id or "run"
        self._callbacks: List[Callable[[], Dict]] = []
        self._file = None
        self._n_since_flush = 0
        self._last_t: Optional[float] = None

    def register_callback(self, fn: Callable[[], Dict]) -> None:
        """Register a ``() -> dict`` sampled on each recorded step (merged into the record)."""
        self._callbacks.append(fn)

    def _ensure_file(self) -> None:
        if self._file is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = (
                self.output_dir
                / f"inference_steps_{self.run_id}_rank{self.rank}_dp{self.dp_rank}.jsonl"
            )
            self._file = open(path, "a")

    def record(
        self,
        step: int,
        active: int,
        waiting: int,
        paused: int,
        prefill: int,
        decode: int,
        active_tokens: int,
        kv_blocks_used: Optional[int] = None,
        kv_blocks_total: Optional[int] = None,
        mamba_slots_used: Optional[int] = None,
        mamba_slots_total: Optional[int] = None,
        max_requests: Optional[int] = None,
    ) -> None:
        """Append one step's batch-size record (subject to the stride).

        ``dt`` = wall-clock seconds since the previous recorded step (``None``
        first); at stride 1 it is the preceding step's duration. No GPU sync.

        ``kv_blocks_*`` / ``mamba_slots_*`` expose the live utilization of the two
        memory pools (attention KV blocks vs. mamba state slots); ``max_requests`` is
        the derived concurrency ceiling. Together they say which pool binds, i.e. how
        to tune ``buffer-size-gb`` and ``mamba-memory-ratio``. ``None`` when
        unavailable (e.g. a non-hybrid model has no mamba pool).
        """
        if step % self.stride != 0:
            return
        now = time.time()
        dt = now - self._last_t if self._last_t is not None else None
        self._last_t = now
        record = {
            "rank": self.rank,
            "dp_rank": self.dp_rank,
            "step": int(step),
            "t": now,
            "dt": dt,
            "active": int(active),
            "waiting": int(waiting),
            "paused": int(paused),
            "prefill": int(prefill),
            "decode": int(decode),
            "active_tokens": int(active_tokens),
            "kv_blocks_used": None if kv_blocks_used is None else int(kv_blocks_used),
            "kv_blocks_total": None if kv_blocks_total is None else int(kv_blocks_total),
            "mamba_slots_used": None if mamba_slots_used is None else int(mamba_slots_used),
            "mamba_slots_total": None if mamba_slots_total is None else int(mamba_slots_total),
            "max_requests": None if max_requests is None else int(max_requests),
        }
        for callback in self._callbacks:
            record.update(callback())
        self._ensure_file()
        self._file.write(json.dumps(record) + "\n")
        self._n_since_flush += 1
        if self._n_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Flush buffered records to disk."""
        if self._file is not None:
            self._file.flush()
            self._n_since_flush = 0

    def close(self) -> None:
        """Flush and close the trace file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


_TRACER: Optional[InferenceStepTracer] = None


def init_inference_step_tracer(
    output_dir: str,
    rank: int,
    dp_rank: int,
    stride: int = 1,
    run_id: Optional[str] = None,
) -> InferenceStepTracer:
    """Initialize the process-global tracer (idempotent: returns the existing one)."""
    global _TRACER
    if _TRACER is None:
        _TRACER = InferenceStepTracer(
            output_dir=output_dir, rank=rank, dp_rank=dp_rank, stride=stride, run_id=run_id
        )
    return _TRACER


def get_inference_step_tracer() -> Optional[InferenceStepTracer]:
    """Return the process-global tracer, or None if tracing is disabled."""
    return _TRACER


def shutdown_inference_step_tracer() -> None:
    """Close and clear the process-global tracer."""
    global _TRACER
    if _TRACER is not None:
        _TRACER.close()
        _TRACER = None
