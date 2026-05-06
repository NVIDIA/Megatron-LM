# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import os
from megatron.training.utils.common_utils import print_rank_0
import torch

from megatron.core._rank_utils import safe_get_rank
from megatron.training.config import ProfilingConfig

import logging

logger = logging.getLogger(__name__)


def start_memory_history_recording(profiling: ProfilingConfig | None) -> None:
    """Enable the CUDA caching allocator trace so memory snapshots contain history.

    ``torch.cuda.memory._snapshot()`` only includes allocation/free events and
    Python stack context after ``_record_memory_history()`` has been enabled.
    Without this call, dumped snapshots contain only the current live
    allocations — no timeline, no call sites.

    Must be invoked before model construction so every tensor allocation is
    captured. Guarded by ``profile_ranks`` so only ranks that will dump a
    snapshot pay the recording overhead.
    """
    if profiling is None or not profiling.record_memory_history:
        return
    if safe_get_rank() not in profiling.profile_ranks:
        return

    torch.cuda.memory._record_memory_history(
        True,
        # Retain up to 100k alloc/free events.
        trace_alloc_max_entries=100_000,
        # Record the Python stack at each event — lets memory_viz show call sites.
        trace_alloc_record_context=True,
    )

    def _oom_observer(
        device: int, alloc: int, device_alloc: int, device_free: int
    ) -> None:
        """Dump a snapshot on OOM so we can inspect what was live at the failure."""
        import pickle

        rank = safe_get_rank()
        base, ext = os.path.splitext(profiling.memory_snapshot_path)
        filename = f"{base}_oom_rank-{rank}{ext}"
        snapshot = torch.cuda.memory._snapshot()
        with open(filename, "wb") as f:
            pickle.dump(snapshot, f)
        # logger.info so the message reaches stderr on any profiled rank, not just rank 0.
        logger.info(f"[OOM] rank {rank} saved memory snapshot to {filename}")

    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
    print_rank_0(
        f"Memory history recording enabled (rank {safe_get_rank()}); "
        f"snapshots will be written to '{profiling.memory_snapshot_path}'."
    )
