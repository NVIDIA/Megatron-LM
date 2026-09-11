# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import logging
from argparse import Namespace
from pathlib import Path

import torch

from megatron.core._rank_utils import safe_get_rank
from megatron.training.config import ProfilingConfig

logger = logging.getLogger(__name__)


def memory_snapshot_path(
    snapshot_path: str, profile_ranks: list[int], rank: int | None = None, tag: str = ""
) -> str:
    """Return the CUDA memory snapshot path this rank writes.

    The rank is only appended when several ranks dump, so the configured path keeps working
    for everyone who does not set ``--profile-ranks``. A tagged snapshot (an OOM dump) is
    always disambiguated, since any rank may produce one.
    """
    if not tag and len(profile_ranks) <= 1:
        return snapshot_path
    rank = safe_get_rank() if rank is None else rank
    path = Path(snapshot_path)
    suffix = path.suffix or ".pickle"
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}{tag}_rank-{rank}{suffix}"))


def start_memory_history_recording(profiling: ProfilingConfig | Namespace | None) -> None:
    """Enable the CUDA caching allocator trace so memory snapshots contain history.

    ``torch.cuda.memory._snapshot()`` only includes allocation/free events and
    Python stack context after ``_record_memory_history()`` has been enabled.
    Without this call, dumped snapshots contain only the current live
    allocations — no timeline, no call sites.

    Must be invoked before model construction so every tensor allocation is
    captured. Guarded by ``profile_ranks`` so only ranks that will dump a
    snapshot pay the recording overhead.
    """
    if profiling is None or not getattr(profiling, "record_memory_history", False):
        return
    profile_ranks = getattr(profiling, "profile_ranks", [])
    if len(profile_ranks) != 0:
        if safe_get_rank() not in profile_ranks:
            return

    torch.cuda.memory._record_memory_history(
        True,
        # Retain up to 100k alloc/free events.
        trace_alloc_max_entries=100_000,
        # Record the Python stack at each event — lets memory_viz show call sites.
        trace_alloc_record_context=True,
    )

    def _oom_observer(device: int, alloc: int, device_alloc: int, device_free: int) -> None:
        """Dump a snapshot on OOM so we can inspect what was live at the failure."""
        rank = safe_get_rank()
        filename = memory_snapshot_path(
            profiling.memory_snapshot_path, profile_ranks, rank, tag="_oom"
        )
        torch.cuda.memory._dump_snapshot(filename)
        # logger.info so the message reaches stderr on any profiled rank, not just rank 0.
        logger.info(f"[OOM] rank {rank} saved memory snapshot to {filename}")

    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
    snapshot_path = memory_snapshot_path(profiling.memory_snapshot_path, profile_ranks)
    Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Memory history recording enabled on rank %s; snapshot will be written to '%s'.",
        safe_get_rank(),
        snapshot_path,
    )
