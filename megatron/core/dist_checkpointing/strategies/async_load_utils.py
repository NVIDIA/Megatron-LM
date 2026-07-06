# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Async load utilities for dist_checkpointing.

The disk read runs in a Python thread of the calling process: it only performs
file I/O and memcpy into the destination CPU tensors (releasing the GIL for
both), and never touches ``torch.distributed`` or CUDA, so it can overlap with
NCCL collectives issued by the main thread. A thread (rather than a spawned
process) writes directly into the caller's pinned tensors with no IPC and no
interpreter re-import cost.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class AsyncLoadPlan:
    """Cacheable result of the plan phase of an async load.

    Built once per checkpoint via
    :meth:`TorchDistLoadShardedStrategy.prepare_async_load` and replayed with
    :func:`spawn_async_read`. The planner is the same instance that was set up
    during the plan phase, bound to ``worker_state_dict`` — the read thread
    reuses it so sharded read items resolve into the correct destination
    tensors.

    Note: ``worker_state_dict`` stays bound to the destination tensors that
    were live at prepare time. If a caller recycles those tensors, the plan
    must be rebuilt (or rebound via ``prepare_async_load_reusing_topology``)
    before reuse.
    """

    storage_reader: Any
    final_local_plan: Any
    metadata: Any
    worker_state_dict: Dict[str, Any]
    worker_planner: Any
    finalize_fn: Callable[[], Any]


def _thread_entry(
    storage_reader,
    final_local_plan,
    worker_planner,
    done_event: threading.Event,
    error_holder: dict,
) -> None:
    """Read-thread entry point: pure file I/O + memcpy, no distributed or CUDA calls."""
    try:
        future = storage_reader.read_data(final_local_plan, worker_planner)
        future.wait()
    except BaseException as e:  # noqa: BLE001 — re-raised on finalize
        error_holder["exc"] = e
    finally:
        done_event.set()


@dataclass
class AsyncLoadRequest:
    """Handle to an in-flight async load.

    ``maybe_finalize`` is collective. With ``blocking=True`` it joins the read
    thread and finalizes. With ``blocking=False`` it first checks across ranks
    whether every read thread has finished and returns None otherwise — this
    check is mandatory, since finalizing on a subset of ranks while others are
    still reading would deadlock any collective issued downstream.
    """

    thread: threading.Thread
    done_event: threading.Event
    error_holder: dict
    finalize_fn: Callable[[], Any]
    call_idx: int = 0
    _finalized: bool = False

    def is_done(self) -> bool:
        """True if this rank's read thread has finished (local check only)."""
        return self.done_event.is_set()

    @staticmethod
    def _sync_all_workers_done(local_is_alive: int) -> bool:
        """Cross-rank check: True iff every rank's read thread is done."""
        if not torch.distributed.is_initialized():
            return local_is_alive == 0
        ten = torch.tensor([local_is_alive], dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.all_reduce(ten)
        return ten[0].item() == 0

    def maybe_finalize(self, blocking: bool = True) -> Optional[Any]:
        """Join the read thread, synchronize with peers and run finalization.

        Args:
            blocking (bool, optional): if False, returns None (without joining)
                unless the read thread finished on every rank. Defaults to True.

        Returns: the loaded state dict, or None if ``blocking=False`` and some
            rank is still reading.
        """
        if self._finalized:
            raise RuntimeError("AsyncLoadRequest.maybe_finalize already called.")

        if not blocking:
            local_is_alive = int(not self.done_event.is_set())
            if not self._sync_all_workers_done(local_is_alive):
                return None

        self.thread.join()

        # Broadcast the error flag before raising locally, so that ranks whose
        # read succeeded don't enter the call_idx collective below (and hang)
        # while a failed rank has already raised.
        local_has_error = int("exc" in self.error_holder)
        if torch.distributed.is_initialized():
            err_ten = torch.tensor(
                [local_has_error], dtype=torch.int, device=torch.cuda.current_device()
            )
            torch.distributed.all_reduce(err_ten, op=torch.distributed.ReduceOp.MAX)
            any_rank_has_error = bool(err_ten.item())
        else:
            any_rank_has_error = bool(local_has_error)

        if local_has_error:
            self._finalized = True
            raise RuntimeError(
                f"Async load thread failed: {self.error_holder['exc']!r}"
            ) from self.error_holder["exc"]
        if any_rank_has_error:
            self._finalized = True
            raise RuntimeError(
                "Async load aborted: a peer rank's read thread raised. "
                "See that rank's stderr for the original exception."
            )

        # Every rank must finalize the same call_idx. MIN and MAX are reduced
        # together so that all ranks observe a mismatch and raise.
        min_idx, max_idx = self._world_min_max(self.call_idx)
        if min_idx != max_idx:
            self._finalized = True
            raise RuntimeError(
                f"Async load call_idx mismatch across ranks (local={self.call_idx}, "
                f"min={min_idx}, max={max_idx}); some ranks skipped or reordered "
                f"maybe_finalize."
            )

        result = self.finalize_fn()
        self._finalized = True
        return result

    @staticmethod
    def _world_min_max(value: int) -> "tuple[int, int]":
        """All-reduce (min, max) of ``value`` across ranks in one collective."""
        if not torch.distributed.is_initialized():
            return value, value
        ten = torch.tensor([value, -value], dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.all_reduce(ten, op=torch.distributed.ReduceOp.MAX)
        return -int(ten[1].item()), int(ten[0].item())


def spawn_async_read(
    storage_reader,
    final_local_plan,
    worker_planner,
    finalize_fn: Callable[[], Any],
    call_idx: int = 0,
) -> AsyncLoadRequest:
    """Launch a background thread performing ``storage_reader.read_data``.

    Args:
        storage_reader: DCP storage reader with metadata already set up.
        final_local_plan: this rank's slice of the load plan.
        worker_planner: the planner used during the plan phase, bound to the
            destination state dict.
        finalize_fn (Callable): produces the final state dict once the read
            is complete.
        call_idx (int, optional): sequence number validated across ranks at
            finalization. Defaults to 0.

    Returns: an AsyncLoadRequest handle.
    """
    done_event = threading.Event()
    error_holder: dict = {}
    thread = threading.Thread(
        target=_thread_entry,
        args=(storage_reader, final_local_plan, worker_planner, done_event, error_holder),
        name=f"dist-checkpointing-async-load-{call_idx}",
        daemon=True,
    )
    thread.start()
    return AsyncLoadRequest(
        thread=thread,
        done_event=done_event,
        error_holder=error_holder,
        finalize_fn=finalize_fn,
        call_idx=call_idx,
    )
