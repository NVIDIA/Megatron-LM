# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Asynchronous device->host read-back of sampled decode tokens.

The launch-before-commit decode pipeline must enqueue the launch of the next forward
``F(K+1)`` *before* the sampled tokens of ``F(K)`` are read to the host (that blocking D2H +
host finish-check is the ~360us inter-forward bubble). :class:`AsyncSampleRing` owns that
read-back so it no longer sits *between* ``sample(K)`` and the launch *enqueue*:

* a small fixed ring (``depth=2``) of pinned CPU slots + one dedicated copy stream;
* per slot a ``source_ready`` event (recorded on the compute stream after the sample
  kernel) and a ``copy_done`` event (recorded on the copy stream after the D2H);
* the D2H is enqueued behind ``source_ready`` and is **not** waited on at launch time;
  the host syncs on ``copy_done`` only *after* the launch enqueue, off the launch path.

``depth=2`` is a **double-buffer correctness margin**, not a deeper pipeline: the speculation
depth stays 1. Step ``K+1``'s sampling rebinds/overwrites the GPU source while step ``K``'s
slot may still be draining, so the reader must own a distinct pinned slot.

**Critical ownership rule (anti-rebind).** Plain-decode ``_sampled_tokens_cuda`` is *rebound*
each step to the sample kernel's output -- it is **not** a stable address. If nothing held a
reference to slot ``K``'s exact source GPU tensor, the caching allocator could reuse that
memory for step ``K+1``'s sample before the (still pending) copy stream drains it, so the
async D2H would silently read step ``K+1``'s samples. The ring therefore retains a reference
to the exact source tensor for slot ``K`` (``RingTicket.source_ref``) until ``copy_done(K)``
clears, pinning that memory for the lifetime of the copy.

This module is a pure data structure (mirrors how :mod:`metadata_slot` /
:mod:`step_transaction` were landed before being wired). It is constructed lazily on the
controller only when ``enable_async_scheduling`` is on; the serial path never touches it.

Scope note: MTP (``num_speculative_tokens > 0``) is out of scope for the overlap path, so the
ring carries only the base sampled tokens; the ``_sampled_mtp_tokens_cuda`` companion D2H
stays on the synchronous serial path.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import torch


@dataclass
class RingTicket:
    """A handle to one enqueued async read-back, consumed after the launch enqueue.

    ``source_ref`` retains the exact GPU source tensor for this slot until ``copy_done``
    clears (the anti-rebind rule above); it is dropped by :meth:`AsyncSampleRing.consume`
    (or :meth:`AsyncSampleRing.drain`) once the copy is known complete.
    """

    slot: int
    copy_done_event: torch.cuda.Event
    snapshot_request_ids: Any  # the launched StepTxn.request_ids (consume-by-request-id tag)
    n: int
    source_ref: Optional[torch.Tensor]  # retained until copy_done clears (anti-rebind)


class AsyncSampleRing:
    """Depth-2 ring of pinned CPU slots draining sampled tokens off the launch critical path.

    One dedicated copy stream serializes the per-step D2Hs; per-slot ``source_ready`` /
    ``copy_done`` events order each copy after its sample kernel and let the host sync on
    completion *after* the launch enqueue. The pinned slots are allocated lazily on the first
    :meth:`enqueue_readback` to match the source dtype exactly, so the consumed tensor is
    byte-identical to the legacy ``sampled_tokens_cuda[:n].cpu()`` it replaces.
    """

    def __init__(self, max_requests: int, device: Any, depth: int = 2):
        assert depth >= 2, "depth must be >= 2 (double-buffer correctness margin)"
        self.max_requests = max_requests
        self.device = device
        self.depth = depth
        # One dedicated copy stream for all D2Hs (kept off the compute stream).
        self.copy_stream = torch.cuda.Stream(device=device)
        # Per-slot fences. source_ready: sample kernel done (compute stream). copy_done: D2H
        # done (copy stream). Allocated up front; the pinned slots are allocated lazily.
        self._source_ready: List[torch.cuda.Event] = [torch.cuda.Event() for _ in range(depth)]
        self._copy_done: List[torch.cuda.Event] = [torch.cuda.Event() for _ in range(depth)]
        # Pinned CPU slots (lazily allocated, see _ensure_slots) + per-slot retained source ref.
        self._slots: Optional[List[torch.Tensor]] = None
        self._slot_dtype: Optional[torch.dtype] = None
        self._source_refs: List[Optional[torch.Tensor]] = [None] * depth
        self.cursor = 0

    def _ensure_slots(self, dtype: torch.dtype) -> None:
        """Lazily allocate the pinned CPU slots matching the source dtype (once)."""
        if self._slots is None:
            self._slot_dtype = dtype
            self._slots = [
                torch.empty(self.max_requests, dtype=dtype, pin_memory=True)
                for _ in range(self.depth)
            ]
        else:
            assert dtype == self._slot_dtype, (
                f"AsyncSampleRing source dtype changed: {self._slot_dtype} -> {dtype}"
            )

    def enqueue_readback(
        self, sampled_tokens_cuda: torch.Tensor, *, n: int, snapshot_request_ids: Any
    ) -> RingTicket:
        """Enqueue an async D2H of ``sampled_tokens_cuda[:n]`` and return its ticket.

        Records ``source_ready`` on the current (compute) stream, then on the copy stream
        waits for it and issues the non-blocking pinned copy, recording ``copy_done``. Retains
        the source tensor (anti-rebind) and advances the cursor. Does **not** synchronize, so
        the caller can immediately enqueue the next forward's launch.
        """
        assert n <= self.max_requests, f"n={n} exceeds max_requests={self.max_requests}"
        self._ensure_slots(sampled_tokens_cuda.dtype)
        slot = self.cursor
        source_ready = self._source_ready[slot]
        copy_done = self._copy_done[slot]

        # source_ready fires on the compute stream once the sample kernel has written
        # sampled_tokens_cuda; the copy stream waits on it before reading.
        source_ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(source_ready)
            self._slots[slot][:n].copy_(sampled_tokens_cuda[:n], non_blocking=True)
            copy_done.record(self.copy_stream)

        # Retain the EXACT source tensor until copy_done clears (anti-rebind): keeps the
        # caching allocator from reusing this memory for the next step's sample mid-copy.
        self._source_refs[slot] = sampled_tokens_cuda
        self.cursor = (self.cursor + 1) % self.depth

        return RingTicket(
            slot=slot,
            copy_done_event=copy_done,
            snapshot_request_ids=snapshot_request_ids,
            n=n,
            source_ref=sampled_tokens_cuda,
        )

    def consume(self, ticket: RingTicket) -> torch.Tensor:
        """Block on the ticket's ``copy_done`` and return its pinned slot view ``[:n]``.

        Called *after* the launch enqueue, so the D2H overlapped the launch. The caller must
        CLONE the returned tensor before any in-place state mutation (the slot is reused after
        ``depth`` further enqueues). Drops the retained source ref now that the copy is done.
        """
        ticket.copy_done_event.synchronize()
        self._source_refs[ticket.slot] = None
        ticket.source_ref = None
        return self._slots[ticket.slot][: ticket.n]

    def drain(self) -> None:
        """Synchronize all in-flight copy events and drop every retained source ref.

        Used on barrier / suspend / reset / shutdown so no pending D2H references a tensor the
        teardown is about to free.
        """
        for event in self._copy_done:
            event.synchronize()
        self._source_refs = [None] * self.depth

    def reset(self) -> None:
        """Drain and rewind the cursor (e.g. an engine reset that rewinds the context)."""
        self.drain()
        self.cursor = 0
