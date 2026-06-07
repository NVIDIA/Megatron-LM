# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""C1 unit tests for :class:`AsyncSampleRing` (CUDA event ordering + source ownership).

These reason about the ring in isolation -- no model, no engine -- exactly as the design's
C1 lands it (data structure + tests, not yet wired). The load-bearing properties:

* ``enqueue_readback`` does NOT host-synchronize (the launch must follow it immediately);
* ``consume`` returns the bytes that were live at ``sample(K)`` time, after the launch enqueue;
* the depth-2 double-buffer + retained source ref keep two in-flight read-backs from aliasing
  even though the GPU source tensor is rebound every step;
* ``drain`` / ``reset`` release every retained source ref.

Run (single GPU is enough; no model parallel)::

    /opt/venv/bin/python -m torch.distributed.run --nproc-per-node 1 -m pytest -v \
        tests/unit_tests/inference/text_generation_controllers/test_async_sample_ring.py
"""

import pytest
import torch

from megatron.core.inference.text_generation_controllers.async_sample_ring import (
    AsyncSampleRing,
    RingTicket,
)


@pytest.mark.internal
class TestAsyncSampleRing:
    """The asynchronous sampled-token read-back ring."""

    def _ring(self, max_requests: int = 8, depth: int = 2) -> AsyncSampleRing:
        return AsyncSampleRing(
            max_requests=max_requests, device=torch.cuda.current_device(), depth=depth
        )

    def test_enqueue_does_not_synchronize(self) -> None:
        """enqueue_readback only enqueues: a copy gated behind a long GPU op is NOT yet done.

        A deterministic GPU sleep busies the compute stream before the sample-ready event is
        recorded; the copy stream waits on that event, so the copy cannot have completed by the
        time enqueue returns. If enqueue had host-synchronized, ``copy_done`` would already be
        ready. ``consume`` is what blocks.
        """
        ring = self._ring()
        src = torch.arange(4, device=torch.cuda.current_device(), dtype=torch.int64)
        # ~tens of ms of GPU-side spin on the compute stream (deterministic, not wall-clock racy).
        torch.cuda._sleep(200_000_000)
        ticket = ring.enqueue_readback(src, n=4, snapshot_request_ids=None)

        # The copy is stream-ordered after the long sleep => cannot be done yet (no host sync).
        assert not ticket.copy_done_event.query()

        out = ring.consume(ticket)  # this blocks until the copy retires
        assert out.tolist() == [0, 1, 2, 3]

    def test_consume_returns_byte_identical_values(self) -> None:
        """consume returns exactly the sampled values, with the source dtype preserved."""
        ring = self._ring()
        for dtype in (torch.int64, torch.int32):
            r = self._ring()
            src = torch.tensor([5, 9, 2, 7, 1], device=torch.cuda.current_device(), dtype=dtype)
            ticket = r.enqueue_readback(src, n=5, snapshot_request_ids="tag")
            out = r.consume(ticket)
            assert out.dtype == dtype, f"dtype not preserved: {out.dtype} != {dtype}"
            assert out.tolist() == [5, 9, 2, 7, 1]
            assert ticket.snapshot_request_ids == "tag"

    def test_depth2_interleaved_no_alias(self) -> None:
        """Two read-backs in flight at once use distinct slots and never alias.

        Mirrors the rebind hazard: step K's source tensor is one object, step K+1's sampling
        produces a *different* tensor. With depth-2 distinct pinned slots + retained source
        refs, consuming both returns each step's own values.
        """
        ring = self._ring()
        dev = torch.cuda.current_device()
        src0 = torch.full((4,), 7, device=dev, dtype=torch.int64)
        t0 = ring.enqueue_readback(src0, n=4, snapshot_request_ids="k")
        # Anti-rebind: the exact source tensor is retained until consume.
        assert t0.source_ref is src0
        assert ring._source_refs[t0.slot] is src0

        src1 = torch.full((4,), 9, device=dev, dtype=torch.int64)  # rebound new tensor (step K+1)
        t1 = ring.enqueue_readback(src1, n=4, snapshot_request_ids="k+1")
        assert t1.slot != t0.slot, "depth-2 must hand out distinct slots for back-to-back reads"
        assert t1.source_ref is src1

        out0 = ring.consume(t0)
        out1 = ring.consume(t1)
        assert out0.tolist() == [7, 7, 7, 7], "slot 0 aliased slot 1"
        assert out1.tolist() == [9, 9, 9, 9], "slot 1 aliased slot 0"
        # Both retained refs dropped once their copies retired.
        assert ring._source_refs == [None, None]
        assert t0.source_ref is None and t1.source_ref is None

    def test_slot_reuse_after_consume(self) -> None:
        """A third read-back wraps back to slot 0 and reads its own (fresh) values."""
        ring = self._ring()
        dev = torch.cuda.current_device()
        t0 = ring.enqueue_readback(torch.full((3,), 1, device=dev, dtype=torch.int64), n=3,
                                   snapshot_request_ids=None)
        t1 = ring.enqueue_readback(torch.full((3,), 2, device=dev, dtype=torch.int64), n=3,
                                   snapshot_request_ids=None)
        assert ring.consume(t0).tolist() == [1, 1, 1]
        assert ring.consume(t1).tolist() == [2, 2, 2]
        # Wraps to slot 0 again.
        t2 = ring.enqueue_readback(torch.full((3,), 3, device=dev, dtype=torch.int64), n=3,
                                   snapshot_request_ids=None)
        assert t2.slot == t0.slot
        assert ring.consume(t2).tolist() == [3, 3, 3]

    def test_partial_n_only_reads_active_rows(self) -> None:
        """Only the first ``n`` rows are read back (the active request count)."""
        ring = self._ring(max_requests=8)
        dev = torch.cuda.current_device()
        src = torch.arange(8, device=dev, dtype=torch.int64)
        ticket = ring.enqueue_readback(src, n=3, snapshot_request_ids=None)
        out = ring.consume(ticket)
        assert out.numel() == 3 and out.tolist() == [0, 1, 2]

    def test_drain_releases_all_refs(self) -> None:
        """drain synchronizes pending copies and drops every retained source ref."""
        ring = self._ring()
        dev = torch.cuda.current_device()
        ring.enqueue_readback(torch.full((4,), 5, device=dev, dtype=torch.int64), n=4,
                              snapshot_request_ids=None)
        assert any(r is not None for r in ring._source_refs)
        ring.drain()
        assert all(r is None for r in ring._source_refs)

    def test_reset_drains_and_rewinds_cursor(self) -> None:
        """reset drains in-flight copies, drops refs, and rewinds the cursor to 0."""
        ring = self._ring()
        dev = torch.cuda.current_device()
        ring.enqueue_readback(torch.full((2,), 1, device=dev, dtype=torch.int64), n=2,
                              snapshot_request_ids=None)
        assert ring.cursor == 1
        ring.reset()
        assert ring.cursor == 0
        assert all(r is None for r in ring._source_refs)

    def test_returns_ring_ticket_shape(self) -> None:
        """The ticket carries the slot, the copy_done event, the snapshot tag, n, source_ref."""
        ring = self._ring()
        dev = torch.cuda.current_device()
        snap = torch.tensor([10, 11], device=dev)
        ticket = ring.enqueue_readback(
            torch.tensor([3, 4], device=dev, dtype=torch.int64), n=2, snapshot_request_ids=snap
        )
        assert isinstance(ticket, RingTicket)
        assert ticket.n == 2
        assert isinstance(ticket.copy_done_event, torch.cuda.Event)
        assert ticket.snapshot_request_ids is snap
        ring.consume(ticket)

    def test_dtype_change_is_rejected(self) -> None:
        """The pinned slots are allocated once; a later source dtype mismatch is loud."""
        ring = self._ring()
        dev = torch.cuda.current_device()
        t = ring.enqueue_readback(torch.tensor([1, 2], device=dev, dtype=torch.int64), n=2,
                                  snapshot_request_ids=None)
        ring.consume(t)
        with pytest.raises(AssertionError, match="dtype changed"):
            ring.enqueue_readback(torch.tensor([1, 2], device=dev, dtype=torch.int32), n=2,
                                  snapshot_request_ids=None)

    def test_depth_must_be_at_least_two(self) -> None:
        """A depth-1 ring is rejected (no double-buffer margin)."""
        with pytest.raises(AssertionError, match="depth must be >= 2"):
            AsyncSampleRing(max_requests=4, device=torch.cuda.current_device(), depth=1)
