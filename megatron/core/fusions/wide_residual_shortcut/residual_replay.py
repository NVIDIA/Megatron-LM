# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay intervals of consecutive fused seam writes.

Every training seam records its write into a ``ReplayInterval`` (one write per interval without
checkpoint replay). An interval keeps only its first input residual (the checkpoint) and the
writes' tapes, which the seams save for their backward anyway; a seam's backward takes its input
residual from its ``ReplayHandle``. The interval replays its writes bit-exactly in one launch
(also emitting the requested readers' columnwise MXFP8 operands, once each, for their
consumers' weight-gradient GEMMs; a one-write interval without them launches nothing) in the
backward of the seam that applies its last write, which precedes every backward GEMM of the
interval, or on first request. The caller decides where intervals start (see ``WriteReadNorm``).
"""

import weakref
from dataclasses import dataclass

import torch


def graph_retained() -> bool:
    """Whether the running backward retains its graph (a later backward needs the handles)."""
    keep_graph = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)
    return True if keep_graph is None else keep_graph()


class ReplayInterval:
    """Consecutive fused writes replayed together from their first input residual."""

    def __init__(self, previous: "ReplayInterval | None" = None):
        # The interval whose last write this one's first seam applies (replayed from its backward).
        self.previous = None if previous is None else weakref.ref(previous)
        self.checkpoint = self.last = None
        self.tapes, self.prepare_columns, self.written = [], [], []
        self.predecessors = self.columns = self.ready = None

    def record(self, previous, current, saved, *, prepare_columns=False) -> "ReplayHandle":
        """Record one write (its input, output and ``SavedWrite`` tape); ``prepare_columns`` is
        a bool (reader zero) or one bool per reader."""
        if self.checkpoint is None:
            self.checkpoint = previous  # the only residual the interval keeps
        elif previous.data_ptr() != self.last:
            raise ValueError("replayed writes must form one connected chain")
        handle = ReplayHandle(self, len(self.tapes))
        self.tapes.append(saved)
        self.prepare_columns.append(prepare_columns)
        written = torch.cuda.Event()
        written.record(torch.cuda.current_stream(current.device))
        self.written.append(written)
        self.last = current.data_ptr()
        return handle

    def _replay(self):
        from .replay_kernel import replay_interval

        stream = torch.cuda.current_stream(self.checkpoint.device)
        for event in self.written:  # the forward writes, possibly on other streams
            stream.wait_event(event)
        tapes = tuple(self.tapes)
        for tensor in (self.checkpoint,) + tuple(
            t for tape in tapes for t in (tape.update, tape.coefficients, *tape.gammas, *tape.rstd)
        ):
            tensor.record_stream(stream)
        checkpoint = self.checkpoint.view(-1, tapes[0].num_streams * tapes[0].update.shape[-1])
        predecessors, columns = replay_interval(
            checkpoint, tapes, prepare_columns=tuple(self.prepare_columns)
        )
        self.predecessors = list(predecessors)
        self.columns = [list(per_tape) for per_tape in columns]
        self.ready = torch.cuda.Event()
        self.ready.record(stream)

    def replay_previous(self):
        """Replay the previous interval unless it already was (its seams' backward follows)."""
        previous = None if self.previous is None else self.previous()
        if previous is not None and previous.predecessors is None:
            with torch.no_grad():
                previous._replay()

    def _take(self, entries, index):
        """Hand out ``entries()[index]`` once; a missing one (not yet replayed, or a retained
        graph's second backward) replays the interval again, bit-exactly."""
        if self.ready is None or entries()[index] is None:  # ``_replay`` sets all three
            with torch.no_grad():
                self._replay()
        value, entries()[index] = entries()[index], None
        stream = torch.cuda.current_stream(self.checkpoint.device)
        stream.wait_event(self.ready)
        return value, stream

    def take_previous(self, index):
        """Write ``index``'s input residual."""
        previous, stream = self._take(lambda: self.predecessors, index)
        previous.record_stream(stream)
        return previous

    def take_columns(self, index, reader=0):
        """Write ``index``'s columnwise MXFP8 operand of ``reader`` (requested with
        ``prepare_columns``), handed out once like a predecessor."""
        requested = self.prepare_columns[index]
        requested = (requested,) if isinstance(requested, bool) else tuple(requested)
        if reader >= len(requested) or not requested[reader]:
            raise RuntimeError("this replayed write did not record this reader's MXFP8 columns")
        column, stream = self._take(lambda: self.columns[index], reader)
        column._columnwise_data.record_stream(stream)
        column._columnwise_scale_inv.record_stream(stream)
        return column


@dataclass(frozen=True)
class ReplayHandle:
    """One write's position in its replay interval."""

    interval: ReplayInterval
    index: int

    def previous(self):
        """The write's input residual."""
        return self.interval.take_previous(self.index)

    def columns(self, reader=0):
        """The write's columnwise MXFP8 operand of ``reader``."""
        return self.interval.take_columns(self.index, reader)
