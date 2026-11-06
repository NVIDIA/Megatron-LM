# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint replay of consecutive fused seam writes.

A ``ReplayInterval`` keeps only its first input residual (the checkpoint) and the writes' tapes,
which the seams save for their backward anyway. A seam recording into an interval does not save
its input residual: its backward takes it from its ``ReplayHandle``, and the interval's first
request replays all of its writes bit-exactly in one launch (also emitting reader zero's
columnwise MXFP8 operands), so replay always precedes the seam backward and the TE weight
gradients deferred to it. The caller decides where intervals start (see ``WriteReadNorm``).
"""

from dataclasses import dataclass

import torch


def graph_retained() -> bool:
    """Whether the running backward retains its graph (a later backward needs the handles)."""
    keep_graph = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)
    return True if keep_graph is None else keep_graph()


class ReplayInterval:
    """Consecutive fused writes replayed together from their first input residual."""

    def __init__(self):
        self.checkpoint = self.last = None
        self.tapes, self.prepare_columns, self.written = [], [], []
        self.predecessors = self.columns = self.ready = None

    def record(self, previous, current, saved, *, prepare_columns=False) -> "ReplayHandle":
        """Record one write (its input, output and ``SavedWrite`` tape)."""
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
        self.predecessors, self.columns = list(predecessors), list(columns)
        self.ready = torch.cuda.Event()
        self.ready.record(stream)

    def take(self, index):
        """Write ``index``'s input residual and reader-zero columns (None unless requested).

        The first request replays the interval; each entry is handed out once, and a retained
        graph's second backward replays again (bit-exactly).
        """
        with torch.no_grad():
            if self.predecessors is None or self.predecessors[index] is None:
                self._replay()
        previous, self.predecessors[index] = self.predecessors[index], None
        column, self.columns[index] = self.columns[index], None
        stream = torch.cuda.current_stream(previous.device)
        stream.wait_event(self.ready)
        previous.record_stream(stream)
        if column is not None:
            column._columnwise_data.record_stream(stream)
            column._columnwise_scale_inv.record_stream(stream)
        return previous, column


@dataclass(frozen=True)
class ReplayHandle:
    """One write's position in its replay interval."""

    interval: ReplayInterval
    index: int

    def take(self):
        """The write's input residual and reader-zero columns, replayed on first need."""
        return self.interval.take(self.index)
