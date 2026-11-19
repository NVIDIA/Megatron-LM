# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay intervals of consecutive fused seam writes.

Every training seam records its write into a ``ReplayInterval`` (one write per interval without
checkpoint replay). An interval keeps only its first input residual (the checkpoint) and the
writes' tapes, which the seams save for their backward anyway. Its one replay launch rebuilds
every seam's input residual bit-exactly and fills its readers' ``ColumnSlot``s with the
columnwise MXFP8 operands their consumers' weight-gradient GEMMs unpack. It runs before any
backward GEMM of the interval: in the backward of the next interval's first seam, or for a
stack's last interval with column slots in a gradient hook on the stack's output
(``replay_before``); a seam's own backward replays it at the latest.

Fusion admits only schedules whose forwards and backwards alternate on one stream (no pipeline
parallelism, overlapped expert-parallel communication or full recompute), so the replay needs no
events and hands each input residual out once: a retained graph's second backward raises.
"""

from dataclasses import dataclass

import torch


@dataclass(eq=False)
class ColumnSlot:
    """One MXFP8 reader's columnwise operand, filled by its interval's replay. Its consumers'
    saved slots hold it until their weight-gradient GEMMs unpack it."""

    columns: object = None


class ReplayInterval:
    """Consecutive fused writes replayed together from their first input residual."""

    def __init__(self, previous: "ReplayInterval | None" = None):
        # The interval whose last write this one's first seam applies (replayed in its backward).
        self.previous = previous
        self.checkpoint = self.last = self.predecessors = None
        self.tapes, self.slots = [], []

    def record(self, previous, current, saved, slots) -> int:
        """Record one write (its input, output, ``SavedWrite`` tape and one ColumnSlot or None
        per reader); returns its index."""
        if self.checkpoint is None:
            self.checkpoint = previous  # the only residual the interval keeps
        elif previous.data_ptr() != self.last:
            raise ValueError("replayed writes must form one connected chain")
        self.tapes.append(saved)
        self.slots.append(tuple(slots))
        self.last = current.data_ptr()
        return len(self.tapes) - 1

    @torch.no_grad()
    def replay(self):
        """Run the interval's replay launch, once; it then keeps only the input residuals."""
        if self.predecessors is not None:
            return
        from .replay_kernel import replay_interval

        tapes = tuple(self.tapes)
        checkpoint = self.checkpoint.view(-1, tapes[0].num_streams * tapes[0].update.shape[-1])
        requested = tuple(tuple(slot is not None for slot in slots) for slots in self.slots)
        predecessors, columns = replay_interval(checkpoint, tapes, prepare_columns=requested)
        for slots, per_tape in zip(self.slots, columns):
            for slot, column in zip(slots, per_tape):
                if slot is not None:
                    slot.columns = column
        self.predecessors = list(predecessors)
        self.checkpoint, self.tapes, self.slots = None, (), ()

    def replay_before(self, tensor):
        """Replay when ``tensor``'s gradient is consumed if a reader needs columns: for a stack's
        last interval, whose readers' consumers run their backward after the stack output's and
        before any seam's (without columns, its last seam's backward replays it, later)."""
        if tensor.requires_grad and any(any(slots) for slots in self.slots):
            tensor.register_hook(lambda grad: self.replay())
        return tensor

    def take(self, index):
        """Write ``index``'s input residual, handed out once."""
        self.replay()
        previous, self.predecessors[index] = self.predecessors[index], None
        if previous is None:
            raise RuntimeError("fused seams do not support a retained graph's second backward")
        return previous


class ReplayChain:
    """One fused stack's open interval during its forward (forwards and backwards alternate)."""

    def __init__(self):
        self.open = None

    def interval(self, starts: bool) -> ReplayInterval:
        """The interval a seam records into: a new one where an interval starts."""
        if starts:
            self.open = ReplayInterval(self.open)
        elif self.open is None:
            raise RuntimeError("a replayed seam has no open replay interval")
        return self.open

    def close(self, output):
        """End the forward at the stack's ``output``: its last interval replays before it."""
        interval, self.open = self.open, None
        return output if interval is None else interval.replay_before(output)
