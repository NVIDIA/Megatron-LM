# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay intervals of consecutive fused seam writes.

Every training seam records its write into a ``ReplayInterval`` (one write per interval without
checkpoint replay). An interval keeps only its first input residual (the checkpoint) and the
writes' tapes, which the seams save for their backward anyway. Its one replay launch rebuilds
every seam's input residual bit-exactly and fills its readers' ``ColumnSlot``s with the
columnwise MXFP8 operands their consumers' weight-gradient GEMMs unpack. It runs before any
backward GEMM of the interval: in the backward of the next interval's first seam, or for a
stack's last interval with column slots in a gradient hook on the stack's output
(``replay_before``); a seam's own backward replays it at the latest. TE Linears consuming an
MXFP8 read unpack its slot (``install_columns``). The caller decides where intervals start.

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


def install_columns(output, operand, slot) -> None:
    """After a TE Linear's forward (with grad) on a read's MXFP8 ``operand``: its backward
    unpacks ``slot``'s replayed columnwise operand in place of the saved rowwise one (TE's own
    save slot only; a scope-wide saved-tensor hook would detach the saved weight and lose GTP
    metadata). TE's backward, including weight-gradient accumulation and distributed-weight
    finalization, runs unchanged."""
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    if not FP8GlobalStateManager.is_fp8_enabled():
        raise RuntimeError("an MXFP8 consumer of a fused read ran outside its FP8 context")

    def unpack(size):
        if slot.columns is None:
            raise RuntimeError("a fused read's replay did not run before its consumer's backward")
        return slot.columns.view(size)

    node = (output[0] if isinstance(output, tuple) else output).grad_fn
    key = operand._rowwise_data.untyped_storage()._cdata
    matched = 0
    for value, saved in zip(node.saved_tensors, node._raw_saved_tensors):
        data = getattr(value, "_rowwise_data", None)
        if data is not None and data.untyped_storage()._cdata == key:
            saved.register_hooks(torch.Tensor.size, unpack)
            matched += 1
    if matched != 1:
        raise RuntimeError("TE did not save exactly one shortcut MXFP8 activation")
