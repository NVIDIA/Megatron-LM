# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint forward replay of residual writes and call-local residual storage ownership.

Within a scope, every interval of physical layers keeps only its checkpoint residual; the
intermediate residuals are freed at their successor's write and replayed (bit-exactly, by the
fused or the native writer) on first use in backward. Each interval owns its allocations.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import groupby
from threading import RLock
from weakref import WeakValueDictionary, ref

import torch

_ACTIVE = ContextVar("wide_residual_replay", default=None)


def current_manager():
    """Return this forward's replay owner, excluding explicit no-grad regions."""
    return _ACTIVE.get() if torch.is_grad_enabled() else None


def watch_residual(tensor):
    """Record a native reader's storage dependency inside autograd forward."""
    manager = _ACTIVE.get()
    return None if manager is None else manager.slot(tensor)


def register_write(previous, current, update, write, retain, streams, max_forget):
    """Register a native streamwise write, replayed by its own Triton implementation."""
    manager = _ACTIVE.get()
    if manager is None:
        return None
    update, write = update.detach(), write.detach()
    retain = None if retain is None else retain.detach()

    def replay(value):
        from megatron.core.transformer.streamwise_residual_ops import (
            _streamwise_sigmoid_write_triton,
        )

        return _streamwise_sigmoid_write_triton(value, update, write, streams, retain, max_forget)

    return manager.register(
        previous,
        current,
        native=replay,
        dependencies=tuple(t for t in (update, write, retain) if t is not None),
    )


class ResidualSlot:
    """Preserve a StorageImpl and its allocator-owned backing across consumers."""

    def __init__(self, tensor, anchor=True, *, backing=None, ready=None):
        self.tensor = tensor.detach()
        self.anchor = anchor
        self.provider = None
        self.backing = backing
        self.lock = RLock()
        self.ready = ready
        if ready is None:
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(tensor.device))

    def resident(self):
        """Whether the original StorageImpl currently has bytes."""
        return self.tensor.untyped_storage().nbytes() != 0

    def ensure(self):
        """Materialize once, then publish and retain storage on the caller stream."""
        with self.lock:
            provider = None if self.resident() else self.provider
        if provider is not None:
            provider.materialize()
        with self.lock:
            if not self.resident():
                raise RuntimeError("checkpoint replay did not restore this residual")
            stream = torch.cuda.current_stream(self.tensor.device)
            stream.wait_event(self.ready)
            # share_storage wraps the DataPtr; also record the original allocation's stream use.
            (self.backing if self.backing is not None else self.tensor).record_stream(stream)
            return self.tensor

    def restore(self, value, ready):
        """Reattach a full zero-offset allocation without changing tensor identity."""
        from megatron.core.tensor_parallel.random import _get_share_storage

        with self.lock:
            self.backing, self.ready = value, ready
            _get_share_storage()(self.tensor, value)
            self.provider = None

    def without_replay_owner(self):
        """Keep stream/lifetime metadata for retained autograd, not interval tapes."""
        with self.lock:
            return ResidualSlot(self.tensor, backing=self.backing, ready=self.ready)


def retained_slot(slot):
    """Retained graphs keep allocation metadata; ordinary backward frees the owners."""
    keep_graph = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)
    retain = True if keep_graph is None else keep_graph()
    return slot.without_replay_owner() if retain else None


@dataclass
class _Transition:
    saved: object = None
    native: object = None
    dependencies: tuple = ()
    prepare_columns: bool = False
    column: object = None


@dataclass(frozen=True)
class ResidualLink:
    """One write's storage handles and interval position."""

    previous_slot: ResidualSlot
    current_slot: ResidualSlot
    interval: object
    index: int

    def columns(self):
        """Hand over the replay-produced reader-zero operand (None after the first call).

        A retained-graph second backward rebuilds its columns from the predecessor.
        """
        self.interval.materialize()
        stream = torch.cuda.current_stream(self.previous_slot.tensor.device)
        stream.wait_event(self.interval.ready)
        transition = self.interval.transitions[self.index]
        column, transition.column = transition.column, None
        if column is not None:
            column._columnwise_data.record_stream(stream)
            column._columnwise_scale_inv.record_stream(stream)
        return column


class _ReplayInterval:
    def __init__(self, checkpoint):
        # The checkpoint is resident; avoid retaining the preceding interval.
        self.checkpoint = checkpoint.without_replay_owner()
        self.transitions = []
        self.outputs = []
        self.sealed = False
        self.materialized = False
        self.lock = RLock()
        self.ready = None
        self.forward_ready = None

    def materialize(self):
        """Replay the whole interval once, on the requesting stream."""
        with self.lock, torch.no_grad():
            if self.materialized:
                return
            if not self.sealed:
                raise RuntimeError("seal the physical-layer interval before replay")
            from .replay_kernel import replay_interval

            stream = torch.cuda.current_stream(self.checkpoint.tensor.device)
            stream.wait_event(self.forward_ready)
            current = self.checkpoint.ensure()
            restored = []
            runs = [list(run) for _, run in groupby(self.transitions, lambda t: t.native is None)]
            for position, run in enumerate(runs):
                if run[0].native is not None:
                    for transition in run:
                        for tensor in transition.dependencies:
                            tensor.record_stream(stream)
                        current = transition.native(current)
                        restored.append(current)
                    continue
                # Replay a run of fused writes in one launch.
                tapes = tuple(t.saved for t in run)
                for tape in tapes:
                    for tensor in (tape.update, tape.coefficients, *tape.gammas, *tape.rstd):
                        tensor.record_stream(stream)
                current = current.view(-1, tapes[0].num_streams * tapes[0].update.shape[-1])
                # The run's last result is needed only to continue into a native writer.
                terminal = torch.empty_like(current) if position + 1 < len(runs) else None
                predecessors, columns = replay_interval(
                    current,
                    tapes,
                    prepare_columns=tuple(t.prepare_columns for t in run),
                    terminal_output=terminal,
                )
                restored.extend(predecessors[1:])
                restored.append(terminal)
                for transition, column in zip(run, columns):
                    transition.column = column
                current = terminal
            ready = torch.cuda.Event()
            ready.record(stream)
            for value, weak in zip(restored, self.outputs):
                slot = weak()
                if slot is not None and not slot.resident():
                    slot.restore(value, ready)
            self.ready = ready
            self.materialized = True


class ResidualReplayManager:
    """Keep periodic physical-layer checkpoints and replay the writes between them."""

    def __init__(self, interval):
        self.interval = interval
        self.layers = 0
        self.last = self.active = None
        self.slots = WeakValueDictionary()

    def slot(self, tensor, *, output=False):
        """Find views of one StorageImpl without retaining unused tensors."""
        key = tensor.untyped_storage()._cdata
        slot = self.slots.get(key)
        if slot is None:
            slot = ResidualSlot(tensor, anchor=not output)
            self.slots[key] = slot
        return slot

    def advance(self, physical_layers):
        """Start an atomic unit of physical layers, sealing first if it would cross an interval.

        Intervals count physical layers, not writes.
        """
        if self.layers and self.layers + physical_layers > self.interval:
            self.seal()
        self.layers += physical_layers

    def register(
        self, previous, current, saved=None, *, prepare_columns=False, native=None, dependencies=()
    ):
        """Record one write: a fused ``SavedWrite`` tape, or a ``native`` replay function."""
        for tensor in (previous, current):
            # Callers validated contiguous CUDA inputs; resize_(0) must free the whole storage.
            if (
                tensor.dtype not in (torch.float32, torch.bfloat16)
                or tensor.dtype != previous.dtype
                or tensor.storage_offset()
                or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
            ):
                raise ValueError(
                    "checkpoint replay requires full FP32 or BF16 storage of one dtype"
                )
        old, new = self.slot(previous), self.slot(current, output=True)
        if old is new or (self.last is not None and old is not self.last):
            raise ValueError("replay writes must form one connected chain with distinct outputs")
        if self.active is None:
            old.anchor = True
            self.active = _ReplayInterval(old)
        index = len(self.active.transitions)
        self.active.transitions.append(_Transition(saved, native, dependencies, prepare_columns))
        self.active.outputs.append(ref(new))
        new.provider = self.active
        if not old.anchor:
            # Free the predecessor now; its successor's write was its last forward use.
            stream = torch.cuda.current_stream(previous.device)
            (old.backing if old.backing is not None else previous).record_stream(stream)
            previous.untyped_storage().resize_(0)
            old.backing = old.ready = None
        self.last = new
        return ResidualLink(old, new, self.active, index)

    def interrupt(self):
        """End the chain before a write that cannot be replayed (autograd saves it).

        The next registered write starts a new interval anchored at its own input.
        """
        self.seal()
        self.last = None

    def seal(self):
        """Publish the last write as the next interval's checkpoint."""
        if self.active is not None:
            self.active.sealed = True
            self.active.forward_ready = self.last.ready
            self.last.anchor = True
            self.last.provider = None
            self.active = None
        self.layers = 0


@contextmanager
def residual_replay_scope(config, training):
    """Create an independent forward-replay owner for each HybridStack call."""
    interval = config.residual_stream_recompute_num_layers
    enabled = (
        config.moe_shortcut_wide_residual_norm_fusion
        and training
        and torch.is_grad_enabled()
        and interval is not None
        and interval > 1
    )
    manager = ResidualReplayManager(interval) if enabled else None
    token = _ACTIVE.set(manager)
    try:
        yield manager
        if manager is not None:
            manager.seal()
    finally:
        _ACTIVE.reset(token)
        if manager is not None:
            manager.last = manager.active = None
