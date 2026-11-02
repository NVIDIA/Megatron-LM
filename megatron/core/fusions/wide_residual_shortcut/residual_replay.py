# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint forward replay and call-local residual storage ownership.

Intervals count physical layers; their transition lists count actual writes.
Each interval owns independent allocations, never overwritten by another interval.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from weakref import WeakValueDictionary, ref

import torch

_ACTIVE = ContextVar("wide_residual_replay", default=None)


def current_manager():
    """Return this forward's owner, excluding explicit no-grad regions."""
    return _ACTIVE.get() if torch.is_grad_enabled() else None


def watch_residual(tensor):
    """Record a native reader's storage dependency inside autograd forward."""
    manager = _ACTIVE.get()
    return None if manager is None else manager.slot(tensor)


def register_write(previous, current, update, write, retain, streams, max_forget):
    """Retain the exact native forward implementation and its detached inputs."""
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
    """Preserve a StorageImpl and the allocator-owned backing across consumers."""

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
            # share_storage wraps the DataPtr. Record the original allocator's
            # DataPtr too, including retained backwards on a different stream.
            (self.backing if self.backing is not None else self.tensor).record_stream(stream)
            return self.tensor

    def restore(self, value, ready):
        """Reattach a full zero-offset allocation without changing tensor identity."""
        if value.storage_offset() or value.untyped_storage().nbytes() != value.numel() * 4:
            raise ValueError("replay restoration requires full zero-offset FP32 storage")
        from megatron.core.tensor_parallel.random import _get_share_storage

        with self.lock:
            self.backing, self.ready = value, ready
            _get_share_storage()(self.tensor, value)
            self.provider = None

    def without_replay_owner(self):
        """Keep stream/lifetime metadata for retained autograd, not interval tapes."""
        with self.lock:
            if not self.resident():
                raise RuntimeError("cannot detach an unmaterialized replay slot")
            return ResidualSlot(self.tensor, backing=self.backing, ready=self.ready)


def retained_slot(slot):
    """Retained graphs keep allocation metadata; ordinary backward frees owners."""
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
    """One write's detached storage handles and interval position."""

    previous_slot: ResidualSlot
    current_slot: ResidualSlot
    interval: object
    index: int

    def columns(self):
        """Fetch the original replay-produced reader-zero operand, without a copy."""
        self.interval.materialize()
        stream = torch.cuda.current_stream(self.previous_slot.tensor.device)
        stream.wait_event(self.interval.ready)
        column = self.interval.transitions[self.index].column
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
        """Replay on the requesting stream, serially before this interval backward.

        Independent storage makes pruned and cross-stream traversal safe. No
        prefetch or pooling is implicit: this is the qualified serial schedule.
        """
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
            index = 0
            while index < len(self.transitions):
                transition = self.transitions[index]
                if transition.native is not None:
                    for tensor in transition.dependencies:
                        tensor.record_stream(stream)
                    current = transition.native(current)
                    restored.append(current)
                    index += 1
                    continue
                end = index + 1
                while end < len(self.transitions) and self.transitions[end].native is None:
                    end += 1
                tapes = tuple(t.saved for t in self.transitions[index:end])
                for tape in tapes:
                    for tensor in (tape.update, tape.coefficients, *tape.gammas, *tape.rstd):
                        tensor.record_stream(stream)
                current = current.view(-1, tapes[0].num_streams * tapes[0].update.shape[-1])
                terminal = torch.empty_like(current) if end < len(self.transitions) else None
                predecessors, columns = replay_interval(
                    current,
                    tapes,
                    prepare_columns=tuple(t.prepare_columns for t in self.transitions[index:end]),
                    terminal_output=terminal,
                )
                if len(predecessors) != len(tapes) or len(columns) != len(tapes):
                    raise RuntimeError("replay returned an incorrect transition count")
                # A run returns R[index]...R[end-1]. Its last output R[end]
                # is needed only to continue into a native writer.
                restored.extend(predecessors[1:])
                restored.append(terminal)
                for transition, column in zip(self.transitions[index:end], columns):
                    transition.column = column if transition.prepare_columns else None
                current = terminal
                index = end
            ready = torch.cuda.Event()
            ready.record(stream)
            for value, weak in zip(restored, self.outputs):
                slot = weak()
                if slot is not None and not slot.resident():
                    if value is None:
                        raise RuntimeError("a checkpoint terminal was unexpectedly released")
                    slot.restore(value, ready)
            self.ready = ready
            self.materialized = True


class ResidualReplayManager:
    """Keep periodic physical-layer checkpoints and replay actual write intervals."""

    def __init__(self, interval):
        self.interval = interval
        self.layers = 0
        self.last = self.active = None
        self.slots = WeakValueDictionary()

    def slot(self, tensor, *, output=False):
        """Find views of one original StorageImpl without retaining unused tensors."""
        key = tensor.untyped_storage()._cdata
        slot = self.slots.get(key)
        if slot is None:
            slot = ResidualSlot(tensor, anchor=not output)
            self.slots[key] = slot
        return slot

    def advance(self, physical_layers):
        """Start an atomic compute/MoE unit, sealing first if it would cross an interval.

        Physical layers need not equal the number of writes. A full interval is
        sealed here, before the next unit's first write, or when the scope exits.
        """
        if self.layers and self.layers + physical_layers > self.interval:
            self.seal()
        self.layers += physical_layers

    def register(
        self, previous, current, saved=None, *, prepare_columns=False, native=None, dependencies=()
    ):
        """Record one write: a fused tape of detached producer tensors (``saved``), or a
        ``native`` writer replayed without changing its arithmetic or contraction."""
        for tensor in (previous, current):
            if (
                tensor.dtype != torch.float32
                or not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.storage_offset()
                or tensor.untyped_storage().nbytes() != tensor.numel() * 4
            ):
                raise ValueError("checkpoint replay requires full contiguous FP32 CUDA storage")
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
            stream = torch.cuda.current_stream(previous.device)
            (old.backing if old.backing is not None else previous).record_stream(stream)
            previous.untyped_storage().resize_(0)
            old.backing = old.ready = None
        self.last = new
        return ResidualLink(old, new, self.active, index)

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
