# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Decode metadata slots + depth-2 rings for async-scheduling (launch-before-commit).

The launch-before-commit pipeline overlaps two decode forwards: while forward ``F(K)``
runs against one GPU metadata buffer, the next step's metadata is *prestaged* into a
**free** buffer so that, after sampling, only a token-scatter + launch separate
``S(K)`` from ``F(K+1)``. That requires two GPU metadata buffers ("slots"):

- ``slot_current`` -- the buffer the running/consumed forward observes.
- ``slot_child``   -- the free buffer the next step is prestaged into.

This module provides the slot/ring abstractions. The slots wrap
:class:`~megatron.core.inference.contexts.gpu_view.ContextGPUView` buffers (whose
addresses are fixed for CUDA-graph compatibility); a forward is bound to exactly one
slot. A slot may not be re-staged until its H2D and forward fences have retired, so a
still-in-flight forward never observes a buffer being overwritten.

The serial (``enable_async_scheduling`` off) decode path uses a single live
``gpu_view`` and never constructs slots, so its behavior is unchanged. The context
constructs ``slot_current`` (wrapping the live ``gpu_view``) and a free ``slot_child``
only when the flag is on; the prestage that builds metadata into ``slot_child`` and the
graph replay that consumes it land in later commits.
"""

from typing import Any, Callable, List, Optional


class DecodeMetadataSlot:
    """A bindable GPU decode-metadata buffer plus its two CUDA fences and identity.

    Wraps a ``ContextGPUView``. Slot reuse (re-staging the next step's metadata into it)
    is gated by ``is_free()``: the slot must not be in use and both of its fences (the
    coalesced-H2D fence and the forward fence) must have completed. This is the slot-level
    expression of the design's "a slot can't be reused before its H2D/forward events
    retire" rule.
    """

    def __init__(self, slot_id: int, gpu_view: Any):
        self.slot_id = slot_id
        self.gpu_view = gpu_view
        # Signalled when the coalesced bookkeeping H2D into this slot completes.
        self.h2d_done_event: Any = None
        # Signalled when the forward bound to this slot completes.
        self.forward_done_event: Any = None
        self._in_use = False

    @property
    def base_ptr(self) -> int:
        """Base address of the slot's backing metadata buffer (fixed for graph capture)."""
        return self.gpu_view._buf.data_ptr()

    def fences_retired(self) -> bool:
        """True iff neither fence is pending (a pending fence means a forward may still read)."""
        for event in (self.h2d_done_event, self.forward_done_event):
            if event is not None and not event.query():
                return False
        return True

    def is_free(self) -> bool:
        """True iff the slot can be re-staged: not in use and both fences retired."""
        return not self._in_use and self.fences_retired()

    def acquire(self) -> None:
        """Mark the slot in use for a new transaction. Must be free first."""
        assert self.is_free(), (
            f"slot {self.slot_id} acquired while busy "
            f"(in_use={self._in_use}, fences_retired={self.fences_retired()})"
        )
        self._in_use = True

    def release(self) -> None:
        """Release the slot back to the free pool (its forward has been consumed)."""
        self._in_use = False

    def reset_fences(self) -> None:
        """Clear both fences (e.g. on a barrier/drain that already synchronized them)."""
        self.h2d_done_event = None
        self.forward_done_event = None

    def graph_slot_key(self) -> int:
        """Slot identity for the CUDA-graph key.

        When a captured graph references this slot's buffers, the key must encode the
        slot's buffer address so a replay never targets the wrong slot's metadata. When
        the key cannot carry slot identity, :func:`assert_slot_pointer_identity` guards
        replay instead.
        """
        return self.base_ptr


def assert_slot_pointer_identity(slot: DecodeMetadataSlot, captured_base_ptr: int) -> None:
    """Assert ``slot``'s buffer still lives where a graph was captured against it.

    Replaying a CUDA graph against a metadata buffer that moved would read stale
    pointers; this guard turns that into a loud failure instead of silent corruption.
    """
    assert slot.base_ptr == captured_base_ptr, (
        f"decode metadata slot {slot.slot_id} buffer moved: graph captured against "
        f"{captured_base_ptr:#x} but slot is now at {slot.base_ptr:#x}; replaying would "
        f"read stale metadata pointers"
    )


class DepthTwoRing:
    """A depth-2 ring of per-step handles (H2D staging / logits / sample buffers).

    Two forwards overlap in the launch-before-commit pipeline, so staging buffers and
    output handles need depth 2: step ``K`` and step ``K+1`` must not share a buffer.
    Index by step (or any monotonically increasing counter); the ring selects by parity.
    """

    def __init__(self, factory: Callable[[int], Any]):
        # Materialize both entries up front so addresses are fixed (graph-friendly).
        self._entries: List[Any] = [factory(0), factory(1)]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, step: int) -> Any:
        return self._entries[step % 2]

    def current(self, step: int) -> Any:
        """The entry bound to ``step``."""
        return self._entries[step % 2]

    def other(self, step: int) -> Any:
        """The entry NOT bound to ``step`` (the one ``step+1`` will use)."""
        return self._entries[(step + 1) % 2]


def make_decode_metadata_slots(
    gpu_view: Any, child_gpu_view_factory: Callable[[], Any]
) -> "tuple[DecodeMetadataSlot, DecodeMetadataSlot]":
    """Build ``(slot_current, slot_child)`` for async scheduling.

    ``slot_current`` wraps the live ``gpu_view`` (the same buffer the serial path uses),
    so binding the running forward to ``slot_current`` is identical to the serial path.
    ``slot_child`` wraps a freshly-allocated metadata buffer that the prestage builds into.
    """
    slot_current = DecodeMetadataSlot(slot_id=0, gpu_view=gpu_view)
    slot_child = DecodeMetadataSlot(slot_id=1, gpu_view=child_gpu_view_factory())
    return slot_current, slot_child
