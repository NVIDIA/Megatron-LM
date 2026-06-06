# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single fixed-address GPU decode-metadata buffer + CPU staging double-buffer.

The launch-before-commit pipeline overlaps two decode forwards: while forward ``F(K)``
runs, the next step's metadata is *prestaged* so that, after sampling, only an in-place
token-scatter + launch separate ``S(K)`` from ``F(K+1)``.

A captured CUDA decode graph reads metadata by **ABSOLUTE ADDRESS**
(``cuda_graphs.py:605-639``: only the embed-input / hidden-states surface is
copy-on-replay; every *other* metadata field is read from the exact address captured), so
the metadata GPU buffer cannot be a swappable child. There is therefore exactly **one**
fixed-address GPU metadata buffer -- the live ``gpu_view`` -- which is **never reallocated
or swapped**, only updated in place between steps and read by the captured graph at the
addresses it captured.

The genuine double-buffer is consequently **CPU-side**: the next step's bookkeeping is
prestaged into a CPU *staging* buffer while the current forward's already-issued H2D still
reads the *live* CPU buffer, then the staging buffer is swapped in and a single coalesced
H2D copies it into the one GPU buffer. :class:`DepthTwoRing` provides that depth-2 CPU
staging<->live double-buffer (and the per-step output handles two overlapping forwards
must not share).

Replay is only valid against the buffer the graph captured, so
:class:`DecodeMetadataBuffer` records the GPU buffer's address (and the cached
input/position view addresses) and :meth:`DecodeMetadataBuffer.assert_pointer_invariant`
asserts they never move across decode steps -- turning an accidental reallocation (which
would make a replay read freed/foreign memory) into a loud failure instead of silent
corruption.

The serial (``enable_async_scheduling`` off) decode path uses the single live ``gpu_view``
directly and never constructs any of these helpers, so its behavior is unchanged. The
prestage that builds the next step's metadata into the CPU staging buffer and the graph
replay that consumes the GPU buffer land in later commits.
"""

from typing import Any, Callable, List, Optional


class DecodeMetadataBuffer:
    """The single fixed-address GPU decode-metadata buffer plus its two CUDA fences.

    Wraps the live ``ContextGPUView``. There is exactly one such buffer; it is never
    reallocated or swapped (a captured decode graph reads its fields by absolute address),
    only updated in place between steps. Reuse for the next step is gated by
    :meth:`is_free`: both fences (the coalesced-H2D fence and the forward fence) must have
    retired, so a still in-flight forward never observes the buffer being overwritten
    mid-flight.

    The buffer also records the address it (and its cached input/position views) was first
    observed at; :meth:`assert_pointer_invariant` asserts those addresses never move,
    because a captured CUDA graph may only be replayed against the exact buffer it
    captured. This is the single-buffer replacement for the (discarded) two-GPU-buffer
    "slot" model: there is no ``slot_child`` and no replay-against-child.
    """

    def __init__(self, gpu_view: Any):
        self.gpu_view = gpu_view
        # Signalled when the coalesced bookkeeping H2D into this buffer completes.
        self.h2d_done_event: Any = None
        # Signalled when the forward reading this buffer completes.
        self.forward_done_event: Any = None
        # Invariant addresses, recorded lazily on first observation. A captured graph is
        # only valid against these exact addresses.
        self._captured_base_ptr: Optional[int] = None
        self._captured_input_ids_ptr: Optional[int] = None
        self._captured_pos_ids_ptr: Optional[int] = None

    @property
    def base_ptr(self) -> int:
        """Base address of the single backing metadata buffer (fixed for graph capture)."""
        return self.gpu_view._buf.data_ptr()

    def fences_retired(self) -> bool:
        """True iff neither fence is pending (a pending fence means a forward may still read)."""
        for event in (self.h2d_done_event, self.forward_done_event):
            if event is not None and not event.query():
                return False
        return True

    def is_free(self) -> bool:
        """True iff the buffer can be updated in place for the next step (both fences retired)."""
        return self.fences_retired()

    def reset_fences(self) -> None:
        """Clear both fences (e.g. on a barrier/drain that already synchronized them)."""
        self.h2d_done_event = None
        self.forward_done_event = None

    def capture_pointers(
        self, input_ids_ptr: Optional[int] = None, pos_ids_ptr: Optional[int] = None
    ) -> None:
        """Record the invariant addresses (call once, after the decode graph is captured).

        Records the GPU metadata buffer's base address and, when supplied, the cached
        input/position view addresses. Idempotent for already-recorded slots.
        """
        self._captured_base_ptr = self.base_ptr
        if input_ids_ptr is not None:
            self._captured_input_ids_ptr = input_ids_ptr
        if pos_ids_ptr is not None:
            self._captured_pos_ids_ptr = pos_ids_ptr

    def assert_pointer_invariant(
        self, input_ids_ptr: Optional[int] = None, pos_ids_ptr: Optional[int] = None
    ) -> None:
        """Assert the single GPU buffer (and cached input/pos views) have not moved.

        Replaying a captured decode graph against a metadata buffer that moved would read
        stale/foreign memory; this guard turns that into a loud failure. The first call
        records the baseline addresses; every subsequent call asserts invariance against
        them. This is the single-buffer ``data_ptr()``-invariance guard that replaces the
        two-slot pointer-identity check.
        """
        if self._captured_base_ptr is None:
            self.capture_pointers(input_ids_ptr, pos_ids_ptr)
            return
        assert self.base_ptr == self._captured_base_ptr, (
            f"decode metadata GPU buffer moved: graph captured against "
            f"{self._captured_base_ptr:#x} but buffer is now at {self.base_ptr:#x}; "
            f"replaying would read stale/foreign metadata. The single GPU metadata buffer "
            f"must never be reallocated or swapped."
        )
        if input_ids_ptr is not None and self._captured_input_ids_ptr is not None:
            assert input_ids_ptr == self._captured_input_ids_ptr, (
                f"cached input_ids view moved: captured {self._captured_input_ids_ptr:#x} "
                f"but now {input_ids_ptr:#x}"
            )
        if pos_ids_ptr is not None and self._captured_pos_ids_ptr is not None:
            assert pos_ids_ptr == self._captured_pos_ids_ptr, (
                f"cached pos_ids view moved: captured {self._captured_pos_ids_ptr:#x} "
                f"but now {pos_ids_ptr:#x}"
            )


class DepthTwoRing:
    """A depth-2 ring for the CPU staging<->live double-buffer (and per-step handles).

    Two forwards overlap in the launch-before-commit pipeline, so the next step's metadata
    must be prestaged into a CPU buffer that is *not* the one the current forward's
    already-issued H2D is reading. Depth 2 is sufficient: step ``K`` and step ``K+1`` use
    opposite entries, selected by parity. The same ring also backs per-step output handles
    (logits / sample buffers) two overlapping forwards must not share.

    Note: the ring's entries are **CPU-side** (staging vs live bookkeeping buffers / output
    handles). The GPU metadata buffer is single and fixed-address (see
    :class:`DecodeMetadataBuffer`) -- it is deliberately NOT ring-buffered, because a
    captured decode graph reads it by absolute address.
    """

    def __init__(self, factory: Callable[[int], Any]):
        # Materialize both entries up front so addresses are fixed (graph-friendly).
        self._entries: List[Any] = [factory(0), factory(1)]

    @classmethod
    def of(cls, first: Any, second: Any) -> "DepthTwoRing":
        """Build a ring from two already-constructed entries (e.g. live + staging buffers)."""
        ring = cls.__new__(cls)
        ring._entries = [first, second]
        return ring

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
