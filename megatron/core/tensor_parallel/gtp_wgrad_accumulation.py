# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Track when a GTP output head has received all gradients in an accumulation window."""

from dataclasses import dataclass
from enum import Enum, auto


class _Phase(Enum):
    # COUNTING --finish_forward()--> FORWARD_COMPLETE --last backward--> COMPLETE
    #    |                                |
    #    | no-grad forward                | any forward
    #    +------------> FENCE_ONLY <-------+
    #
    # FENCE_ONLY waits for the final fence; COMPLETE allows early reduction.
    # reset() after the final fence: any state -> COUNTING.
    COUNTING = auto()  # More forwards may arrive.
    FORWARD_COMPLETE = auto()  # The caller declared the last forward.
    FENCE_ONLY = auto()  # Forward calls cannot reliably predict backward contributions.
    COMPLETE = auto()  # All declared contributions arrived; early reduction is due.


@dataclass
class WgradAccumulationWindow:
    """One outstanding-use count, with an explicit fallback to the final GTP fence.

    Each counted forward adds one pending backward; each backward removes one.
    Early reduction is allowed only after the caller closes the forward window
    and no backwards remain. Unexpected or grad-disabled forwards select the
    fence fallback for the rest of this window. No counts carry across windows.
    """

    _pending_backwards: int = 0
    _phase: _Phase = _Phase.COUNTING

    def record_forward(self, *, grad_enabled: bool) -> None:
        """Record a use, or fall back when it cannot be counted reliably."""
        if self._phase is _Phase.COMPLETE:
            raise RuntimeError("GTP forward after the accumulated gradient was reduced")
        if not grad_enabled or self._phase is _Phase.FORWARD_COMPLETE:
            # A no-grad use may be checkpointed; a late use may be recomputation.
            self._phase = _Phase.FENCE_ONLY
        if self._phase is _Phase.COUNTING:
            self._pending_backwards += 1

    def finish_forward(self) -> None:
        """Declare that no more forwards belong to this accumulation window."""
        if self._phase is _Phase.COUNTING:
            self._phase = _Phase.FORWARD_COMPLETE

    def record_backward(self) -> None:
        """Record a contribution before its GEMM writes the accumulation buffer."""
        if self._phase is _Phase.COMPLETE or (
            self._phase is _Phase.FORWARD_COMPLETE and self._pending_backwards <= 0
        ):
            raise RuntimeError("GTP backward exceeded the declared forward count")
        if self._phase is _Phase.FENCE_ONLY:
            return
        self._pending_backwards -= 1
        if self._phase is _Phase.FORWARD_COMPLETE and self._pending_backwards == 0:
            self._phase = _Phase.COMPLETE

    @property
    def ready_to_reduce(self) -> bool:
        """Whether the final declared backward contribution has arrived."""
        return self._phase is _Phase.COMPLETE

    def reset(self) -> None:
        """Start a fresh window after the final GTP fence."""
        self._pending_backwards = 0
        self._phase = _Phase.COUNTING
