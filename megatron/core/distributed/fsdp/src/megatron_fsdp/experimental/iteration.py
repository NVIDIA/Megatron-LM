# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Record and replay demand occurrences over a complete FSDP iteration."""

from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from itertools import islice
from typing import TYPE_CHECKING, Literal
from weakref import ref

if TYPE_CHECKING:
    from .context import FsdpContext
    from .module import FsdpModule

DemandPhase = Literal["forward", "backward", "recompute", "none"]


class IterationOrder:
    """Weak demand sequences, with one strict cursor for the active iteration.

    Forward and backward occurrences share a sequence so prefetch follows the
    actual VPP/combined schedule. Only successful recordings are cached. This
    state lives on FsdpContext rather than a ContextVar: autograd hooks may run
    on another thread. A context cannot execute concurrent iteration scopes.
    """

    def __init__(self) -> None:
        self._orders: dict[Hashable, tuple[tuple[ref, DemandPhase], ...]] = {}
        self._events: list[tuple[ref, DemandPhase]] | None = None
        self._recording = False
        self._position = 0

    @property
    def active(self) -> bool:
        """Whether demand hooks should use iteration scheduling."""
        return self._events is not None

    @contextmanager
    def iteration(self, key: Hashable, *, rerecord: bool = False) -> Iterator[None]:
        """Record a new schedule or strictly replay a previously completed one.

        Use distinct keys for training/validation and scheduling configurations.
        Failed iterations invalidate their cached order; callers must handle the
        failed computation before attempting another iteration.
        """
        if self.active:
            raise RuntimeError("FSDP iteration scopes cannot overlap on the same context.")
        self._recording = rerecord or key not in self._orders
        self._events = [] if self._recording else list(self._orders[key])
        self._position = 0
        try:
            yield
            if not self._recording and self._position != len(self._events):
                raise RuntimeError(
                    "FSDP iteration ended before consuming its recorded demand order."
                )
            if self._recording:
                self._orders[key] = tuple(self._events)
        except BaseException:
            self._orders.pop(key, None)
            raise
        finally:
            self._events = None
            self._recording = False
            self._position = 0

    def advance(self, module: "FsdpModule", phase: DemandPhase) -> bool:
        """Record/validate a demand before communication; return whether scoped."""
        if self._events is None:
            return False
        if self._recording:
            self._events.append((ref(module), phase))
        else:
            if self._position >= len(self._events):
                raise RuntimeError("FSDP demand exceeded the recorded iteration order.")
            expected, expected_phase = self._events[self._position]
            if expected() is not module or expected_phase != phase:
                raise RuntimeError(
                    "FSDP module calls diverged from the recorded iteration order "
                    f"at occurrence {self._position} ({phase}, expected {expected_phase})."
                )
        self._position += 1
        return True

    def prefetch(self, budget: int | None) -> None:
        """Gather upcoming owners without advancing or recording demand.

        Skip materialized/repeated owners within this lookahead. Prefetch is
        speculative: an intervening reshard may invalidate it, and demand must
        still gather/wait normally. Release and gradient readiness stay with
        the existing FSDP hooks.
        """
        if self._events is None or self._recording or budget == 0:
            return
        size = 0
        seen = set()
        for module_ref, _phase in islice(self._events, self._position, None):
            module = module_ref()
            if module is None:
                raise RuntimeError("A module in the recorded FSDP iteration order was destroyed.")
            if module in seen or module._unshard_event is not None:
                continue
            seen.add(module)
            module._unshard_parameter_groups()
            size += module.num_parameter_elements
            if budget is None or size >= budget:
                break


@contextmanager
def iteration(
    context: "FsdpContext", *, key: Hashable = "default", rerecord: bool = False
) -> Iterator[None]:
    """Scope one complete iteration, including every local VPP model chunk.

    All ranks sharing a data-parallel mesh must use the same stable schedule.
    This scope owns demand recording only, not backward completion or optimizer
    synchronization. Call finish_grad_sync after all gradient producers as usual.
    """
    context.ensure_finalized()
    with context.iteration_order.iteration(key, rerecord=rerecord):
        yield
