# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execution-order tracer and prefetch planner for fine-grained FSDP.

The combined-1F1B + VPP schedule is occurrence-based: the same FSDP unit can
be consumed in forward and backward, model chunks interleave, and
warmup/steady/cooldown differ per pipeline rank. The static
``forward_order`` / ``backward_order`` sequences cannot express that runtime
path, so a per-context runner traces the real execution and replays it to
drive prefetch.

Two cooperating paths:

- **Trace path**: during the first global batch, every fine-grained
  execution event (consume, reshard) is recorded in actual order. No
  prefetch is issued and no reshard is optimized away.
- **Optimization path**: from the second batch, each real op is validated
  against the traced cycle and translated into optimization directives:
  a prefetch target after an unshard, and a skip-reshard decision when the
  traced schedule re-unshards the same module with the same orientation
  immediately.

State machine::

    TRACING --(complete_trace)--> REPLAYING --(divergence)--> TRACING
       ^                                                          |
       +----------------------(complete_trace)--------------------+

Four per-op interfaces:

- ``record_unshard(module, orientation)`` / ``record_reshard(module)``:
  trace path — record the real event, or during replay validate it against
  the traced cycle and advance the cursor.
- ``suggest_prefetch()``: optimization path — the next traced unshard
  (skipping reshard events) to all-gather ahead, or ``None`` while tracing.
- ``suggest_skip_reshard(module)``: optimization path — whether this reshard
  can be skipped because the next traced unshard reuses the same module and
  orientation, keeping the storage resident.
"""

import dataclasses
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

# Forward reference; FsdpModule is imported lazily to avoid a cycle.
if TYPE_CHECKING:
    from .module import FsdpModule

logger = logging.getLogger(__name__)


class RunnerPhase(Enum):
    """Lifecycle phase of an :class:`FsdpExecutionRunner`."""

    TRACING = auto()
    REPLAYING = auto()


class EventKind(Enum):
    """Kind of an execution event on the trace path."""

    UNSHARD = auto()
    RESHARD = auto()


@dataclasses.dataclass(frozen=True)
class RunnerEvent:
    """One fine-grained execution event.

    Attributes:
        kind: Whether the module's parameters are consumed or resharded.
        module: The FSDP module the event applies to.
        orientation: Payload orientation (``"rowwise"`` forward,
            ``"colwise"`` backward); ``None`` for reshard events.
    """

    kind: EventKind
    module: "FsdpModule"
    orientation: str | None = None


class FsdpExecutionRunner:
    """Record the fine-grained execution and plan prefetches.

    The runner is owned by an :class:`FsdpContext` and driven by the
    fine-grained unshard/reshard entry points plus the global-batch boundary
    signaled by the training loop. It never decides compute order — it only
    observes events and, during replay, suggests what to prefetch and which
    reshards to skip.
    """

    def __init__(self, context, *, use_trace_replay: bool = False) -> None:
        """Create a runner in the tracing phase.

        Args:
            context: The owning :class:`FsdpContext`, used for the static
                orders in default mode.
            use_trace_replay: Enable trace-and-replay prefetch.
        """
        self._context = context
        self._use_trace_replay = use_trace_replay
        self._phase = RunnerPhase.TRACING
        self._trace: list[RunnerEvent] = []
        self._replay_index = 0
        self._cycles_observed = 0
        # Modules consumed in the current round. The fine-grained schedule
        # fires one hook per sub-module (dense, experts), so the same module
        # can be recorded several times within a round; only the first is a
        # real unshard. Cleared by record_reshard() and at the batch boundary.
        self._consumed_this_round: set[FsdpModule] = set()
        # Suggested all-gathers and skipped reshards leave orientation-specific
        # materialized storage resident until the expected consumer arrives.
        # Keep only weak ownership so tracing state cannot retain deleted chunks.
        self._pending_prefetches: WeakKeyDictionary[FsdpModule, str] = WeakKeyDictionary()
        # Orientation of each module's most recent consume during replay,
        # used to decide whether a reshard can be skipped (storage only needs
        # to stay resident for an immediate same-orientation re-unshard).
        self._last_orientation: dict[FsdpModule, str] = {}
        # Diagnostics: how many events were validated during replay, how many
        # diverged (re-trace), and how many complete_trace calls ran.
        self._replayed_occurrences = 0
        self._divergences = 0
        self._complete_trace_calls = 0
        if use_trace_replay:
            logger.info("FsdpExecutionRunner: trace-and-replay prefetch enabled.")

    @property
    def phase(self) -> RunnerPhase:
        """Current runner phase."""
        return self._phase

    @property
    def is_tracing(self) -> bool:
        """Whether the runner is recording a fresh cycle."""
        return self._phase is RunnerPhase.TRACING

    @property
    def use_trace_replay(self) -> bool:
        """Whether trace-and-replay prefetch is enabled."""
        return self._use_trace_replay

    # ------------------------------------------------------------------
    # Interface 1: record execution events (consume, reshard)
    # ------------------------------------------------------------------

    def record_unshard(self, module: "FsdpModule", orientation: str) -> None:
        """Record (tracing) or validate (replay) an unshard event.

        The fine-grained schedule fires one hook per sub-module (dense,
        experts), so the same module can arrive several times within a round;
        only the first arrival is a real unshard for the trace. Call
        this before materializing the requested orientation, then call
        ``suggest_prefetch()`` after the materialization to get the next
        target (replay only).

        Args:
            module: The FSDP module being unsharded for compute.
            orientation: Payload orientation (``"rowwise"`` forward,
                ``"colwise"`` backward).
        """
        prefetched_orientation = self._pending_prefetches.pop(module, None)
        if prefetched_orientation is not None and prefetched_orientation != orientation:
            # A schedule divergence can consume a prefetched module with the
            # opposite MXFP8 payload orientation. Drop the stale materialization
            # before the caller attempts its demand all-gather.
            module._release_unsharded_parameter_groups()
        if not self._use_trace_replay:
            return
        if module in self._consumed_this_round:
            return
        self._consumed_this_round.add(module)
        self._last_orientation[module] = orientation
        self._validate_and_advance(EventKind.UNSHARD, module, orientation)

    def record_reshard(self, module: "FsdpModule") -> None:
        """Record (tracing) or validate (replay) a reshard event.

        The reshard ends the module's current unshard round: it clears the
        per-round dedup entry so the next unshard of the same module (e.g. the
        backward pass after the forward pass) records a fresh event. Call
        ``suggest_skip_reshard(module)`` right after this returns to learn
        whether the actual reshard can be skipped (replay only).

        Args:
            module: The FSDP module whose unsharded storage is released.
        """
        self._pending_prefetches.pop(module, None)
        if not self._use_trace_replay:
            return
        # The reshard ends the module's unshard round; discard its dedup
        # entry so the next unshard (e.g. backward after forward) records a
        # fresh event.
        self._consumed_this_round.discard(module)
        self._validate_and_advance(EventKind.RESHARD, module, None)

    # ------------------------------------------------------------------
    # Interface 2: prefetch suggestion
    # ------------------------------------------------------------------

    def suggest_prefetch(
        self, module: "FsdpModule", orientation: str
    ) -> tuple["FsdpModule", str] | None:
        """Return the next module to all-gather ahead of this unshard.

        In default mode, resolves the static ``forward_order`` /
        ``backward_order`` successor. In trace-replay mode, returns the next
        traced unshard (skipping reshard events) with its recorded
        orientation.

        Args:
            module: The FSDP module just unsharded for compute.
            orientation: Payload orientation (``"rowwise"`` forward,
                ``"colwise"`` backward).

        Returns:
            ``(module, orientation)`` to prefetch, or ``None`` while tracing,
            after a divergence, or at the end of the static order.
        """
        if not self._use_trace_replay:
            prefetch = self._static_successor(module, orientation)
            if prefetch is not None:
                self._pending_prefetches[prefetch[0]] = prefetch[1]
            return prefetch
        # Tracing and divergence (re-trace) both disable prefetch; only a
        # validated replay cycle suggests a prefetch target.
        if self._phase is not RunnerPhase.REPLAYING or not self._trace:
            return None
        # Do not prefetch the first operation of the next global step before
        # complete_trace() resets the replay cursor. Scan only the remaining
        # suffix so trailing reshard events cannot cross that boundary.
        if self._replay_index >= len(self._trace):
            return None
        for event in self._trace[self._replay_index :]:
            if event.kind is EventKind.UNSHARD:
                assert event.orientation is not None
                self._pending_prefetches[event.module] = event.orientation
                return event.module, event.orientation
        return None

    # ------------------------------------------------------------------
    # Interface 3: reshard-skip suggestion
    # ------------------------------------------------------------------

    def suggest_skip_reshard(self, module: "FsdpModule") -> bool:
        """Return whether the reshard of ``module`` can be skipped.

        The optimization path: if the traced schedule immediately re-unshards
        the same module with the same orientation right after this reshard,
        the reshard is unnecessary — the storage can stay resident and the
        following all-gather can be skipped. Returns whether to skip the
        reshard.

        Args:
            module: The FSDP module whose unsharded storage is released.

        Returns:
            True to skip the actual reshard (keep storage resident), False to
            reshard normally.
        """
        if not self._use_trace_replay or self._phase is RunnerPhase.TRACING:
            return False
        if not self._trace or self._replay_index >= len(self._trace):
            return False
        next_event = self._trace[self._replay_index]
        skip_reshard = (
            next_event.kind is EventKind.UNSHARD
            and next_event.module is module
            and next_event.orientation == self._last_orientation.get(module)
        )
        if skip_reshard:
            assert next_event.orientation is not None
            self._pending_prefetches[module] = next_event.orientation
        return skip_reshard

    # ------------------------------------------------------------------
    # Lifecycle: batch boundary
    # ------------------------------------------------------------------

    def complete_trace(self) -> None:
        """Compile the recorded trace into the replay cycle.

        Called at every global-batch boundary before the optimizer mutates parameters.
        The first batch (with a non-empty trace) transitions to ``REPLAYING``;
        subsequent calls reset the replay cursor for the next batch while keeping the
        compiled cycle.
        """
        released_prefetch = self._release_abandoned_prefetches()
        if released_prefetch:
            # Raw release is queued on the all-gather stream. Optimizer writes happen on
            # the current stream immediately after this callback, so fence them from any
            # stale full-weight storage before the sharded source weights are updated.
            self._context.current_stream().wait_stream(self._context.allgather_stream)
        self._complete_trace_calls += 1
        if self._phase is RunnerPhase.REPLAYING and self._replay_index != len(self._trace):
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: global step ended after %d of %d replay events. "
                "Discarding the stale trace and tracing the next full step "
                "(divergence #%d).",
                self._replay_index,
                len(self._trace),
                self._divergences,
            )
            self._phase = RunnerPhase.TRACING
            self._trace.clear()
            self._cycles_observed = 0
            self._last_orientation.clear()
        if self._phase is RunnerPhase.TRACING and self._trace:
            self._phase = RunnerPhase.REPLAYING
            logger.info(
                "FsdpExecutionRunner: compiled %d-event trace, entering replay.", len(self._trace)
            )
        self._replay_index = 0
        # The batch boundary ends every module's unshard round; without this,
        # dedup entries from the trace batch (whose final unshards were never
        # followed by a reshard) would suppress the first replay unshards.
        self._consumed_this_round.clear()
        if self._phase is RunnerPhase.REPLAYING:
            self._cycles_observed += 1
        # Log every few batches so a training run shows whether replay is
        # actually validating events or stuck re-tracing.
        if self._complete_trace_calls % 10 == 0:
            self.report()

    def report(self) -> None:
        """Log the runner's replay statistics.

        A healthy runner shows ``cycles_observed`` increasing with every
        batch and ``replayed_occurrences`` much larger than ``divergences``.
        A runner that never replays (e.g. no complete_trace call, or a
        permanent divergence loop) is visible from this summary.
        """
        if self._use_trace_replay:
            logger.info(
                "FsdpExecutionRunner: phase=%s trace_len=%d cycles_observed=%d "
                "replayed_occurrences=%d divergences=%d complete_trace_calls=%d",
                self._phase.name,
                len(self._trace),
                self._cycles_observed,
                self._replayed_occurrences,
                self._divergences,
                self._complete_trace_calls,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _static_successor(
        self, module: "FsdpModule", orientation: str
    ) -> tuple["FsdpModule", str] | None:
        """Default mode: resolve the static-order successor.

        Activation recomputation runs forward hooks inside backward, whose
        forward-order prefetch must be skipped (its backward may already be
        complete and would not reshard the prefetched successor). Trace-replay
        mode owns all prefetch decisions and intentionally skips this check.
        """
        if (
            orientation == "rowwise"
            and getattr(module, "_phase", None) is not None
            and (module._phase == module.Phase.BACKWARD or torch._C._current_graph_task_id() != -1)
        ):
            return None
        if orientation == "rowwise":
            next_module = self._context.forward_order.next_item(module)
        else:
            next_module = self._context.backward_order.next_item(module)
        if next_module is None:
            return None
        return next_module, orientation

    def _validate_and_advance(
        self, kind: EventKind, module: "FsdpModule", orientation: str | None
    ) -> None:
        """Trace (append) or validate-and-advance (replay) one event.

        The trace path records the real op stream (consume/reshard). During
        replay each real op is validated against the traced event at the
        current position; on success the cursor advances. A mismatch is a
        divergence: the trace is cleared and re-traced from this event,
        degrading to demand-only execution until a full cycle matches again.

        Args:
            kind: Expected event kind.
            module: The FSDP module the real op applies to.
            orientation: Expected orientation (``None`` for reshard).
        """
        if self._phase is RunnerPhase.TRACING:
            self._trace.append(RunnerEvent(kind=kind, module=module, orientation=orientation))
            return

        if self._replay_index >= len(self._trace):
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: received %s(%s) past the %d-event global-step "
                "boundary. Re-tracing from this event (divergence #%d).",
                getattr(module, "_name", None) or type(module).__name__,
                orientation,
                len(self._trace),
                self._divergences,
            )
            self._retrace(kind, module, orientation)
            return

        expected = self._trace[self._replay_index]
        if (
            expected.kind is not kind
            or expected.module is not module
            or expected.orientation != orientation
        ):
            # Schedule diverged from the trace (e.g. batch-size or topology
            # change). Re-trace from this event; prefetch stays disabled
            # until a full cycle matches again.
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: replay divergence at index %d: expected %s(%s), "
                "got %s(%s). Re-tracing from this event (divergence #%d).",
                self._replay_index,
                getattr(expected.module, "_name", None) or type(expected.module).__name__,
                expected.orientation,
                getattr(module, "_name", None) or type(module).__name__,
                orientation,
                self._divergences,
            )
            self._retrace(kind, module, orientation)
            return

        self._replayed_occurrences += 1
        self._replay_index += 1

    def _release_abandoned_prefetches(self) -> bool:
        """Release materialized storage whose expected consumer never arrived.

        Returns:
            Whether any release work was queued on the all-gather stream.
        """
        prefetched_modules = tuple(self._pending_prefetches)
        self._pending_prefetches.clear()
        for module in prefetched_modules:
            module._release_unsharded_parameter_groups()
        return bool(prefetched_modules)

    def _retrace(self, kind: EventKind, module: "FsdpModule", orientation: str | None) -> None:
        """Reset to tracing and seed the new trace with the current event."""
        self._release_abandoned_prefetches()
        self._phase = RunnerPhase.TRACING
        self._trace = [RunnerEvent(kind=kind, module=module, orientation=orientation)]
        self._cycles_observed = 0
        # The divergence event ends the aborted replay round; dedup entries
        # from it must not suppress the re-traced remainder of the batch.
        # Re-mark the seed module for an unshard seed so duplicate hooks of
        # its current round stay deduped (a reshard seed ends that round).
        self._consumed_this_round.clear()
        self._last_orientation.clear()
        if kind is EventKind.UNSHARD:
            self._consumed_this_round.add(module)
