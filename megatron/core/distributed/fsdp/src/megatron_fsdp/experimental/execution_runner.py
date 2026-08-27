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

Three per-op interfaces:

- ``record_unshard(module, orientation)`` / ``record_reshard(module)``:
  trace path — record the real event, or during replay validate it against
  the traced cycle and advance the cursor.
- ``suggest_prefetch_plan()``: optimization path — the configured future
  traced unshard occurrence and its parameter-lifetime gate, or ``None``
  while tracing or near the global-batch boundary.
- ``suggest_skip_reshard(module)``: optimization path — whether this reshard
  can be skipped because the next traced unshard reuses the same module and
  orientation, keeping the storage resident.
"""

import dataclasses
import logging
from enum import Enum, auto

# Forward reference; FsdpModule is imported lazily to avoid a cycle.
from typing import TYPE_CHECKING

import torch

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


@dataclasses.dataclass(frozen=True)
class _PrefetchSuggestion:
    """One traced prefetch target and its last intervening physical reshard."""

    module: "FsdpModule"
    orientation: str
    release_after_reshard_index: int | None = None
    target_unshard_index: int | None = None


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
        self._unshard_count = 0
        self._skipped_reshard_indices: set[int] = set()
        # A deep target may already be materialized for an earlier occurrence.
        # In that case its prefetch is a reservation: retain the storage at the
        # last intervening physical reshard, then consume it at the exact target.
        self._pending_prefetches: dict[int, list[_PrefetchSuggestion]] = {}
        self._retained_prefetches: dict[int, _PrefetchSuggestion] = {}
        self._resident_prefetches: dict[int, _PrefetchSuggestion] = {}
        self._cycles_observed = 0
        # Modules consumed in the current round. The fine-grained schedule
        # fires one hook per sub-module (dense, experts), so the same module
        # can be recorded several times within a round; only the first is a
        # real unshard. Cleared by record_reshard() and at the batch boundary.
        self._consumed_this_round: set[FsdpModule] = set()
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

    def record_unshard(self, module: "FsdpModule", orientation: str) -> bool:
        """Record (tracing) or validate (replay) an unshard event.

        The fine-grained schedule fires one hook per sub-module (dense,
        experts), so the same module can arrive several times within a round;
        only the first arrival is a real unshard for the trace. Call
        ``suggest_prefetch_plan()`` right after this returns to get the
        prefetch target (replay only).

        Args:
            module: The FSDP module being unsharded for compute.
            orientation: Payload orientation (``"rowwise"`` forward,
                ``"colwise"`` backward).

        Returns:
            Whether this is the first fine-grained hook for the occurrence.
        """
        if not self._use_trace_replay:
            return True
        if module in self._consumed_this_round:
            return False
        self._consumed_this_round.add(module)
        self._last_orientation[module] = orientation
        replayed_index = self._validate_and_advance(EventKind.UNSHARD, module, orientation)
        if replayed_index is not None:
            retained = self._retained_prefetches.pop(replayed_index, None)
            resident = self._resident_prefetches.pop(replayed_index, None)
            suggestion = retained or resident
            if retained is not None and resident is not None:
                raise RuntimeError("FSDP target has duplicate speculative prefetches.")
            if suggestion is not None:
                if suggestion.module is not module or suggestion.orientation != orientation:
                    raise RuntimeError("Speculative FSDP prefetch does not match its traced target.")
                module_name = module.name if module.name else "<root>"
                reuse_kind = "retained" if retained is not None else "resident"
                torch.cuda.nvtx.range_push(
                    f"MFSDP AG {reuse_kind} reuse target={module_name} orientation={orientation}"
                )
                torch.cuda.nvtx.range_pop()
        return True

    def record_reshard(self, module: "FsdpModule") -> int | None:
        """Record (tracing) or validate (replay) a reshard event.

        The reshard ends the module's current unshard round: it clears the
        per-round dedup entry so the next unshard of the same module (e.g. the
        backward pass after the forward pass) records a fresh event. Call
        ``suggest_skip_reshard(module)`` right after this returns to learn
        whether the actual reshard can be skipped (replay only).

        Args:
            module: The FSDP module whose unsharded storage is released.

        Returns:
            The exact replay trace index, or ``None`` outside validated replay.
        """
        if not self._use_trace_replay:
            return None
        # The reshard ends the module's unshard round; discard its dedup
        # entry so the next unshard (e.g. backward after forward) records a
        # fresh event.
        self._consumed_this_round.discard(module)
        return self._validate_and_advance(EventKind.RESHARD, module, None)

    # ------------------------------------------------------------------
    # Interface 2: prefetch suggestion
    # ------------------------------------------------------------------

    def suggest_prefetch_plan(
        self, module: "FsdpModule", orientation: str, *, depth: int = 1
    ) -> _PrefetchSuggestion | None:
        """Return the depth-th future consume and its parameter-lifetime gate."""
        if depth < 1:
            raise ValueError(f"Prefetch depth must be positive, got {depth}.")
        if not self._use_trace_replay:
            if depth != 1:
                raise ValueError("Prefetch depth greater than one requires trace replay.")
            successor = self._static_successor(module, orientation)
            return _PrefetchSuggestion(*successor) if successor is not None else None
        if self._phase is not RunnerPhase.REPLAYING or not self._trace:
            return None
        if depth > self._unshard_count:
            raise ValueError(
                f"Prefetch depth {depth} exceeds the {self._unshard_count} "
                "UNSHARD occurrences in the replay trace."
            )

        remaining = depth
        target_index = None
        target_event = None
        # Deliberately do not wrap: a gather may not cross the optimizer/global-
        # batch boundary because the materialized weights would become stale.
        for index in range(self._replay_index, len(self._trace)):
            event = self._trace[index]
            if event.kind is EventKind.UNSHARD:
                remaining -= 1
                if remaining == 0:
                    target_index = index
                    target_event = event
                    break
        if target_event is None:
            return None

        release_after_reshard_index = None
        for index in range(self._replay_index, target_index):
            event = self._trace[index]
            if (
                event.kind is EventKind.RESHARD
                and event.module is target_event.module
                and index not in self._skipped_reshard_indices
            ):
                release_after_reshard_index = index
        assert target_event.orientation is not None
        return _PrefetchSuggestion(
            target_event.module,
            target_event.orientation,
            release_after_reshard_index,
            target_index,
        )

    def defer_prefetch(self, suggestion: _PrefetchSuggestion) -> None:
        """Reserve an existing materialization for a later traced occurrence."""
        gate = suggestion.release_after_reshard_index
        if gate is None or suggestion.target_unshard_index is None:
            raise ValueError("Only a gated prefetch can be deferred.")
        self._pending_prefetches.setdefault(gate, []).append(suggestion)
        module_name = suggestion.module.name if suggestion.module.name else "<root>"
        torch.cuda.nvtx.range_push(
            f"MFSDP AG queued target={module_name} orientation={suggestion.orientation} "
            f"reshard_index={gate} target_index={suggestion.target_unshard_index}"
        )
        torch.cuda.nvtx.range_pop()

    def track_prefetch(self, suggestion: _PrefetchSuggestion) -> None:
        """Track an immediately submitted gather until its exact traced demand."""
        target_index = suggestion.target_unshard_index
        if target_index is None:
            # Static-order prefetch has no replay boundary or occurrence token.
            return
        if suggestion.release_after_reshard_index is not None:
            raise ValueError("A gated prefetch must be deferred instead of tracked as resident.")
        if target_index in self._resident_prefetches:
            raise RuntimeError("Duplicate immediate FSDP prefetch for one trace occurrence.")
        self._resident_prefetches[target_index] = suggestion

    def retain_prefetches_across_reshard(
        self, module: "FsdpModule", reshard_index: int | None
    ) -> bool:
        """Retain storage when this exact reshard gates a future depth target."""
        if reshard_index is not None:
            suggestions = self._pending_prefetches.pop(reshard_index, [])
            for suggestion in suggestions:
                if suggestion.module is not module or suggestion.target_unshard_index is None:
                    raise RuntimeError("FSDP prefetch reservation reached the wrong reshard.")
                # Ordinary groups ignore orientation, while MXFP8 groups materialize
                # both rowwise and colwise payloads in one unshard. The current
                # storage can therefore satisfy a future occurrence of either phase.
                self._retained_prefetches[suggestion.target_unshard_index] = suggestion
            if suggestions:
                return True
        # A direct prefetch is expected to remain resident until its target. If
        # divergence instead reaches a physical reshard first, discard the
        # occurrence token and let the caller release that storage normally.
        resident_indices = [
            target_index
            for target_index, suggestion in self._resident_prefetches.items()
            if suggestion.module is module
        ]
        for target_index in resident_indices:
            del self._resident_prefetches[target_index]
        # A retained target may encounter a replay divergence before its demand.
        # Keep its live storage until it is consumed or explicitly flushed.
        return any(suggestion.module is module for suggestion in self._retained_prefetches.values())

    def release_speculative_prefetches(self) -> None:
        """Release unconsumed gathers before optimizer-side parameter updates."""
        self._pending_prefetches.clear()
        speculative = tuple(self._retained_prefetches.values()) + tuple(
            self._resident_prefetches.values()
        )
        self._retained_prefetches.clear()
        self._resident_prefetches.clear()
        released: set[int] = set()
        for suggestion in speculative:
            key = id(suggestion.module)
            if key in released:
                continue
            released.add(key)
            suggestion.module._reshard_parameter_groups(record_execution=False)

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
            # Never retain full parameters across the optimizer boundary.
            return False
        next_event = self._trace[self._replay_index]
        return (
            next_event.kind is EventKind.UNSHARD
            and next_event.module is module
            and next_event.orientation == self._last_orientation.get(module)
        )

    # ------------------------------------------------------------------
    # Lifecycle: batch boundary
    # ------------------------------------------------------------------

    def complete_trace(self) -> None:
        """Compile the recorded trace into the replay cycle.

        Called once by the optimizer at every global-batch boundary. The first
        batch (with a non-empty trace) transitions to ``REPLAYING``; subsequent
        calls reset the replay cursor for the next batch while keeping the
        compiled cycle.
        """
        if not self._use_trace_replay:
            return
        self._complete_trace_calls += 1
        if self._phase is RunnerPhase.REPLAYING and self._replay_index != len(self._trace):
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: replay ended at index %d of %d at the global-batch "
                "boundary. Re-tracing the next batch (divergence #%d).",
                self._replay_index,
                len(self._trace),
                self._divergences,
            )
            self._phase = RunnerPhase.TRACING
            self._trace.clear()
            self._unshard_count = 0
            self._skipped_reshard_indices.clear()
            self._pending_prefetches.clear()
        if self._phase is RunnerPhase.TRACING and self._trace:
            self._phase = RunnerPhase.REPLAYING
            self._compile_trace()
            logger.info(
                "FsdpExecutionRunner: compiled %d-event trace, entering replay.",
                len(self._trace),
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
        if getattr(module, "_phase", None) is not None and (
            module._phase == module.Phase.BACKWARD
            or torch._C._current_graph_task_id() != -1
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
        self,
        kind: EventKind,
        module: "FsdpModule",
        orientation: str | None,
    ) -> int | None:
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
            return None

        if self._replay_index >= len(self._trace):
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: replay emitted an event beyond the traced "
                "global-batch boundary. Re-tracing from this event (divergence #%d).",
                self._divergences,
            )
            self._retrace(kind, module, orientation)
            return None

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
            return None

        replayed_index = self._replay_index
        self._replayed_occurrences += 1
        self._replay_index += 1
        return replayed_index

    def _compile_trace(self) -> None:
        """Cache occurrence count and logical reshard skips for replay."""
        self._unshard_count = sum(event.kind is EventKind.UNSHARD for event in self._trace)
        self._skipped_reshard_indices.clear()
        last_orientation: dict[FsdpModule, str] = {}
        for index, event in enumerate(self._trace):
            if event.kind is EventKind.UNSHARD:
                assert event.orientation is not None
                last_orientation[event.module] = event.orientation
                continue
            if index + 1 >= len(self._trace):
                continue
            next_event = self._trace[index + 1]
            if (
                next_event.kind is EventKind.UNSHARD
                and next_event.module is event.module
                and next_event.orientation == last_orientation.get(event.module)
            ):
                self._skipped_reshard_indices.add(index)

    def _retrace(
        self,
        kind: EventKind,
        module: "FsdpModule",
        orientation: str | None,
    ) -> None:
        """Reset to tracing and seed the new trace with the current event."""
        self._phase = RunnerPhase.TRACING
        self._trace = [RunnerEvent(kind=kind, module=module, orientation=orientation)]
        self._replay_index = 0
        self._unshard_count = 0
        self._skipped_reshard_indices.clear()
        self._pending_prefetches.clear()
        self._cycles_observed = 0
        # The divergence event ends the aborted replay round; dedup entries
        # from it must not suppress the re-traced remainder of the batch.
        # Re-mark the seed module for an unshard seed so duplicate hooks of
        # its current round stay deduped (a reshard seed ends that round).
        self._consumed_this_round.clear()
        self._last_orientation.clear()
        if kind is EventKind.UNSHARD:
            self._consumed_this_round.add(module)
