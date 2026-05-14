# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from enum import Enum


class EPAsyncPhase(str, Enum):
    """Tagged collective phases owned by the EP async protocol."""

    WORK_CONSENSUS = "ep_work_consensus"
    WORK_CONSENSUS_ACK = "ep_work_consensus_ack"
    STEP_COMPLETE = "ep_step_complete"
    STEP_COMPLETE_ACK = "ep_step_complete_ack"
    GRAPH_SHAPE = "ep_graph_shape"
    STEP_BEGIN = "ep_step_begin"
    STEP_BEGIN_ACK = "ep_step_begin_ack"
    ASYNC_HANDOFF = "ep_async_handoff"
    ASYNC_HANDOFF_ACK = "ep_async_handoff_ack"


@dataclass(frozen=True)
class EPWorkConsensus:
    """EP-wide work and state-transition consensus for one engine-loop iteration."""

    step_id: int
    global_work: int
    all_pausing: bool


@dataclass(frozen=True)
class EPStepBeginDecision:
    """EP-wide decision for state carried into the next decode step."""

    step_id: int
    has_real_work: bool
    use_pending_async_sample: bool
    reuse_pending_forward: bool
    discard_pending_forward: bool
    row_mapped_forward: bool


@dataclass(frozen=True)
class EPAsyncHandoffDecision:
    """EP-wide real-rank decision for launching or skipping the async handoff."""

    step_id: int
    has_real_work: bool
    launch_async_forward: bool
    skip_async_forward: bool
    any_launch_request: bool
    any_skip_request: bool


class EPAsyncStepProtocol:
    """Owns tagged EP async collectives and their per-phase step ordering."""

    def __init__(self, communicator=None):
        self.communicator = communicator
        self._phase_step_ids: dict[EPAsyncPhase, int] = {phase: 0 for phase in EPAsyncPhase}
        self._next_ep_step_id = 0
        self._active_ep_step_id: int | None = None
        self._work_consensus_count = 0
        self._work_completion_count = 0
        self._idle_completion_count = 0
        self._step_begin_reuse_count = 0
        self._step_begin_discard_count = 0
        self._async_handoff_launch_count = 0
        self._async_handoff_skip_count = 0
        self._collective_error_count = 0

    @property
    def enabled(self) -> bool:
        """Whether this rank participates in multi-rank EP protocol collectives."""
        return self.communicator is not None and self.communicator.world_size > 1

    def diagnostics(self) -> dict[str, int | bool | None]:
        """Return protocol counters for tests and benchmark logs."""
        return {
            "enabled": self.enabled,
            "active_step_id": self._active_ep_step_id,
            "next_step_id": self._next_ep_step_id,
            "work_consensus": self._work_consensus_count,
            "work_completions": self._work_completion_count,
            "idle_completions": self._idle_completion_count,
            "step_begin_reuses": self._step_begin_reuse_count,
            "step_begin_discards": self._step_begin_discard_count,
            "handoff_launches": self._async_handoff_launch_count,
            "handoff_skips": self._async_handoff_skip_count,
            "collective_errors": self._collective_error_count,
            "phase_mismatches": getattr(self.communicator, "protocol_mismatch_count", 0),
        }

    def _next_step_id(self, phase: EPAsyncPhase) -> int:
        step_id = self._phase_step_ids[phase]
        self._phase_step_ids[phase] = step_id + 1
        return step_id

    def _begin_ep_step(self) -> int:
        if self._active_ep_step_id is not None:
            raise RuntimeError(f"EP protocol step {self._active_ep_step_id} is still active")
        step_id = self._next_ep_step_id
        self._next_ep_step_id += 1
        self._active_ep_step_id = step_id
        return step_id

    def _finish_ep_step(self) -> None:
        self._active_ep_step_id = None

    def _step_id_for_phase(self, phase: EPAsyncPhase) -> int:
        if self._active_ep_step_id is not None:
            return self._active_ep_step_id
        return self._next_step_id(phase)

    async def _all_reduce_max_at_step(
        self, phase: EPAsyncPhase, step_id: int, *local_vals: int, async_op: bool = True
    ) -> int | tuple[int, ...]:
        if not self.enabled:
            return local_vals[0] if len(local_vals) == 1 else local_vals
        try:
            return await self.communicator.all_reduce_max(
                *local_vals, async_op=async_op, phase=phase.value, step_id=step_id
            )
        except Exception:
            self._collective_error_count += 1
            raise

    async def _ack_at_step(
        self, phase: EPAsyncPhase, step_id: int, *, async_op: bool = True
    ) -> None:
        await self._all_reduce_max_at_step(phase, step_id, 1, async_op=async_op)

    def _sync_all_reduce_max_at_step(
        self, phase: EPAsyncPhase, step_id: int, *local_vals: int
    ) -> int | tuple[int, ...]:
        if not self.enabled:
            return local_vals[0] if len(local_vals) == 1 else local_vals
        try:
            return self.communicator.sync_all_reduce_max(
                *local_vals, phase=phase.value, step_id=step_id
            )
        except Exception:
            self._collective_error_count += 1
            raise

    def _sync_ack_at_step(self, phase: EPAsyncPhase, step_id: int) -> None:
        self._sync_all_reduce_max_at_step(phase, step_id, 1)

    async def all_reduce_max(
        self, phase: EPAsyncPhase, *local_vals: int, async_op: bool = True
    ) -> int | tuple[int, ...]:
        """Run a tagged EP MAX collective for an async call site."""
        if len(local_vals) == 0:
            raise ValueError("EP async protocol all_reduce_max requires at least one value")
        step_id = self._step_id_for_phase(phase)
        return await self._all_reduce_max_at_step(phase, step_id, *local_vals, async_op=async_op)

    def sync_all_reduce_max(self, phase: EPAsyncPhase, *local_vals: int) -> int | tuple[int, ...]:
        """Run a tagged EP MAX collective for a synchronous call site."""
        if len(local_vals) == 0:
            raise ValueError("EP async protocol sync_all_reduce_max requires at least one value")
        step_id = self._step_id_for_phase(phase)
        return self._sync_all_reduce_max_at_step(phase, step_id, *local_vals)

    async def establish_work_consensus(
        self, local_work: int, signal_consensus: bool, *, async_op: bool = True
    ) -> EPWorkConsensus:
        """Share pending work and pause intent across EP ranks."""
        consensus_val = -1 if signal_consensus else 0
        step_id = self._begin_ep_step()
        global_work, global_consensus = await self._all_reduce_max_at_step(
            EPAsyncPhase.WORK_CONSENSUS, step_id, local_work, consensus_val, async_op=async_op
        )
        await self._ack_at_step(EPAsyncPhase.WORK_CONSENSUS_ACK, step_id, async_op=async_op)
        self._work_consensus_count += 1

        return EPWorkConsensus(
            step_id=step_id, global_work=global_work, all_pausing=(global_consensus == -1)
        )

    async def complete_work_step(self, *, async_op: bool = True) -> None:
        """Close the active EP work step after real and dummy ranks have finished it."""
        if self._active_ep_step_id is None:
            return
        step_id = self._active_ep_step_id
        try:
            await self._all_reduce_max_at_step(
                EPAsyncPhase.STEP_COMPLETE, step_id, 1, async_op=async_op
            )
            await self._ack_at_step(EPAsyncPhase.STEP_COMPLETE_ACK, step_id, async_op=async_op)
            self._work_completion_count += 1
        finally:
            self._finish_ep_step()

    def complete_idle_step(self) -> None:
        """Close an EP step that ended at consensus without model work."""
        if self._active_ep_step_id is not None:
            self._idle_completion_count += 1
        self._finish_ep_step()

    def decide_step_begin(
        self,
        *,
        has_real_work: bool,
        has_pending_forward: bool,
        has_pending_async_sample: bool,
        pending_forward_reusable: bool,
        pending_forward_row_mapped: bool,
    ) -> EPStepBeginDecision:
        """Synchronize per-rank pending async state at the start of an EP work step."""
        step_id = self._step_id_for_phase(EPAsyncPhase.STEP_BEGIN)
        local_real = int(has_real_work)
        local_pending_forward = int(has_pending_forward)
        local_pending_sample = int(has_pending_async_sample)
        local_reusable = int(has_pending_forward and pending_forward_reusable)
        local_row_mapped = int(has_pending_forward and pending_forward_row_mapped)
        local_discard = int(has_pending_forward and not pending_forward_reusable)
        local_real_missing_forward = int(has_real_work and not has_pending_forward)
        local_real_missing_sample = int(has_real_work and not has_pending_async_sample)

        (
            any_real,
            any_pending_forward,
            any_pending_sample,
            any_reusable,
            any_row_mapped,
            any_discard,
            any_real_missing_forward,
            any_real_missing_sample,
        ) = self._sync_all_reduce_max_at_step(
            EPAsyncPhase.STEP_BEGIN,
            step_id,
            local_real,
            local_pending_forward,
            local_pending_sample,
            local_reusable,
            local_row_mapped,
            local_discard,
            local_real_missing_forward,
            local_real_missing_sample,
        )
        self._sync_ack_at_step(EPAsyncPhase.STEP_BEGIN_ACK, step_id)

        use_pending_async_sample = bool(any_pending_sample and not any_real_missing_sample)
        reuse_pending_forward = bool(
            any_pending_forward
            and any_reusable
            and not any_discard
            and not any_real_missing_forward
        )
        discard_pending_forward = bool(any_pending_forward and not reuse_pending_forward)
        if reuse_pending_forward:
            self._step_begin_reuse_count += 1
        if discard_pending_forward:
            self._step_begin_discard_count += 1

        return EPStepBeginDecision(
            step_id=step_id,
            has_real_work=bool(any_real),
            use_pending_async_sample=use_pending_async_sample,
            reuse_pending_forward=reuse_pending_forward,
            discard_pending_forward=discard_pending_forward,
            row_mapped_forward=bool(any_row_mapped and reuse_pending_forward),
        )

    def decide_async_handoff(
        self, *, has_real_work: bool, can_launch_async_handoff: bool
    ) -> EPAsyncHandoffDecision:
        """Synchronize whether the current EP work step launches an async forward."""
        step_id = self._step_id_for_phase(EPAsyncPhase.ASYNC_HANDOFF)
        local_real = int(has_real_work)
        local_launch = int(has_real_work and can_launch_async_handoff)
        local_real_skip = int(has_real_work and not can_launch_async_handoff)

        any_real, any_launch, any_real_skip = self._sync_all_reduce_max_at_step(
            EPAsyncPhase.ASYNC_HANDOFF, step_id, local_real, local_launch, local_real_skip
        )
        self._sync_ack_at_step(EPAsyncPhase.ASYNC_HANDOFF_ACK, step_id)
        launch_async_forward = bool(any_launch and not any_real_skip)
        if launch_async_forward:
            self._async_handoff_launch_count += 1
        else:
            self._async_handoff_skip_count += 1

        return EPAsyncHandoffDecision(
            step_id=step_id,
            has_real_work=bool(any_real),
            launch_async_forward=launch_async_forward,
            skip_async_forward=not launch_async_forward,
            any_launch_request=bool(any_launch),
            any_skip_request=bool(any_real_skip),
        )
