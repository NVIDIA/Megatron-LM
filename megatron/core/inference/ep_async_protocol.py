# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from enum import Enum


class EPAsyncPhase(str, Enum):
    """Tagged collective phases owned by the EP async protocol."""

    WORK_CONSENSUS = "ep_work_consensus"
    STEP_COMPLETE = "ep_step_complete"
    GRAPH_SHAPE = "ep_graph_shape"
    STEP_BEGIN = "ep_step_begin"
    PENDING_FORWARD_REUSE = "ep_pending_forward_reuse"
    ASYNC_HANDOFF = "ep_async_handoff"


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


class EPAsyncStepProtocol:
    """Owns tagged EP async collectives and their per-phase step ordering."""

    def __init__(self, communicator=None):
        self.communicator = communicator
        self._phase_step_ids: dict[EPAsyncPhase, int] = {phase: 0 for phase in EPAsyncPhase}
        self._next_ep_step_id = 0
        self._active_ep_step_id: int | None = None

    @property
    def enabled(self) -> bool:
        """Whether this rank participates in multi-rank EP protocol collectives."""
        return self.communicator is not None and self.communicator.world_size > 1

    def _next_step_id(self, phase: EPAsyncPhase) -> int:
        step_id = self._phase_step_ids[phase]
        self._phase_step_ids[phase] = step_id + 1
        return step_id

    def _begin_ep_step(self) -> int:
        if self._active_ep_step_id is not None:
            raise RuntimeError(
                f"EP protocol step {self._active_ep_step_id} is still active"
            )
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
        self,
        phase: EPAsyncPhase,
        step_id: int,
        *local_vals: int,
        async_op: bool = True,
    ) -> int | tuple[int, ...]:
        if not self.enabled:
            return local_vals[0] if len(local_vals) == 1 else local_vals
        return await self.communicator.all_reduce_max(
            *local_vals, async_op=async_op, phase=phase.value, step_id=step_id
        )

    def _sync_all_reduce_max_at_step(
        self, phase: EPAsyncPhase, step_id: int, *local_vals: int
    ) -> int | tuple[int, ...]:
        if not self.enabled:
            return local_vals[0] if len(local_vals) == 1 else local_vals
        return self.communicator.sync_all_reduce_max(
            *local_vals, phase=phase.value, step_id=step_id
        )

    async def all_reduce_max(
        self, phase: EPAsyncPhase, *local_vals: int, async_op: bool = True
    ) -> int | tuple[int, ...]:
        """Run a tagged EP MAX collective for an async call site."""
        if len(local_vals) == 0:
            raise ValueError("EP async protocol all_reduce_max requires at least one value")
        step_id = self._step_id_for_phase(phase)
        return await self._all_reduce_max_at_step(
            phase, step_id, *local_vals, async_op=async_op
        )

    def sync_all_reduce_max(
        self, phase: EPAsyncPhase, *local_vals: int
    ) -> int | tuple[int, ...]:
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
            EPAsyncPhase.WORK_CONSENSUS,
            step_id,
            local_work,
            consensus_val,
            async_op=async_op,
        )

        return EPWorkConsensus(
            step_id=step_id,
            global_work=global_work,
            all_pausing=(global_consensus == -1),
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
        finally:
            self._finish_ep_step()

    def complete_idle_step(self) -> None:
        """Close an EP step that ended at consensus without model work."""
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

        use_pending_async_sample = bool(any_pending_sample and not any_real_missing_sample)
        reuse_pending_forward = bool(
            any_pending_forward
            and any_reusable
            and not any_discard
            and not any_real_missing_forward
        )
        discard_pending_forward = bool(any_pending_forward and not reuse_pending_forward)

        return EPStepBeginDecision(
            step_id=step_id,
            has_real_work=bool(any_real),
            use_pending_async_sample=use_pending_async_sample,
            reuse_pending_forward=reuse_pending_forward,
            discard_pending_forward=discard_pending_forward,
            row_mapped_forward=bool(any_row_mapped and reuse_pending_forward),
        )
