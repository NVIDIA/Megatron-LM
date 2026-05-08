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
