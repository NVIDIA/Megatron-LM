# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from enum import Enum


class EPAsyncPhase(str, Enum):
    """Tagged collective phases owned by the EP async protocol."""

    WORK_CONSENSUS = "ep_work_consensus"
    GRAPH_SHAPE = "ep_graph_shape"
    STEP_BEGIN = "ep_step_begin"
    PENDING_FORWARD_REUSE = "ep_pending_forward_reuse"
    ASYNC_HANDOFF = "ep_async_handoff"


@dataclass(frozen=True)
class EPWorkConsensus:
    """EP-wide work and state-transition consensus for one engine-loop iteration."""

    global_work: int
    all_pausing: bool


class EPAsyncStepProtocol:
    """Owns tagged EP async collectives and their per-phase step ordering."""

    def __init__(self, communicator=None):
        self.communicator = communicator
        self._phase_step_ids: dict[EPAsyncPhase, int] = {phase: 0 for phase in EPAsyncPhase}

    @property
    def enabled(self) -> bool:
        """Whether this rank participates in multi-rank EP protocol collectives."""
        return self.communicator is not None and self.communicator.world_size > 1

    def _next_step_id(self, phase: EPAsyncPhase) -> int:
        step_id = self._phase_step_ids[phase]
        self._phase_step_ids[phase] = step_id + 1
        return step_id

    async def all_reduce_max(
        self, phase: EPAsyncPhase, *local_vals: int, async_op: bool = True
    ) -> int | tuple[int, ...]:
        """Run a tagged EP MAX collective for an async call site."""
        if len(local_vals) == 0:
            raise ValueError("EP async protocol all_reduce_max requires at least one value")
        step_id = self._next_step_id(phase)
        if not self.enabled:
            return local_vals[0] if len(local_vals) == 1 else local_vals
        return await self.communicator.all_reduce_max(
            *local_vals, async_op=async_op, phase=phase.value, step_id=step_id
        )

    def sync_all_reduce_max(
        self, phase: EPAsyncPhase, *local_vals: int
    ) -> int | tuple[int, ...]:
        """Run a tagged EP MAX collective for a synchronous call site."""
        if len(local_vals) == 0:
            raise ValueError("EP async protocol sync_all_reduce_max requires at least one value")
        step_id = self._next_step_id(phase)
        if not self.enabled:
            return local_vals[0] if len(local_vals) == 1 else local_vals
        return self.communicator.sync_all_reduce_max(
            *local_vals, phase=phase.value, step_id=step_id
        )

    async def establish_work_consensus(
        self, local_work: int, signal_consensus: bool, *, async_op: bool = True
    ) -> EPWorkConsensus:
        """Share pending work and pause intent across EP ranks."""
        consensus_val = -1 if signal_consensus else 0
        if self.enabled:
            global_work, global_consensus = await self.all_reduce_max(
                EPAsyncPhase.WORK_CONSENSUS,
                local_work,
                consensus_val,
                async_op=async_op,
            )
        else:
            global_work, global_consensus = local_work, consensus_val

        return EPWorkConsensus(
            global_work=global_work,
            all_pausing=(global_consensus == -1),
        )
