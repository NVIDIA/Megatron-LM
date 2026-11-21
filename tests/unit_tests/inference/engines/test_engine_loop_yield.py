# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The engine loop does not spin on waiting requests that a step cannot admit."""

import asyncio
from collections import defaultdict
from types import SimpleNamespace

import pytest

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.engines.dynamic.loop import EngineLoopMixin, EPWorkLevel
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeOnly,
    DynamicBatchControllerStepResult,
)


class _ForwardEngine(EngineLoopMixin):
    """Just enough engine state for one legacy-mode `async_forward`."""

    def __init__(self, controller_result):
        self.state = EngineState.RUNNING
        self.logging_step_interval = 0
        self.context = SimpleNamespace(
            config=SimpleNamespace(async_sched_mode=AsyncScheduleMode.LEGACY),
            num_prefill_requests=1,
            step_count=0,
            prefix_cache_lru_clock=0,
            active_token_count=0,
            chunked_prefill_request_id=-1,
        )

        async def generate(**kwargs):
            return controller_result

        self.controller = SimpleNamespace(async_generate_output_tokens_dynamic_batch=generate)

    def schedule_waiting_requests(self):
        pass

    def _finalize_finished_requests(self, *args):
        pass


NO_FORWARD = DecodeOnly(consumed=None, launched=None)


@pytest.mark.parametrize(
    "controller_result, ran_model",
    [
        # Nothing was admitted, so the controller returned before any forward.
        (DynamicBatchControllerStepResult(decode_only=NO_FORWARD), False),
        (DynamicBatchControllerStepResult(decode_only=NO_FORWARD, output={}), True),
        # An async-scheduling primer launches a forward without producing output yet.
        (DynamicBatchControllerStepResult(decode_only=NO_FORWARD, primer_only=True), True),
    ],
)
def test_forward_records_whether_the_step_ran_the_model(controller_result, ran_model):
    engine = _ForwardEngine(controller_result)
    engine._last_step_ran_model = not ran_model
    asyncio.run(engine.async_forward())
    assert engine._last_step_ran_model is ran_model


class _WaitingRequestEngine(EngineLoopMixin):
    """An engine with no running requests and one waiting request.

    Each step either admits the request or, standing in for a request whose
    recurrent-state slot a pending KV handoff holds, admits nothing. With EP
    peers, `peer_work` is the largest work level the peers report.
    """

    def __init__(self, admits, ep_world_size=1, peer_work=0):
        self.state = EngineState.RUNNING
        self.context = SimpleNamespace(
            get_active_request_count=lambda: 0, step_count=0, prefix_cache_lru_clock=0
        )
        self.controller = SimpleNamespace(dummy_forward=self._dummy_forward)
        self.step_start_event = self.step_end_event = SimpleNamespace(
            record=lambda: None, synchronize=lambda: None
        )
        self.waiting_request_ids = [1]
        self.has_admittable_kv_import = False
        self.pending_kv_import_count = 0
        self.pending_kv_push_count = 0
        self.tp_ep_world_size = ep_world_size
        self.ep_consensus_interval = 20
        self._ep_consensus_loop_counter = 0
        self._last_ep_consensus = (0, False)
        self.rank = 0
        self.admits = admits
        self.peer_work = peer_work
        self._state_events = defaultdict(asyncio.Event)
        self.reported_work = []
        self.events = []
        self.steps = 0
        self.idle_passes = 0

    def schedule_requests(self):
        pass

    async def _ep_establish_consensus(self, local_work, signal_consensus):
        self.reported_work.append(local_work)
        return max(local_work, self.peer_work), False

    def _dummy_forward(self):
        self.events.append("dummy")
        assert len(self.events) < 1000, "the engine loop ran dummy forwards without yielding"

    async def async_step(self):
        # Returns without awaiting anything, as a step that admits nothing does, and
        # advances the step and LRU clocks as async_forward does for every step.
        self.steps += 1
        assert self.steps < 1000, "the engine loop stepped without yielding"
        self.context.step_count += 1
        self.context.prefix_cache_lru_clock += 1
        self.events.append("model" if self.admits else "no-op step")
        self._last_step_ran_model = self.admits
        if self.admits:
            self.waiting_request_ids.clear()

    def _collect_gc_while_idle(self, drained):
        assert drained == self.admits, "only an emptied queue is a drained engine"
        self.idle_passes += 1

    async def _world_barrier(self):
        pass

    async def shutdown(self):
        pass


@pytest.mark.parametrize("admits", [False, True])
def test_coordinator_loop_idles_after_a_step_that_admits_nothing(admits):
    async def main():
        engine = _WaitingRequestEngine(admits)

        async def stop():
            # Another task on the engine's event loop; it runs only if the loop yields.
            engine.state = EngineState.STOPPING

        stopper = asyncio.ensure_future(stop())
        await engine.run_engine_with_coordinator()
        await stopper
        return engine

    engine = asyncio.run(main())
    # Either way one step runs, then the loop idles and lets the stopper run. A
    # blocked request is retried by stepping again after the idle pass.
    assert (engine.steps, engine.idle_passes) == (1, 1)
    assert engine.waiting_request_ids == ([] if admits else [1])


def test_coordinator_loop_retries_a_blocked_request_after_each_idle_pass():
    async def main():
        engine = _WaitingRequestEngine(admits=False)
        passes = 0

        def schedule_requests():
            nonlocal passes
            passes += 1
            if passes > 6:
                engine.state = EngineState.STOPPING

        engine.schedule_requests = schedule_requests
        await engine.run_engine_with_coordinator()
        return engine

    engine = asyncio.run(main())
    assert (engine.steps, engine.idle_passes) == (3, 3)


def _run_passes(engine, count):
    """Run the coordinator loop for `count` passes; return each pass's events."""

    async def main():
        starts = []
        inner = engine.schedule_requests

        def schedule_requests():
            inner()
            starts.append(len(engine.events))
            if len(starts) > count:
                engine.state = EngineState.STOPPING

        engine.schedule_requests = schedule_requests
        await engine.run_engine_with_coordinator()
        return starts

    starts = asyncio.run(main())
    return [engine.events[start:end] for start, end in zip(starts, starts[1:])]


def test_blocked_expert_parallel_rank_retries_in_every_collective_of_busy_peers():
    """A blocked EP rank must not leave a peer waiting in a model collective.

    While a peer runs the model, this rank joins its collective on every pass,
    after a no-op admission retry, and advances its clocks once per pass, as the
    peer does.
    """
    engine = _WaitingRequestEngine(admits=False, ep_world_size=2, peer_work=EPWorkLevel.MODEL_WORK)
    per_pass = _run_passes(engine, 45)
    assert per_pass == [["no-op step", "dummy"]] * 45
    assert engine.context.step_count == engine.context.prefix_cache_lru_clock == 45


def test_blocked_expert_parallel_group_idles_between_admission_attempts():
    """An EP group whose ranks are all blocked must reach idle, past any cached consensus.

    The peer mirrors this rank, so consensus sees what every rank reports. After an
    attempt admits nothing, the next consensus must see no work and the group idles,
    instead of running dummy forwards on cached work until the interval refresh.
    """
    engine = _WaitingRequestEngine(admits=False, ep_world_size=2, peer_work=0)
    per_pass = _run_passes(engine, 45)
    assert per_pass == [["no-op step", "dummy"], []] * 22 + [["no-op step", "dummy"]]
    attempt, no_work = EPWorkLevel.ADMISSION_ATTEMPT, EPWorkLevel.NO_WORK
    assert engine.reported_work == [attempt, no_work] * 22 + [attempt]
    assert engine.idle_passes == 22


def test_blocked_expert_parallel_rank_lets_other_tasks_run():
    async def main():
        engine = _WaitingRequestEngine(
            admits=False, ep_world_size=2, peer_work=EPWorkLevel.MODEL_WORK
        )

        async def stop():
            # Another task on the engine's event loop; it runs only if the loop yields.
            engine.state = EngineState.STOPPING

        stopper = asyncio.ensure_future(stop())
        await engine.run_engine_with_coordinator()
        await stopper
        return engine

    engine = asyncio.run(main())
    assert engine.events == ["no-op step", "dummy"]


class _ReleasesAFailedImportEngine(_WaitingRequestEngine):
    """Starts with a failed KV import to release and no waiting request."""

    def __init__(self):
        super().__init__(admits=True)
        self.waiting_request_ids = []
        self.has_admittable_kv_import = True

    async def async_step(self):
        if self.has_admittable_kv_import:
            # Releasing a failed import runs no model.
            self.has_admittable_kv_import = False
            self.events.append("release import")
            self._last_step_ran_model = False
            return
        await super().async_step()


def test_step_that_runs_no_model_leaves_an_empty_queue_ready_for_new_requests():
    engine = _ReleasesAFailedImportEngine()

    def arrive_after_first_pass():
        if engine.events and not engine.steps:
            engine.waiting_request_ids = [1]

    engine.schedule_requests = arrive_after_first_pass
    per_pass = _run_passes(engine, 2)
    # The request that arrives after the release is admitted on the next pass.
    assert per_pass == [["release import"], ["model"]]


def test_unpausing_retries_a_stalled_queue_at_once():
    engine = _WaitingRequestEngine(admits=True)
    # The queue stalled before the pause; the pause may have released its blocker.
    engine._waiting_queue_stalled = True
    engine.state = EngineState.UNPAUSING
    per_pass = _run_passes(engine, 2)
    assert per_pass == [[], ["model"]]


class _MaxAllReduce:
    """An all-reduce of (work, consensus) by max across ranks, matched by call order."""

    def __init__(self, size):
        self.size = size
        self.rounds = []
        self.calls = [0] * size

    async def __call__(self, rank, work=0, consensus=0):
        index = self.calls[rank]
        self.calls[rank] += 1
        if index == len(self.rounds):
            self.rounds.append(([], asyncio.get_running_loop().create_future()))
        values, done = self.rounds[index]
        values.append((work, consensus))
        if len(values) == self.size:
            done.set_result((max(w for w, _ in values), max(c for _, c in values) == -1))
        return await done


class _ExpertParallelRank(_WaitingRequestEngine):
    """One of two EP ranks that share collectives and record each one they enter.

    Real ranks hang unless they enter the same collectives in the same order, so
    both ranks' records must match. A rank may get a waiting request that it can
    admit at `admit_at_pass`, after which it decodes for `decode_steps` steps.
    """

    def __init__(
        self,
        rank,
        collectives,
        *,
        blocked,
        admit_at_pass=None,
        decode_steps=0,
        pause_at_pass=None,
        stop_after=30,
    ):
        super().__init__(admits=not blocked, ep_world_size=2)
        self.rank = rank
        self.all_reduce, self.barrier = collectives
        self.waiting_request_ids = [rank] if blocked else []
        self.ep_consensus_interval = 4
        self.active = 0
        self.context.get_active_request_count = lambda: self.active
        self.admit_at_pass = admit_at_pass
        self.decode_steps = decode_steps
        self.pause_at_pass = pause_at_pass
        self.stop_after = stop_after
        self.passes = 0
        self.entered = []
        self.states = []

    def schedule_requests(self):
        self.passes += 1
        self.states.append(self.state)
        if self.passes == self.admit_at_pass:
            self.waiting_request_ids.append(self.rank)
        if self.passes == self.pause_at_pass:
            self.state = EngineState.PAUSING
        if self.passes > self.stop_after:
            self.state = EngineState.STOPPING

    async def _ep_establish_consensus(self, local_work, signal_consensus):
        self.entered.append("consensus")
        return await self.all_reduce(self.rank, local_work, -1 if signal_consensus else 0)

    async def _world_barrier(self):
        self.entered.append("barrier")
        await self.barrier(self.rank)

    def _dummy_forward(self):
        self.entered.append("forward")

    def _collect_gc_while_idle(self, drained):
        self.idle_passes += 1

    async def async_step(self):
        self.context.step_count += 1
        self.context.prefix_cache_lru_clock += 1
        if self.active:
            self.active -= 1
            self._last_step_ran_model = True
        elif self.waiting_request_ids and self.admits:
            self.waiting_request_ids.clear()
            self.active = self.decode_steps
            self._last_step_ran_model = True
        else:
            self._last_step_ran_model = False
        if self._last_step_ran_model:
            self.entered.append("forward")


def _run_ranks(make_ranks):
    async def main():
        collectives = (_MaxAllReduce(2), _MaxAllReduce(2))
        ranks = make_ranks(collectives)
        # A rank left alone in a collective never returns.
        await asyncio.wait_for(
            asyncio.gather(*(rank.run_engine_with_coordinator() for rank in ranks)), timeout=10
        )
        return ranks

    return asyncio.run(main())


def _assert_in_lockstep(blocked, busy):
    assert blocked.entered == busy.entered
    assert blocked.context.step_count == busy.context.step_count
    assert blocked.context.prefix_cache_lru_clock == busy.context.prefix_cache_lru_clock
    # The blocked request is never admitted.
    assert blocked.waiting_request_ids == [0]


def test_blocked_rank_stays_in_lockstep_while_a_peer_admits_runs_and_drains():
    # The peer's 6 decode steps span cached-consensus refreshes, then both ranks idle.
    blocked, busy = _run_ranks(
        lambda collectives: (
            _ExpertParallelRank(0, collectives, blocked=True),
            _ExpertParallelRank(1, collectives, blocked=False, admit_at_pass=3, decode_steps=6),
        )
    )
    _assert_in_lockstep(blocked, busy)
    assert busy.entered.count("forward") > 7 and busy.idle_passes > 0


def test_blocked_rank_and_busy_peer_pause_together_when_they_pause_at_different_passes():
    blocked, busy = _run_ranks(
        lambda collectives: (
            _ExpertParallelRank(0, collectives, blocked=True, pause_at_pass=3),
            _ExpertParallelRank(
                1, collectives, blocked=False, admit_at_pass=1, decode_steps=20, pause_at_pass=6
            ),
        )
    )
    _assert_in_lockstep(blocked, busy)
    # Both pause at the same pass, while the peer still has requests to decode.
    assert blocked.states.index(EngineState.PAUSED) == busy.states.index(EngineState.PAUSED)
    assert busy.active > 0
