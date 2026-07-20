# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Caller-owned orchestration of grouped rollout generation over an agent."""

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, AsyncIterator, NamedTuple

import numpy as np

from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

from ..inference import InferenceResponse, ReturnsRaw
from ..rollout_granularity import (
    GRANULARITY_RANK,
    RELEASE_STATE_BY_SUBMISSION,
    ConsumptionGranularity,
    ReleaseState,
    SubmissionGranularity,
)
from .api import GroupedRolloutRequest, GroupRolloutParams, RolloutGroup

if TYPE_CHECKING:
    from .api import GroupedRolloutGenerator


class _GranularityConfig(NamedTuple):
    submission: SubmissionGranularity
    consumption: ConsumptionGranularity
    num_groups_per_batch: int
    num_groups_per_env: tuple[int, ...]

    @classmethod
    def from_request(
        cls, request: GroupedRolloutRequest, num_groups_per_env: list[int]
    ) -> "_GranularityConfig":
        """Build the per-request granularity policy.

        Args:
            request: Grouped rollout request carrying the granularity choices.
            num_groups_per_env: Groups each env contributes to one batch, in env order.

        Returns:
            A validated _GranularityConfig.
        """
        cls._validate(request, num_groups_per_env)
        return cls(
            submission=request.submission_granularity,
            consumption=request.consumption_granularity,
            num_groups_per_batch=request.num_groups,
            num_groups_per_env=tuple(num_groups_per_env),
        )

    def env_of_index(self, index_in_batch: int) -> int:
        """Map a batch slot to the env owning it (slots are env-blocked, in env order).

        Args:
            index_in_batch: Slot index within one trainer batch.

        Returns:
            The env_index owning the slot.

        Raises:
            IndexError: If the slot lies outside the batch.
        """
        boundary = 0
        for env_index, groups in enumerate(self.num_groups_per_env):
            boundary += groups
            if index_in_batch < boundary:
                return env_index
        raise IndexError(
            f"index_in_batch {index_in_batch} outside batch of {self.num_groups_per_batch}"
        )

    def units_per_batch(self, rollouts_per_group: int) -> int:
        """Submission units in one batch; gate capacity = depth-in-batches x this.

        Args:
            rollouts_per_group: Rollouts per group, needed only for R granularity.

        Returns:
            The number of submission units one trainer batch contains.
        """
        return {
            "R": self.num_groups_per_batch * rollouts_per_group,
            "G": self.num_groups_per_batch,
            "E": len(self.num_groups_per_env),
            "B": 1,
        }[self.submission]

    @staticmethod
    def _validate(request: GroupedRolloutRequest, num_groups_per_env: list[int]) -> None:
        """Reject invalid granularity, layout, and filter combinations.

        Args:
            request: Grouped rollout request to check.
            num_groups_per_env: Proposed per-env group layout.

        Raises:
            AssertionError: If consumption is finer than submission, the layout
                starves an env or does not sum to num_groups, or reward
                filtering is requested.
        """
        assert (
            GRANULARITY_RANK[request.consumption_granularity]
            >= GRANULARITY_RANK[request.submission_granularity]
        ), (
            f"Consumption granularity ({request.consumption_granularity}) must be no finer "
            f"than submission granularity ({request.submission_granularity})."
        )
        assert all(
            groups > 0 for groups in num_groups_per_env
        ), "Each environment must request at least one group per batch."
        assert (
            sum(num_groups_per_env) == request.num_groups
        ), "The sum of groups per environment must equal the total number of groups requested."
        assert not request.filter_groups_with_same_reward, (
            "filter_groups_with_same_reward is not currently supported: dropped groups "
            "are not regenerated, so non-streaming callers receive fewer groups than "
            "requested and batch-order consumers stall on incomplete batches."
        )


class _SubmissionGate:
    """Gate capacity is measured in units of the configured submission granularity."""

    def __init__(
        self,
        *,
        capacity: int,
        submission: SubmissionGranularity,
    ) -> None:
        """Create a gate with `capacity` slots counted at `submission` granularity.

        Args:
            capacity: Maximum submission units in flight.
            submission: Configured submission granularity; only matching
                acquire_for/release_after calls touch the semaphore.
        """
        self._sem = asyncio.Semaphore(capacity)
        self._submission = submission
        self._release_on = RELEASE_STATE_BY_SUBMISSION[submission]
        self.capacity = capacity
        # Observability counters, updated only on the configured submission
        # granularity (the only path that touches the semaphore). `held`
        # counts slots currently held; `prepare_blocked_seconds` accumulates
        # time stage_prepare spent waiting on the semaphore.
        self.held = 0
        self.prepare_blocked_seconds = 0.0
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire_for(self, granularity: SubmissionGranularity) -> None:
        """Take one slot when crossing a boundary of the configured granularity.

        Args:
            granularity: The dispatch boundary being crossed; no-op unless it
                matches the gate's configured submission granularity.
        """
        if self._submission == granularity:
            start = time.monotonic()
            await self._sem.acquire()
            self.prepare_blocked_seconds += time.monotonic() - start
            self.held += 1
            self.acquire_calls += 1

    def release_after(self, state: ReleaseState) -> None:
        """Release one slot when work reaches the configured release state.

        Args:
            state: The lifecycle state just reached; no-op unless it matches
                RELEASE_STATE_BY_SUBMISSION for the configured granularity.
        """
        if self._release_on == state:
            self._sem.release()
            self.held -= 1
            self.release_calls += 1


class _InferWorkItem(NamedTuple):
    """One rollout's worth of work flowing from prepare to infer.

    Timestamps are wall-clock monotonic seconds: `prepared_at` is stamped at
    construction and `infer_dequeued_at` is filled in via `_replace` when an
    infer worker dequeues the item. Zero means "not yet reached".
    """

    group_id: int
    rollout_idx: int
    batch_id: int
    index_in_batch: int
    params: GroupRolloutParams
    env_index: int = 0
    prepared_at: float = 0.0
    infer_dequeued_at: float = 0.0


class _InferredItem(NamedTuple):
    """One rollout post-inference, flowing from infer to assemble."""

    item: _InferWorkItem
    response: InferenceResponse
    inferred_at: float = 0.0


class RolloutPipeline:
    """Orchestrates grouped rollout generation over an agent, one instance per request.

    Constructed and driven by the caller (e.g. the trainer via run()); the agent
    only supplies the env layout, per-group preparation, and inference calls.
    """

    def __init__(
        self,
        agent: "GroupedRolloutGenerator",
        request: GroupedRolloutRequest,
        parallel_generation_tasks: int,
    ) -> None:
        """Validate the request and size the gate, queues, and worker pool.

        Args:
            agent: Agent supplying the env layout, preparation, and inference.
            request: Grouped rollout request to serve; one pipeline per request.
            parallel_generation_tasks: Submission gate depth in trainer
                batches; units_per_batch scales it to submission units.
        """
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."
        self.agent = agent
        self.request = request
        self.gran_policy = _GranularityConfig.from_request(
            request, agent.rollout_group_layout(request.num_groups)
        )
        self.gate = _SubmissionGate(
            capacity=parallel_generation_tasks
            * self.gran_policy.units_per_batch(request.rollouts_per_group),
            submission=self.gran_policy.submission,
        )
        self.num_infer_workers = (
            parallel_generation_tasks
            * self.gran_policy.num_groups_per_batch
            * request.rollouts_per_group
        )
        if not request.streaming:
            self.num_infer_workers = min(
                self.num_infer_workers, request.num_groups * request.rollouts_per_group
            )

        # Core queues.
        self.infer_queue = asyncio_Queue()
        self.assemble_queue = asyncio_Queue()
        self.output_queue = asyncio_Queue()
        self.bank = asyncio_Queue()
        self.banked_batches = 0
        self.consumed_batches = 0

        # Buffers of partial results.
        self._assemble_pending: dict[int, list[_InferredItem]] = {}
        self._env_unit_assembled: dict[tuple[int, int], int] = {}
        self._consume_pending: dict[int, list[RolloutGroup]] = {}
        self._output_enqueued_at: dict[tuple[int, int], float] = {}

        # Observability accumulators.
        self.infer_queue_dwell: list[float] = []
        self.engine_dwell: list[float] = []
        self.assemble_queue_dwell: list[float] = []
        self.output_queue_dwell: list[float] = []
        self.prepared_count = 0
        self.inferred_count = 0
        self.assembled_count = 0
        self.yielded_count = 0
        self.prepared_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)
        self.assembled_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)
        self.yielded_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)

    async def run(self) -> AsyncIterator[RolloutGroup]:
        """Run the pipeline stages; cancels them when the iterator is closed.

        Yields:
            RolloutGroup: Groups in consumption-granularity order.
        """
        tasks = (
            asyncio.create_task(self.stage_prepare()),
            asyncio.create_task(self.stage_infer()),
            asyncio.create_task(self.stage_assemble()),
            asyncio.create_task(self.stage_bank()),
        )
        try:
            async for group in self.stage_consume():
                yield group
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stage_prepare(self) -> None:
        """Generate gated inference work items."""
        assert (
            self.request.streaming
            or self.request.num_groups % self.gran_policy.num_groups_per_batch == 0
        ), "non-streaming requires num_groups to be a multiple of num_groups_per_batch"
        group_id = 0
        try:
            while self.request.streaming or group_id < self.request.num_groups:
                await self.gate.acquire_for("B")
                batch_id = group_id // self.gran_policy.num_groups_per_batch

                env_index = -1
                next_env_start = 0
                for index_in_batch in range(self.gran_policy.num_groups_per_batch):
                    if index_in_batch == next_env_start:
                        # Env-unit boundary: under E submission, hold one gate
                        # slot per env-unit until its last group assembles.
                        env_index += 1
                        next_env_start += self.gran_policy.num_groups_per_env[env_index]
                        await self.gate.acquire_for("E")
                    await self.gate.acquire_for("G")
                    params: GroupRolloutParams = await self.agent.prepare_group_rollout(
                        self.request, env_index=env_index
                    )
                    self.prepared_groups_per_env[env_index] += 1

                    for rollout_idx in range(self.request.rollouts_per_group):
                        await self.gate.acquire_for("R")
                        item = _InferWorkItem(
                            group_id=group_id,
                            rollout_idx=rollout_idx,
                            batch_id=batch_id,
                            index_in_batch=index_in_batch,
                            params=params,
                            env_index=env_index,
                            prepared_at=time.monotonic(),
                        )
                        await self.infer_queue.put(item)
                        self.prepared_count += 1
                    group_id += 1
        finally:
            self.infer_queue.shutdown()

    async def stage_infer(self) -> None:
        """Run a persistent pool of inference workers, spawned once per pipeline."""
        workers = [
            asyncio.create_task(self._infer_worker()) for _ in range(self.num_infer_workers)
        ]
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            for worker in workers:
                worker.cancel()
            self.assemble_queue.shutdown()

    async def _infer_worker(self) -> None:
        while True:
            try:
                item = await self.infer_queue.get()
            except asyncio_QueueShutDown:
                return
            item = item._replace(infer_dequeued_at=time.monotonic())
            if item.prepared_at:
                self.infer_queue_dwell.append(item.infer_dequeued_at - item.prepared_at)
            await self._infer_one(item)

    @trace_async_exceptions(verbose=True)
    async def _infer_one(self, item: _InferWorkItem) -> None:
        """Run inference for one work item and hand the result to assemble.

        Args:
            item: The dequeued work item; its params carry the serving agent.
        """
        response = await item.params.agent.get_rollout_response(
            self.request, item.params.inference_request
        )
        inferred_at = time.monotonic()
        self.gate.release_after("inferred")
        if item.infer_dequeued_at:
            self.engine_dwell.append(inferred_at - item.infer_dequeued_at)
        self.inferred_count += 1
        await self.assemble_queue.put(
            _InferredItem(item=item, response=response, inferred_at=inferred_at)
        )

    async def stage_assemble(self) -> None:
        """Build complete rollout groups from inferred items."""
        pending = self._assemble_pending
        try:
            while True:
                try:
                    inferred = await self.assemble_queue.get()
                except asyncio_QueueShutDown:
                    break
                dequeued_at = time.monotonic()
                if inferred.inferred_at:
                    self.assemble_queue_dwell.append(dequeued_at - inferred.inferred_at)
                bucket = pending.setdefault(inferred.item.group_id, [])
                bucket.append(inferred)
                if len(bucket) < self.request.rollouts_per_group:
                    continue
                completed = pending.pop(inferred.item.group_id)
                completed.sort(key=lambda item: item.item.rollout_idx)
                rollouts = await asyncio.gather(
                    *[item.item.params.build_rollout(item.response) for item in completed]
                )
                first = completed[0]
                self.gate.release_after("assembled")
                self.assembled_count += 1
                self.assembled_groups_per_env[first.item.env_index] += 1
                # An env-unit is complete once all of its env's groups in this
                # batch have assembled; that releases the E-submission slot.
                env_unit_key = (first.item.batch_id, first.item.env_index)
                self._env_unit_assembled[env_unit_key] = (
                    self._env_unit_assembled.get(env_unit_key, 0) + 1
                )
                unit_size = self.gran_policy.num_groups_per_env[first.item.env_index]
                if self._env_unit_assembled[env_unit_key] >= unit_size:
                    del self._env_unit_assembled[env_unit_key]
                    self.gate.release_after("env_assembled")
                # NOTE: this filter is currently non-functional dead code:
                # _GranularityConfig._validate rejects filter_groups_with_same_reward
                # at pipeline construction, so `keep` is always True. Kept for a
                # future PR that regenerates dropped groups instead of
                # under-delivering to the caller.
                keep = (
                    not self.request.filter_groups_with_same_reward
                    or np.std([rollout.reward for rollout in rollouts]) > 1e-6
                )
                if keep:
                    output_enqueued_at = time.monotonic()
                    self._output_enqueued_at[
                        (first.item.batch_id, first.item.index_in_batch)
                    ] = output_enqueued_at
                    await self.output_queue.put(
                        RolloutGroup(
                            rollouts=rollouts,
                            batch_id=first.item.batch_id,
                            index_in_batch=first.item.index_in_batch,
                        )
                    )
        finally:
            self.output_queue.shutdown()

    def _record_output_dwell(self, group: RolloutGroup) -> None:
        """Record how long a group waited between assembly and being yielded.

        Args:
            group: The group being yielded to the consumer.
        """
        key = (group.batch_id, group.index_in_batch)
        enqueued_at = self._output_enqueued_at.pop(key, 0.0)
        if enqueued_at:
            self.output_queue_dwell.append(time.monotonic() - enqueued_at)
        self.yielded_count += 1
        self.yielded_groups_per_env[self.gran_policy.env_of_index(group.index_in_batch)] += 1

    async def _next_group(self) -> RolloutGroup | None:
        """Pop the next group off output_queue.

        Returns:
            The next RolloutGroup, or None once the queue shuts down.
        """
        try:
            return await self.output_queue.get()
        except asyncio_QueueShutDown:
            return None

    @property
    def ready_batches(self) -> int:
        """Full batches banked and not yet dequeued for consumption."""
        return self.banked_batches - self.consumed_batches

    @trace_async_exceptions(verbose=True)
    async def stage_bank(self) -> None:
        """Bank complete batches cut from the consumption-ordered group stream."""
        order = {
            "G": self._consume_completion_order,
            "E": self._consume_env_units,
            "B": self._consume_batch_order,
        }[self.gran_policy.consumption]
        batch: list[RolloutGroup] = []
        try:
            async for group in order():
                batch.append(group)
                if len(batch) == self.gran_policy.num_groups_per_batch:
                    self.bank.put_nowait(batch)
                    self.banked_batches += 1
                    batch = []
            assert self.request.streaming or not (batch or self._consume_pending), (
                "Stream ended with groups not forming a full batch."
            )
        finally:
            self.bank.shutdown()

    async def stage_consume(self) -> AsyncIterator[RolloutGroup]:
        """Unwrap banked batches for the consumer.

        Yields:
            RolloutGroup: Groups ordered by the configured consumption mode.
        """
        while True:
            try:
                batch = await self.bank.get()
            except asyncio_QueueShutDown:
                return
            self.consumed_batches += 1
            for group in batch:
                self._record_output_dwell(group)
                yield group

    async def _consume_completion_order(self) -> AsyncIterator[RolloutGroup]:
        """G consumption: deliver each group as soon as it assembles.

        Yields:
            RolloutGroup: Groups in global completion order.
        """
        while (group := await self._next_group()) is not None:
            yield group

    async def _consume_env_units(self) -> AsyncIterator[RolloutGroup]:
        """Balanced-E consumption.

        Within each env, deliver groups in completion order, cut into env-units
        of num_groups_per_env[e] — each unit is the env's earliest unclaimed groups,
        which may span dispatch batches. One unit per env per delivered batch;
        a fast env's extra units wait until every env has served the current
        batch, so no env runs more than one delivered batch ahead.

        Yields:
            RolloutGroup: Groups in balanced env-unit order.
        """
        num_envs = len(self.gran_policy.num_groups_per_env)
        pending: list[list[RolloutGroup]] = [[] for _ in range(num_envs)]
        ready_units: list[deque[list[RolloutGroup]]] = [deque() for _ in range(num_envs)]
        delivered_units = [0] * num_envs
        current_batch = 0
        while (group := await self._next_group()) is not None:
            env_index = self.gran_policy.env_of_index(group.index_in_batch)
            pending[env_index].append(group)
            unit_size = self.gran_policy.num_groups_per_env[env_index]
            if len(pending[env_index]) >= unit_size:
                ready_units[env_index].append(pending[env_index][:unit_size])
                pending[env_index] = pending[env_index][unit_size:]
            progressed = True
            while progressed:
                progressed = False
                for env in range(num_envs):
                    if delivered_units[env] == current_batch and ready_units[env]:
                        for unit_group in ready_units[env].popleft():
                            yield unit_group
                        delivered_units[env] += 1
                        progressed = True
                if all(count > current_batch for count in delivered_units):
                    current_batch += 1
                    progressed = True

    async def _consume_batch_order(self) -> AsyncIterator[RolloutGroup]:
        """B consumption: deliver whole batches in dataset order.

        Yields:
            RolloutGroup: Batch b's groups sorted by index_in_batch, batches in order.
        """
        next_batch_id = 0
        pending = self._consume_pending
        while (group := await self._next_group()) is not None:
            pending.setdefault(group.batch_id, []).append(group)
            while (
                len(pending.get(next_batch_id, []))
                >= self.gran_policy.num_groups_per_batch
            ):
                batch = pending.pop(next_batch_id)
                batch.sort(key=lambda group: group.index_in_batch)
                next_batch_id += 1
                for group in batch:
                    yield group
                self.gate.release_after("consumed")
