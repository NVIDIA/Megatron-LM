# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Awaitable, Callable, Generic, NamedTuple, TypeVar

import numpy as np
from pydantic import BaseModel

from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

from ..__init__ import Request, TypeLookupable
from ..inference import (
    InferenceInterface,
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
)
from ..rollout_granularity import (
    RELEASE_STATE_BY_SUBMISSION,
    ConsumptionGranularity,
    ReleaseState,
    SubmissionGranularity,
)


class AgentBaseModel(BaseModel, extra='allow'):
    pass


class RolloutRequest(Request):
    """Request to agent to generate Rollouts."""

    num_rollouts: int
    inference_interface: InferenceInterface
    validation: bool = False


class GroupedRolloutRequest(Request):
    """Request to agent to generate grouped Rollouts."""

    num_groups: int
    rollouts_per_group: int
    inference_interface: InferenceInterface
    validation: bool = False
    filter_groups_with_same_reward: bool = False
    streaming: bool = False
    submission_granularity: SubmissionGranularity = "B"
    consumption_granularity: ConsumptionGranularity = "B"


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: list[str]
    prompt_length: list[int] | None = None
    reward: float = None
    env_id: str = ''
    problem_id: str | None = None
    policy_epoch: list[list[tuple[int, int]]]
    kv_cache_epoch: list[list[tuple[int, int]]]
    num_evictions: list[int]


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[list[int]]
    reward: list[float] | float
    generation_mask: list[list[bool]] | None = None
    logprobs: list[list[float]] | None = None
    env_id: str = ''
    problem_id: str | None = None
    policy_epoch: list[list[tuple[int, int]]]
    kv_cache_epoch: list[list[tuple[int, int]]]
    num_evictions: list[int]


Rollouts = list[TokenRollout | Rollout]


class RolloutGroup(AgentBaseModel):
    """A group of rollouts (e.g. multiple completions for one prompt) with batch metadata."""

    rollouts: Rollouts
    batch_id: int = 0
    index_in_batch: int = 0

    def __iter__(self):
        return iter(self.rollouts)

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]


GroupedRollouts = list[RolloutGroup]


class GroupRolloutParams(NamedTuple):
    """Returned by agent.prepare_group_rollout.

    One instance is created per group call and reused for all rollouts in that group.
    """

    inference_request: InferenceRequest
    build_rollout: Callable[[InferenceResponse], Awaitable[Rollout]]


class ContrastiveRollout(AgentBaseModel):
    """Contrastive/Preference data for language-based Rollout."""

    chosen_trajectory: list[str]
    rejected_trajectory: list[str]


class Head2HeadRolloutRequest(Request):
    num_rollouts: int
    inference_interface: list[InferenceInterface]
    validation: bool = False


class EvaluationRequest(Request):
    """Request to evaluate N prompts, optionally distributed across ranks."""

    inference_interface: InferenceInterface
    num_prompts: int
    rank_info: tuple[int, int] | None = (
        None  # (rank, total_ranks) if distributed, None for full evaluation
    )
    validation: bool = True


class EvaluationResult(AgentBaseModel):
    prompt: str | list[LLMChatMessage]
    response: str | LLMChatMessage


class RewardEvaluationResult(EvaluationResult):
    reward: float
    problem_id: str | None = None


T = TypeVar('T', bound=EvaluationResult)


class EvaluationResponse(AgentBaseModel, TypeLookupable, Generic[T]):
    env_id: str
    results: list[T]

    def metrics(self):
        raise NotImplementedError(f"{type(self)} did not provide metric aggregation.")


class Agent(ABC, AgentBaseModel):

    @abstractmethod
    async def get_rollout_response(
        self,
        request: "RolloutRequest | GroupedRolloutRequest | EvaluationRequest",
        inference_request: InferenceRequest,
    ) -> InferenceResponse:
        """Obtain the model response for a single rollout. Subclasses implement how."""
        ...


class RolloutGenerator(Agent, ABC):
    """An agent that produces Rollout objects containing rollout string and associated reward."""

    @abstractmethod
    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]: ...


class ContrastiveRolloutGenerator(Agent, ABC):
    """An agent that produces ContrastiveRollout objects containing two rollout strings, one chosen and one rejected."""

    @abstractmethod
    async def get_contrastive_rollouts(
        self, request: RolloutRequest
    ) -> list[ContrastiveRollout]: ...


class TokenizedRolloutGenerator(Agent, ABC):
    """An agent that produces TokenRollout objects containing rollout token ids and associated rewards.

    Optionally can also provide generation masks to indicate which tokens were generated and token masks to indicate which
    tokens were possible at any given step.
    """

    @abstractmethod
    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]: ...


class _GranularityConfig(NamedTuple):
    submission: SubmissionGranularity
    consumption: ConsumptionGranularity
    num_groups_per_batch: int

    @classmethod
    def from_request(cls, request: GroupedRolloutRequest) -> "_GranularityConfig":
        cls._validate(request)
        return cls(
            submission=request.submission_granularity,
            consumption=request.consumption_granularity,
            num_groups_per_batch=request.num_groups,
        )

    @property
    def prevent_dataset_reorder(self) -> bool:
        return self.consumption == "B"

    @staticmethod
    def _validate(request: GroupedRolloutRequest) -> None:
        assert not (
            request.submission_granularity == "B" and request.consumption_granularity == "G"
        ), "Batch submission with group consumption is not supported."


class _SubmissionGate:
    """Gate capacity is measured in units of the configured submission granularity."""

    def __init__(
        self,
        *,
        capacity: int,
        submission: SubmissionGranularity,
    ) -> None:
        self._sem = asyncio.Semaphore(capacity)
        self._submission = submission
        self._release_on = RELEASE_STATE_BY_SUBMISSION[submission]
        self.capacity = capacity
        # Observability counters, updated only on the configured submission granularity
        # (the only path that touches the semaphore). `held` counts slots currently held;
        # `prepare_blocked_seconds` accumulates time submitters
        # (stage_prepare and stage_filter regeneration) spent waiting on the semaphore.
        self.held = 0
        self.prepare_blocked_seconds = 0.0
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire_for(self, granularity: SubmissionGranularity) -> None:
        if self._submission == granularity:
            start = time.monotonic()
            await self._sem.acquire()
            self.prepare_blocked_seconds += time.monotonic() - start
            self.held += 1
            self.acquire_calls += 1

    def release_after(self, state: ReleaseState) -> None:
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
    prepared_at: float = 0.0
    infer_dequeued_at: float = 0.0


class _InferredItem(NamedTuple):
    """One rollout post-inference, flowing from infer to assemble."""

    item: _InferWorkItem
    response: InferenceResponse
    inferred_at: float = 0.0


class _AssembledGroup(NamedTuple):
    """One complete group flowing from assemble to filter."""

    group: RolloutGroup
    assembled_at: float = 0.0


class _RolloutPipeline:
    """Per-call orchestrator for grouped rollout generation."""

    def __init__(
        self,
        agent: "GroupedRolloutGenerator",
        request: GroupedRolloutRequest,
        parallel_generation_tasks: int,
    ) -> None:
        self.agent = agent
        self.request = request
        self.gran_policy = _GranularityConfig.from_request(request)
        self.gate = _SubmissionGate(
            capacity=parallel_generation_tasks,
            submission=self.gran_policy.submission,
        )
        rollouts_per_submission_unit = {
            "R": 1,
            "G": request.rollouts_per_group,
            "B": self.gran_policy.num_groups_per_batch * request.rollouts_per_group,
        }[self.gran_policy.submission]
        self.num_infer_workers = parallel_generation_tasks * rollouts_per_submission_unit
        if not request.streaming:
            self.num_infer_workers = min(
                self.num_infer_workers, request.num_groups * request.rollouts_per_group
            )
        self.infer_queue = asyncio_Queue()
        self.assemble_queue = asyncio_Queue()
        self.filter_queue = asyncio_Queue()
        self.output_queue = asyncio_Queue()
        # Regenerated groups draw ids from a negative namespace to avoid collisions.
        self._next_regen_group_id = -1
        # Groups submitted but not yet delivered by stage_filter.
        self._groups_in_flight = 0
        self._prepare_done = False
        # Buffer of pending groups (incomplete groups being filled by
        # stage_assemble). Held here so metric collection can report its size.
        self._assemble_pending: dict[int, list[_InferredItem]] = {}
        # Pending groups waiting for their batch to fill in stage_consume
        # (only populated when prevent_dataset_reorder is True).
        self._consume_pending: dict[int, list[RolloutGroup]] = {}
        # Per-group "output entry" times, keyed by (batch_id, index_in_batch),
        # so stage_consume can compute output_queue_dwell when yielding.
        self._output_enqueued_at: dict[tuple[int, int], float] = {}
        # Observability accumulators. Measured here; snapshot/reset and
        # wandb formatting happen in rl_utils during metric logging.
        self.infer_queue_dwell: list[float] = []
        self.engine_dwell: list[float] = []
        self.assemble_queue_dwell: list[float] = []
        self.filter_queue_dwell: list[float] = []
        self.output_queue_dwell: list[float] = []
        self.prepared_count = 0
        self.inferred_count = 0
        self.assembled_count = 0
        self.filtered_count = 0
        self.yielded_count = 0

    async def _submit_group(self, *, group_id: int, batch_id: int, index_in_batch: int) -> None:
        """Acquire sub-batch gate slots and enqueue one group's inference items."""
        await self.gate.acquire_for("G")
        params: GroupRolloutParams = await self.agent.prepare_group_rollout(self.request)

        for rollout_idx in range(self.request.rollouts_per_group):
            await self.gate.acquire_for("R")
            item = _InferWorkItem(
                group_id=group_id,
                rollout_idx=rollout_idx,
                batch_id=batch_id,
                index_in_batch=index_in_batch,
                params=params,
                prepared_at=time.monotonic(),
            )
            await self.infer_queue.put(item)
            self.prepared_count += 1

    def _maybe_close_intake(self) -> None:
        """Shut down infer_queue once no work can ever be submitted again."""
        if self._prepare_done and self._groups_in_flight <= 0:
            self.infer_queue.shutdown()

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

                for index_in_batch in range(self.gran_policy.num_groups_per_batch):
                    self._groups_in_flight += 1
                    await self._submit_group(
                        group_id=group_id, batch_id=batch_id, index_in_batch=index_in_batch
                    )
                    group_id += 1
        except BaseException:
            self.infer_queue.shutdown()
            raise
        finally:
            self._prepare_done = True
            self._maybe_close_intake()

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
        response = await self.agent.get_rollout_response(
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
                self.gate.release_after("assembled")
                self.assembled_count += 1
                first = completed[0]
                await self.filter_queue.put(
                    _AssembledGroup(
                        group=RolloutGroup(
                            rollouts=rollouts,
                            batch_id=first.item.batch_id,
                            index_in_batch=first.item.index_in_batch,
                        ),
                        assembled_at=time.monotonic(),
                    )
                )
        finally:
            self.filter_queue.shutdown()

    async def stage_filter(self) -> None:
        """Deliver assembled groups, regenerating any dropped by the reward filter."""
        try:
            while True:
                try:
                    assembled = await self.filter_queue.get()
                except asyncio_QueueShutDown:
                    break
                dequeued_at = time.monotonic()
                if assembled.assembled_at:
                    self.filter_queue_dwell.append(dequeued_at - assembled.assembled_at)
                group = assembled.group
                if self._should_drop(group):
                    self.filtered_count += 1
                    try:
                        await self._regenerate_group(group)
                    except asyncio_QueueShutDown:
                        # Intake closed mid-regeneration (teardown or prepare
                        # failure): the slot can no longer be refilled.
                        self._groups_in_flight -= 1
                        self._maybe_close_intake()
                    continue
                self._output_enqueued_at[(group.batch_id, group.index_in_batch)] = (
                    time.monotonic()
                )
                await self.output_queue.put(group)
                self._groups_in_flight -= 1
                self._maybe_close_intake()
        finally:
            self.output_queue.shutdown()

    def _should_drop(self, group: RolloutGroup) -> bool:
        """A group with zero reward variance carries no learning signal."""
        if not self.request.filter_groups_with_same_reward:
            return False
        return np.std([rollout.reward for rollout in group.rollouts]) <= 1e-6

    async def _regenerate_group(self, dropped: RolloutGroup) -> None:
        """Resubmit a replacement group for a dropped group's batch slot."""
        group_id = self._next_regen_group_id
        self._next_regen_group_id -= 1
        await self._submit_group(
            group_id=group_id, batch_id=dropped.batch_id, index_in_batch=dropped.index_in_batch
        )

    def _record_output_dwell(self, group: RolloutGroup) -> None:
        """Record how long a group sat in output_queue before being yielded."""
        key = (group.batch_id, group.index_in_batch)
        enqueued_at = self._output_enqueued_at.pop(key, 0.0)
        if enqueued_at:
            self.output_queue_dwell.append(time.monotonic() - enqueued_at)
        self.yielded_count += 1

    async def stage_consume(self) -> AsyncIterator[RolloutGroup]:
        if not self.gran_policy.prevent_dataset_reorder:
            while True:
                try:
                    group = await self.output_queue.get()
                except asyncio_QueueShutDown:
                    return
                self._record_output_dwell(group)
                yield group

        next_batch_id = 0
        pending = self._consume_pending
        while True:
            try:
                group = await self.output_queue.get()
            except asyncio_QueueShutDown:
                return
            self._record_output_dwell(group)
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


class GroupedRolloutGenerator(Agent, ABC):
    """An interface to return grouped Rollout objects to support algorithms like GRPO."""

    parallel_generation_tasks: int = 512

    def __init__(self, *, parallel_generation_tasks: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if parallel_generation_tasks is not None:
            self.parallel_generation_tasks = parallel_generation_tasks

    @abstractmethod
    async def prepare_group_rollout(
        self,
        request: GroupedRolloutRequest,
    ) -> GroupRolloutParams:
        """Return the params for one group's rollouts."""
        ...

    async def get_grouped_rollouts(
        self, request: GroupedRolloutRequest
    ) -> AsyncIterator[RolloutGroup]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."
        pipeline = _RolloutPipeline(
            agent=self,
            request=request,
            parallel_generation_tasks=self.parallel_generation_tasks,
        )
        # Expose the live pipeline for observability; rl_utils reads its
        # queue sizes, gate state, and timing accumulators during logging.
        self._active_pipeline = pipeline
        stage_prepare_task = asyncio.create_task(pipeline.stage_prepare())
        infer_task = asyncio.create_task(pipeline.stage_infer())
        assemble_task = asyncio.create_task(pipeline.stage_assemble())
        filter_task = asyncio.create_task(pipeline.stage_filter())
        tasks = (stage_prepare_task, infer_task, assemble_task, filter_task)

        try:
            async for group in pipeline.stage_consume():
                yield group
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._active_pipeline = None


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
