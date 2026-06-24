# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Generic, TypeVar

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


@dataclass
class GroupRolloutParams:
    """Returned by agent.group_rollout.

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
    pass


class RolloutGenerator(Agent, ABC):
    """An agent that produces Rollout objects containing rollout string and associated reward."""

    @abstractmethod
    async def rollout(self, request: RolloutRequest) -> Rollout: ...

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        return await asyncio.gather(
            *[self.rollout(request=request) for _ in range(request.num_rollouts)]
        )


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
    async def rollout(self, request: RolloutRequest) -> TokenRollout: ...

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        return await asyncio.gather(
            *[self.rollout(request=request) for _ in range(request.num_rollouts)]
        )


@dataclass(frozen=True)
class _GranularityConfig:
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
    def consume_in_batch_order(self) -> bool:
        return self.consumption == "B"

    @staticmethod
    def _validate(request: GroupedRolloutRequest) -> None:
        assert not (
            request.submission_granularity == "B" and request.consumption_granularity == "G"
        ), "Batch submission with group consumption is not supported."
        if request.num_groups > 1 and request.submission_granularity == "B":
            assert not request.filter_groups_with_same_reward, (
                "Cannot use filter_groups_with_same_reward with num_groups > 1."
            )


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

    async def acquire_for(self, granularity: SubmissionGranularity) -> None:
        if self._submission == granularity:
            await self._sem.acquire()

    def release_after(self, state: ReleaseState) -> None:
        if self._release_on == state:
            self._sem.release()


@dataclass(frozen=True)
class _InferWorkItem:
    """One rollout's worth of work flowing from prepare to infer."""

    group_id: int
    rollout_idx: int
    batch_id: int
    index_in_batch: int
    params: GroupRolloutParams


@dataclass(frozen=True)
class _InferredItem:
    """One rollout post-inference, flowing from infer to assemble."""

    item: _InferWorkItem
    response: InferenceResponse


class _RolloutPipeline:
    """Per-call orchestrator for grouped rollout generation."""

    def __init__(
        self,
        agent: "GroupedRolloutGenerator",
        request: GroupedRolloutRequest,
        buffer_size: int,
        parallel_generation_tasks: int,
    ) -> None:
        self.agent = agent
        self.request = request
        self.gran_policy = _GranularityConfig.from_request(request)
        self.gate = _SubmissionGate(
            capacity=parallel_generation_tasks,
            submission=self.gran_policy.submission,
        )
        self.infer_queue = asyncio_Queue()
        self.assemble_queue = asyncio_Queue()
        self.output_queue = asyncio_Queue(maxsize=buffer_size if request.streaming else 0)

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
                    await self.gate.acquire_for("G")
                    params = await self.agent.group_rollout(self.request)
                    for rollout_idx in range(self.request.rollouts_per_group):
                        await self.gate.acquire_for("R")
                        await self.infer_queue.put(
                            _InferWorkItem(
                                group_id=group_id,
                                rollout_idx=rollout_idx,
                                batch_id=batch_id,
                                index_in_batch=index_in_batch,
                                params=params,
                            )
                        )
                    group_id += 1
        finally:
            self.infer_queue.shutdown()

    async def stage_infer(self) -> None:
        """Dispatch inference work into child tasks."""
        in_flight: set[asyncio.Task[None]] = set()
        try:
            while True:
                try:
                    item = await self.infer_queue.get()
                except asyncio_QueueShutDown:
                    break
                task = asyncio.create_task(self._infer_one(item))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
            await asyncio.gather(*in_flight, return_exceptions=True)
        finally:
            for task in in_flight:
                task.cancel()
            self.assemble_queue.shutdown()

    @trace_async_exceptions(verbose=True)
    async def _infer_one(self, item: _InferWorkItem) -> None:
        response = await self.request.inference_interface.agenerate(item.params.inference_request)
        self.gate.release_after("inferred")
        await self.assemble_queue.put(_InferredItem(item=item, response=response))

    async def stage_assemble(self) -> None:
        """Build complete rollout groups from inferred items."""
        pending: dict[int, list[_InferredItem]] = {}
        try:
            while True:
                try:
                    inferred = await self.assemble_queue.get()
                except asyncio_QueueShutDown:
                    break
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
                keep = (
                    not self.request.filter_groups_with_same_reward
                    or np.std([rollout.reward for rollout in rollouts]) > 1e-6
                )
                if keep:
                    first = completed[0]
                    await self.output_queue.put(
                        RolloutGroup(
                            rollouts=rollouts,
                            batch_id=first.item.batch_id,
                            index_in_batch=first.item.index_in_batch,
                        )
                    )
        finally:
            self.output_queue.shutdown()

    async def stage_consume(self) -> AsyncIterator[RolloutGroup]:
        if not self.gran_policy.consume_in_batch_order:
            while True:
                try:
                    yield await self.output_queue.get()
                except asyncio_QueueShutDown:
                    return

        next_batch_id = 0
        pending: dict[int, list[RolloutGroup]] = {}
        while True:
            try:
                group = await self.output_queue.get()
            except asyncio_QueueShutDown:
                return
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
    buffer_size: int = 10

    def __init__(self, *, parallel_generation_tasks: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if parallel_generation_tasks is not None:
            self.parallel_generation_tasks = parallel_generation_tasks

    @abstractmethod
    async def group_rollout(
        self,
        request: GroupedRolloutRequest,
    ) -> GroupRolloutParams:
        """Return the params for one group's rollouts.

        Called once per group by _RolloutPipeline.stage_prepare. The returned
        build_rollout closure is invoked once per inference response in
        _RolloutPipeline.stage_assemble.
        """
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
            buffer_size=self.buffer_size,
            parallel_generation_tasks=self.parallel_generation_tasks,
        )
        stage_prepare_task = asyncio.create_task(pipeline.stage_prepare())
        infer_task = asyncio.create_task(pipeline.stage_infer())
        assemble_task = asyncio.create_task(pipeline.stage_assemble())
        tasks = (stage_prepare_task, infer_task, assemble_task)

        try:
            async for group in pipeline.stage_consume():
                yield group
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
