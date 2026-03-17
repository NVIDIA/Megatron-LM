# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel

from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

from ..__init__ import Request, TypeLookupable
from ..inference import (
    InferenceInterface,
    LLMChatMessage,
    ReturnsRaw,
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
    enforce_order: bool = False


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: list[str]
    prompt_length: list[int] | None = None
    reward: float = None
    env_id: str = ''
    problem_id: str | None = None
    policy_staleness: list[list[int]]
    kv_cache_staleness: list[list[int]]
    completed_at_step: list[int]
    num_evictions: list[int]


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[list[int]]
    reward: list[float] | float
    generation_mask: list[list[bool]] | None = None
    logprobs: list[list[float]] | None = None
    env_id: str = ''
    problem_id: str | None = None
    policy_staleness: list[list[int]]
    kv_cache_staleness: list[list[int]]
    completed_at_step: list[int]
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


class GroupedRolloutGenerator(Agent, ABC):
    """An interface to return grouped Rollout objects to support algorithms like GRPO."""

    parallel_generation_tasks: int = 512
    buffer_size: int = 10

    def __init__(self, *, parallel_generation_tasks: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if parallel_generation_tasks is not None:
            self.parallel_generation_tasks = parallel_generation_tasks

    @abstractmethod
    async def group_rollout(self, request: GroupedRolloutRequest) -> list[Rollout]: ...

    async def get_grouped_rollouts(self, request: GroupedRolloutRequest):
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        # When streaming, use buffer_size to create backpressure
        # for balanced generation in a multi-task setting.
        grouped_rollouts: asyncio_Queue[RolloutGroup] = asyncio_Queue(
            maxsize=self.buffer_size if request.streaming else 0
        )
        submitted_groups = 0

        # num_groups controls how many groups each worker generates and yields together.
        # When it's 1, the semaphore is a no-op.
        groups_per_worker = request.num_groups
        if groups_per_worker > 1:
            assert not request.filter_groups_with_same_reward, \
                "Cannot use filter_groups_with_same_reward with num_groups > 1."
        assert self.parallel_generation_tasks >= groups_per_worker, \
            f"{self.parallel_generation_tasks=} must be >= {groups_per_worker=}"
        num_workers = self.parallel_generation_tasks // groups_per_worker
        unused = self.parallel_generation_tasks % groups_per_worker
        if unused:
            logging.warning(
                f"parallel_generation_tasks ({self.parallel_generation_tasks}) is not "
                f"divisible by num_groups ({groups_per_worker}); "
                f"{unused} generation task(s) will be unused."
            )
        submission_gate = asyncio.Semaphore(num_workers)

        async def generate_and_enqueue(batch_id, index_in_batch):
            group = await self.group_rollout(request=request)
            if (
                not request.filter_groups_with_same_reward
                or np.std([r.reward for r in group]) > 1e-6
            ):
                await grouped_rollouts.put(
                    RolloutGroup(rollouts=group, batch_id=batch_id, index_in_batch=index_in_batch)
                )
                return True
            return False

        @trace_async_exceptions(verbose=True)
        async def generate_task():
            nonlocal submitted_groups
            while request.streaming or submitted_groups < self.parallel_generation_tasks:
                await submission_gate.acquire()
                batch_id = submitted_groups // groups_per_worker
                submitted_groups += groups_per_worker
                if groups_per_worker > 1:
                    await asyncio.gather(*[
                        generate_and_enqueue(batch_id, i)
                        for i in range(groups_per_worker)
                    ])
                else:
                    if not await generate_and_enqueue(batch_id, 0):
                        submitted_groups -= groups_per_worker
                        submission_gate.release()

        tasks = [asyncio.create_task(generate_task()) for _ in range(num_workers)]

        async def shutdown_queue_when_done():
            """Wait for all workers to finish, then shut down the queue."""
            await asyncio.gather(*tasks)
            grouped_rollouts.shutdown()

        shutdown_task = asyncio.create_task(shutdown_queue_when_done())

        try:
            next_batch_id = 0
            pending: dict[int, GroupedRollouts] = {}
            while True:
                try:
                    group = await grouped_rollouts.get()
                except asyncio_QueueShutDown:
                    break
                if request.enforce_order:
                    # Accumulate groups and enforce submission order across batches.
                    pending.setdefault(group.batch_id, []).append(group)
                    while len(pending.get(next_batch_id, [])) >= groups_per_worker:
                        batch = pending.pop(next_batch_id)
                        batch.sort(key=lambda g: g.index_in_batch)
                        next_batch_id += 1
                        for g in batch:
                            yield g
                        submission_gate.release()
                else:
                    # Yield groups as soon as they're completed.
                    yield group
                    submission_gate.release()
        finally:
            shutdown_task.cancel()
            for task in tasks:
                task.cancel()


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
