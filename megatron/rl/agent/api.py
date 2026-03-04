# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel

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
    batch_results: bool = False


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
        grouped_rollouts: asyncio.Queue[list[Rollout]] = asyncio.Queue(
            maxsize=self.buffer_size if request.streaming else 0
        )
        submitted_groups = 0

        if request.batch_results:
            assert not request.filter_groups_with_same_reward, \
                "Filtering is not supported with batch_results"
            # Each worker gets a permit for a batch of num_groups groups.
            # The semaphore ensures that each batch only starts after the previous is consumed.
            groups_per_worker = request.num_groups
            num_workers = self.parallel_generation_tasks // groups_per_worker
            submission_gate = asyncio.Semaphore(num_workers)
        else:
            groups_per_worker = 1
            num_workers = self.parallel_generation_tasks
            submission_gate = None

        async def generate_and_enqueue(batch_id, index_in_batch):
            group = await self.group_rollout(request=request)
            if (
                not request.filter_groups_with_same_reward
                or np.std([r.reward for r in group]) > 1e-6
            ):
                for rollout in group:
                    rollout.submission_index = batch_id * groups_per_worker + index_in_batch
                    rollout.batch_id = batch_id
                await grouped_rollouts.put(group)
                return True
            return False

        @trace_async_exceptions(verbose=True)
        async def generate_task():
            nonlocal submitted_groups
            while request.streaming or submitted_groups < self.parallel_generation_tasks:
                if submission_gate is not None:
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
                        submitted_groups -= 1

        tasks = [asyncio.create_task(generate_task()) for _ in range(num_workers)]

        try:
            if request.batch_results:
                # Accumulate groups into ordered batches of num_groups.
                next_batch_id = 0
                pending: dict[int, list[list[Rollout]]] = {}
                while grouped_rollouts.qsize() > 0 or not all(task.done() for task in tasks):
                    group = await grouped_rollouts.get()
                    pending.setdefault(group[0].batch_id, []).append(group)
                    if len(pending.get(next_batch_id, [])) >= request.num_groups:
                        batch = pending.pop(next_batch_id)
                        batch.sort(key=lambda g: g[0].submission_index)
                        next_batch_id += 1
                        for g in batch:
                            yield g
                        submission_gate.release()
            else:
                while grouped_rollouts.qsize() > 0 or not all(task.done() for task in tasks):
                    yield await grouped_rollouts.get()
        finally:
            for task in tasks:
                task.cancel()


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
