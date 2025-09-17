# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable

import numpy as np
from pydantic import BaseModel

from ..__init__ import Request, trace_async_exceptions
from ..inference import (
    ChatInferenceInterface,
    ChatInferenceRequest,
    InferenceInterface,
    InferenceRequest,
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


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: str
    prompt_length: int | None = None
    reward: float = None
    env_id: str | None = None
    problem_id: str | None = None


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[int]
    reward: list[float] | float
    generation_mask: list[list[int]] | list[bool] | None = None
    logprobs: list[float] | None = None
    env_id: str | None = None
    problem_id: str | None = None


class ContrastiveRollout(AgentBaseModel):
    """Contrastive/Preference data for language-based Rollout."""

    chosen_trajectory: str
    rejected_trajectory: str


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


class EvaluationResponse(AgentBaseModel):
    env_id: str | None = None
    results: list[AgentBaseModel]

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

        if isinstance(request.inference_interface, ChatInferenceInterface):
            self.chat_mode = True
        else:
            self.chat_mode = False

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

        if isinstance(request.inference_interface, ChatInferenceInterface):
            self.chat_mode = True
        else:
            self.chat_mode = False

        return await asyncio.gather(
            *[self.rollout(request=request) for _ in range(request.num_rollouts)]
        )


class GroupedRolloutGenerator(Agent, ABC):
    """An interface to return grouped Rollout objects to support algorithms like GRPO."""

    parallel_generation_tasks: int = 512
    buffer_size: int = 10

    @abstractmethod
    async def group_rollout(self, request: GroupedRolloutRequest) -> list[Rollout]: ...

    async def get_grouped_rollouts(self, request: GroupedRolloutRequest):
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        if isinstance(request.inference_interface, ChatInferenceInterface):
            self.chat_mode = True
        else:
            self.chat_mode = False

        # If num_groups is -1, we generate a stream of groups.
        # The buffer size is used to create backpressure for each agent in order to balance group generation in a multi-task setting.
        grouped_rollouts: asyncio.Queue[list[Rollout]] = asyncio.Queue(
            maxsize=self.buffer_size if request.num_groups < 0 else 0
        )
        submitted_groups = 0

        @trace_async_exceptions
        async def group_task():
            nonlocal submitted_groups
            while request.num_groups == -1 or submitted_groups < request.num_groups:
                submitted_groups += 1
                group = await self.group_rollout(request=request)
                if (
                    not request.filter_groups_with_same_reward
                    or np.std([r.reward for r in group]) > 1e-6
                ):
                    await grouped_rollouts.put(group)
                else:
                    submitted_groups -= 1

        tasks = [asyncio.create_task(group_task()) for _ in range(self.parallel_generation_tasks)]

        try:
            while grouped_rollouts.qsize() > 0 or not all(task.done() for task in tasks):
                yield await grouped_rollouts.get()
        finally:
            for task in tasks:
                task.cancel()


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
