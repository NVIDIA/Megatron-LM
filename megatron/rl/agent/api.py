# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Generic, NamedTuple, TypeVar

from ..__init__ import Request, TypeLookupable
from ..inference import InferenceInterface, InferenceRequest, InferenceResponse, LLMChatMessage
from ..rollout_granularity import ConsumptionGranularity, SubmissionGranularity
from ..types import (
    AgentBaseModel,
    GroupedRollouts,
    Rollout,
    RolloutGroup,
    Rollouts,
    TokenRollout,
)


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
    submission_granularity: SubmissionGranularity = "B"
    consumption_granularity: ConsumptionGranularity = "B"


class EpisodeResult(NamedTuple):
    """All per-turn responses of one (possibly multi-turn) episode plus the final conversation."""

    responses: list[InferenceResponse]
    conversation: list[LLMChatMessage]


class GroupRolloutParams(NamedTuple):
    """Returned by agent.prepare_group_rollout.

    One instance is created per group call and reused for all rollouts in that group.
    Every rollout is an episode: run_episode generates it (one or more turns), while
    build_rollout turns the completed episode into a Rollout.

    Args:
        run_episode: A callable that returns an EpisodeResult.
        build_rollout: A callable that returns a Rollout.
        problem_state: A dictionary of problem state used to restore the group.
    """

    run_episode: Callable[[], Awaitable[EpisodeResult]]
    build_rollout: Callable[[EpisodeResult], Awaitable[Rollout]]
    problem_state: dict | None = None


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


class EnvAllocation(NamedTuple):
    """One env's constant share of every trainer batch."""

    agent: "GroupedRolloutGenerator"
    env_id: str
    num_groups: int


class GroupedRolloutGenerator(Agent, ABC):
    """Agent contract consumed by RolloutPipeline to generate grouped rollouts (e.g. GRPO)."""

    @abstractmethod
    async def prepare_group_rollout(
        self, request: GroupedRolloutRequest, *, problem_state: dict | None = None
    ) -> GroupRolloutParams:
        """Return the params for one group's rollouts.

        Args:
            request: The grouped rollout request being served.
            problem_state: When None, draw a fresh problem as usual. When given, it
                is a state this agent previously returned on ``GroupRolloutParams``;
                prepare for that same problem instead of drawing a new one, so a
                restored group's missing members are regenerated against the prompt
                its existing members already answered.
        """
        ...

    def rollout_allocations(self, num_groups: int) -> list[EnvAllocation]:
        """Returns each env's per-trainer-batch allocation, in env order."""
        return [
            EnvAllocation(
                agent=self,
                env_id=getattr(self, "env_id", None) or "rollout",
                num_groups=num_groups,
            )
        ]

    def take_restored_group(self, env_id: str) -> RolloutGroup | None:
        """Return one recovered group for ``env_id``, if one is available."""
        return None


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
