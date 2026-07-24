# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Generic, NamedTuple, TypeVar

from pydantic import BaseModel

from ..__init__ import Request, TypeLookupable
from ..inference import (
    InferenceInterface,
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
)
from ..rollout_granularity import ConsumptionGranularity, SubmissionGranularity


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
    agent: "Agent"


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


class GroupedRolloutGenerator(Agent, ABC):
    """Agent contract consumed by RolloutPipeline to generate grouped rollouts (e.g. GRPO)."""

    @abstractmethod
    async def prepare_group_rollout(
        self,
        request: GroupedRolloutRequest,
        env_index: int = 0,
    ) -> GroupRolloutParams:
        """Return the params for one group's rollouts.

        Called once per group by `RolloutPipeline.stage_prepare`; the returned `build_rollout` is
        invoked once per inference response in `RolloutPipeline.stage_assemble`.

        Args:
            request: The grouped rollout request being served.
            env_index: An index of the environment this group belongs to.

        Returns:
            GroupRolloutParams carrying the inference request, the
            build_rollout closure, and the serving agent stamped on `agent`.
        """
        ...

    def rollout_group_layout(self, num_groups: int) -> list[int]:
        """Returns the groups each env contributes to every trainer batch, in env order.

        Args:
            num_groups: Total groups in one trainer batch.

        Returns:
            Positive per-env group counts summing to num_groups.
        """
        return [num_groups]


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
