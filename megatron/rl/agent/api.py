# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..__init__ import Request
from ..inference import InferenceInterface


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


class NextStateRequest(Request):
    """Request to enumerate all possible next states, useful for tree search."""

    current_state: list[str] | str
    inference_interface: list[InferenceInterface]


class NextStateResponse(AgentBaseModel):
    """Response containing next states for the provided states."""

    possible_next_states: list[list[str]] | list[str]


class Agent(ABC, AgentBaseModel):
    pass


class NextStateGenerator(Agent, ABC):
    """An agent that allows querying next state from a given state."""

    @abstractmethod
    async def get_next_state(self, request: NextStateRequest) -> NextStateResponse: ...


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
    """An interface to return grouped Rollout objects to support algorithms like GRPO."""

    @abstractmethod
    async def get_grouped_rollouts(self, request: GroupedRolloutRequest) -> list[list[Rollout]]: ...


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
