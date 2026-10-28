# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from typing import TypeAlias

from pydantic import BaseModel, model_validator

#: An environment identifier, as reported by an agent's ``env_id`` attribute
#: (see ``WeightedMultiTask._env_ids``).
EnvId: TypeAlias = str

KNOWN_ROLLOUT_STATUSES = ('ok', 'placeholder', 'masked', 'graded')


class AgentBaseModel(BaseModel, extra='allow'):
    """Base model for agent data types."""


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: list[str]
    prompt_length: list[int] | None = None
    reward: float | None = None
    env_id: str = ''
    problem_id: str | None = None
    rollout_status: str = 'ok'
    failure_reason: str | None = None

    @property
    def is_placeholder(self) -> bool:
        """An empty trajectory is the placeholder left by a failed episode."""
        return not self.trajectory


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[list[int]] | None
    reward: list[float] | float
    generation_mask: list[list[bool]] | None = None
    logprobs: list[list[float]] | None = None
    env_id: str = ''
    problem_id: str | None = None
    completion_ids: list[str] = []
    policy_epochs: list[list[tuple[int, int]]] | None = None
    kv_cache_epochs: list[list[tuple[int, int]]] | None = None
    num_evictions: list[int] | None = None
    generation_cap: int | None = None
    rollout_status: str = 'ok'
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _one_request_per_turn(self):
        if self.trajectory and len(self.completion_ids) != len(self.trajectory):
            raise ValueError(
                f"TokenRollout has {len(self.trajectory)} turns but "
                f"{len(self.completion_ids)} completion ids; each turn needs its request id"
            )
        return self

    @property
    def is_placeholder(self) -> bool:
        """A failed episode: no turns; its completion ids name the calls it made."""
        return self.trajectory is not None and not self.trajectory


Rollouts = list[TokenRollout | Rollout]


class RolloutGroup(AgentBaseModel):
    """A group of rollouts (e.g. multiple completions for one prompt) with batch metadata."""

    rollouts: Rollouts
    batch_id: int = 0
    index_in_batch: int = 0
    uid: str | None = None

    def __iter__(self):
        return iter(self.rollouts)

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]


GroupedRollouts = list[RolloutGroup]

#: Maps each ``EnvId`` to a FIFO queue of completed ``RolloutGroup``s for that env.
#: Used by ``WeightedMultiTask`` to consume restored groups before generating fresh
#: groups for the same environment.
GroupQueuesPerEnv: TypeAlias = dict[EnvId, deque[RolloutGroup]]
