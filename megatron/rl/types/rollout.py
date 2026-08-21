# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from typing import TypeAlias

from pydantic import BaseModel

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


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[list[int]]
    reward: list[float] | float
    generation_mask: list[list[bool]] | None = None
    logprobs: list[list[float]] | None = None
    env_id: str = ''
    problem_id: str | None = None
    completion_ids: list[str] = []
    generation_cap: int | None = None
    rollout_status: str = 'ok'
    failure_reason: str | None = None


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
