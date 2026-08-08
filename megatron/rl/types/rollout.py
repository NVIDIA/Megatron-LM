# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from typing import TypeAlias

from pydantic import BaseModel

#: An environment identifier, as reported by an agent's ``env_id`` attribute
#: (see ``WeightedMultiTask._env_ids``).
EnvId: TypeAlias = str

#: Maps each ``EnvId`` to the number of groups to generate for that env. Used to
#: override the default weight-proportional split of ``num_groups``.
GroupsPerEnv: TypeAlias = dict[EnvId, int]


class AgentBaseModel(BaseModel, extra='allow'):
    """Base model for agent data types."""


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: list[str]
    prompt_length: list[int] | None = None
    reward: float | None = None
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
    # When set, replaces this rollout's group-normalized advantage with a fixed
    # value after group normalization in calculate_grpo_advantages.
    advantage_override: float | None = None


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
#: Used by the durable rollout bank to bucket restored groups (and buffer streaming
#: overflow) per env so injection can respect per-env weight targets.
GroupQueuesPerEnv: TypeAlias = dict[EnvId, deque[RolloutGroup]]
