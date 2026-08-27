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

    @property
    def is_placeholder(self) -> bool:
        """An empty trajectory is the placeholder left by a failed episode."""
        return not self.trajectory


Rollouts = list[TokenRollout | Rollout]


class RolloutGroup(AgentBaseModel):
    """A group of rollouts (e.g. multiple completions for one prompt) with batch metadata.

    A group restored from the rollout bank may be *incomplete*: a kill can land
    after some members were persisted but before the group filled. Such a group
    carries ``problem_state`` (what the agent needs to regenerate the rest against
    the same prompt) and ``member_indices`` (the slots its members occupy, so the
    complement can be regenerated).

    Args:
        rollouts: The rollouts in this group.
        batch_id: The batch ID of this group.
        index_in_batch: The index of this group in the batch.
        uid: The UID of this group.
        problem_state: What the agent needs to regenerate this group's remaining
            members against the same prompt. Present on restored incomplete groups.
        member_indices: The slot each member occupies, parallel to ``rollouts``.
            Defaults to the positional slots, so there is one source of truth about
            which slots are filled rather than a fallback at each read site.
    """

    rollouts: Rollouts
    batch_id: int = 0
    index_in_batch: int = 0
    uid: str | None = None
    problem_state: dict | None = None
    member_indices: list[int] | None = None

    @model_validator(mode="after")
    def _fill_member_indices(self) -> "RolloutGroup":
        """Default the slots to positional, and reject a list that disagrees."""
        if self.member_indices is None:
            self.member_indices = list(range(len(self.rollouts)))
        elif len(self.member_indices) != len(self.rollouts):
            raise ValueError(
                f"member_indices has {len(self.member_indices)} slot(s) but the group has "
                f"{len(self.rollouts)} rollout(s); they must correspond one-to-one."
            )
        return self

    def is_complete(self, rollouts_per_group: int) -> bool:
        """Whether this group has every member it needs to be trained on."""
        return len(self.rollouts) >= rollouts_per_group

    def missing_indices(self, rollouts_per_group: int) -> list[int]:
        """
        Slots that still need to be generated, in ascending order.

        Example:
        >>> group = RolloutGroup(rollouts=[r0, r2], member_indices=[0, 2])
        >>> group.missing_indices(4)
        [1, 3]
        """
        occupied = set(self.member_indices)
        return [index for index in range(rollouts_per_group) if index not in occupied]

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
