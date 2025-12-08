"""
Intermediate representations for pipeline simulation.

This module defines data structures and enumerations for representing computations and actions in a
pipeline parallel neural network training system. Several data structures (e.g. ActionType) are
copied from Pytorch, please check the license and copyright information in the original repository.

Classes:
    ActionType: Enumeration of different computation/communication types in the pipeline.
    Action: Represents a specific action to be performed on a chunk.
"""

import re
from enum import Enum, auto
from typing import NamedTuple, Optional


class ActionType(Enum):
    """Types of actions that can be performed in the pipeline."""

    FORWARD = auto()
    BACKWARD_INPUT = auto()
    BACKWARD_WEIGHT = auto()
    SEND_F = auto()
    RECV_F = auto()
    SEND_B = auto()
    RECV_B = auto()
    FULL_BACKWARD = auto()

    def __str__(self) -> str:
        m = {
            FORWARD: "F",
            BACKWARD_INPUT: "I",
            BACKWARD_WEIGHT: "W",
            SEND_F: "SEND_F",
            RECV_F: "RECV_F",
            SEND_B: "SEND_B",
            RECV_B: "RECV_B",
            FULL_BACKWARD: "B",
        }
        return m.get(self, f"Unknown({self.value})")

    @classmethod
    def from_str(cls, action: str) -> "ActionType":
        """Convert string representation to ActionType."""
        m = {
            "F": FORWARD,
            "I": BACKWARD_INPUT,
            "W": BACKWARD_WEIGHT,
            "SEND_F": SEND_F,
            "RECV_F": RECV_F,
            "SEND_B": SEND_B,
            "RECV_B": RECV_B,
            "B": FULL_BACKWARD,
        }
        if action in m:
            return m[action]
        raise ValueError(f"Invalid action type: {action}")


# Global constants for convenience
FORWARD = ActionType.FORWARD
BACKWARD_INPUT = ActionType.BACKWARD_INPUT
BACKWARD_WEIGHT = ActionType.BACKWARD_WEIGHT
SEND_F = ActionType.SEND_F
RECV_F = ActionType.RECV_F
SEND_B = ActionType.SEND_B
RECV_B = ActionType.RECV_B
FULL_BACKWARD = ActionType.FULL_BACKWARD

# Convenience shorthand for compute actions only
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

# Regular expression for parsing action strings
_ACTION_REGEX = re.compile(r"(.+)(F|I|B|W|SEND_F|RECV_F|SEND_B|RECV_B)(.+)")


class Action(NamedTuple):
    """
    An action to be performed on a chunk.

    Attributes:
        stage_id: The id of the stage in the pipeline.
        action_type: The type of action to be performed.
        data_id: The id of the data chunk.
    """

    stage_id: int
    action_type: ActionType
    data_id: str | int | tuple[int, ...]

    def __repr__(self) -> str:
        return f"{self.stage_id}{self.action_type}{self.data_id}"

    @staticmethod
    def from_str(action_string: str) -> Optional["Action"]:
        """
        Parse a string representation of an Action.

        Args:
            action_string: String formatted as [stage_id][action_type][chunk_id]
                e.g. `2F0`, `3SEND_F1`

        Returns:
            The parsed Action object, or None if the string is empty.

        Raises:
            ValueError: If the action string format is invalid.
        """
        action_string = action_string.strip()
        if not action_string:
            return None

        match = _ACTION_REGEX.match(action_string)
        if not match:
            raise ValueError(
                f"Invalid action string: {action_string}, should be formatted as "
                f"[stage_id][action_type][data_id] (e.g. 2F0, 5B(2, 3))."
            )

        stage_id, action_type, data_id = match.groups()
        return Action(
            stage_id=int(stage_id),
            action_type=ActionType.from_str(action_type),
            data_id=data_id,
        )


class Stats(NamedTuple):
    """
    Timing information for an action.

    Attributes:
        start_time: The start time of the action.
        end_time: The end time of the action.
    """

    start_time: int | float
    end_time: int | float

    def __repr__(self) -> str:
        return f"({self.start_time}, {self.end_time})"
