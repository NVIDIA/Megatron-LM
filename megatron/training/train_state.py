# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bridge-compatible mutable training state for Megatron-LM checkpoints."""

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from megatron.core.msc_utils import maybe_msc

TRAIN_STATE_FILENAME = "train_state.pt"


@dataclass
class TrainState(Stateful):
    """Mutable training progress stored alongside a model checkpoint.

    The serialized tensor dictionary intentionally matches Megatron Bridge's
    ``TrainState`` schema so either project can resume the other's checkpoint.
    """

    step: int = 0
    consumed_train_samples: int = 0
    skipped_train_samples: int = 0
    consumed_valid_samples: int = 0
    floating_point_operations_so_far: int = 0
    do_train: bool = False
    do_valid: bool = False
    do_test: bool = False

    @classmethod
    def from_args(cls, args: Any, step: int, floating_point_operations_so_far: int) -> "TrainState":
        """Create a train state from Megatron-LM's mutable arguments.

        Args:
            args: Megatron-LM argument namespace.
            step: Completed training iteration.
            floating_point_operations_so_far: FLOPs accumulated by the run.

        Returns:
            A populated train state.
        """
        return cls(
            step=step,
            consumed_train_samples=getattr(args, "consumed_train_samples", 0),
            skipped_train_samples=getattr(args, "skipped_train_samples", 0),
            consumed_valid_samples=getattr(args, "consumed_valid_samples", 0),
            floating_point_operations_so_far=floating_point_operations_so_far,
            do_train=getattr(args, "do_train", False),
            do_valid=getattr(args, "do_valid", False),
            do_test=getattr(args, "do_test", False),
        )

    def apply_to_args(self, args: Any) -> None:
        """Copy mutable training fields into a Megatron-LM argument namespace.

        Args:
            args: Megatron-LM argument namespace to update.
        """
        args.consumed_train_samples = self.consumed_train_samples
        args.skipped_train_samples = self.skipped_train_samples
        args.consumed_valid_samples = self.consumed_valid_samples
        args.do_train = self.do_train
        args.do_valid = self.do_valid
        args.do_test = self.do_test

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the training state using the Megatron Bridge schema."""
        return {
            "step": torch.tensor(self.step, dtype=torch.int64),
            "consumed_train_samples": torch.tensor(self.consumed_train_samples, dtype=torch.int64),
            "skipped_train_samples": torch.tensor(self.skipped_train_samples, dtype=torch.int64),
            "consumed_valid_samples": torch.tensor(self.consumed_valid_samples, dtype=torch.int64),
            "floating_point_operations_so_far": torch.tensor(
                self.floating_point_operations_so_far, dtype=torch.float64
            ),
            "do_train": torch.tensor(self.do_train, dtype=torch.bool),
            "do_valid": torch.tensor(self.do_valid, dtype=torch.bool),
            "do_test": torch.tensor(self.do_test, dtype=torch.bool),
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore the training state from a Bridge-compatible tensor dictionary.

        Args:
            state_dict: Serialized mutable training fields.
        """
        self.step = state_dict["step"].item()
        self.consumed_train_samples = state_dict["consumed_train_samples"].item()
        self.skipped_train_samples = state_dict["skipped_train_samples"].item()
        self.consumed_valid_samples = state_dict["consumed_valid_samples"].item()
        self.floating_point_operations_so_far = state_dict[
            "floating_point_operations_so_far"
        ].item()
        self.do_train = state_dict["do_train"].item()
        self.do_valid = state_dict["do_valid"].item()
        self.do_test = state_dict["do_test"].item()


def save_train_state(train_state: TrainState, filename: str | PathLike[str]) -> None:
    """Write a Bridge-compatible train-state sidecar.

    Args:
        train_state: Mutable state to serialize.
        filename: Destination ``train_state.pt`` path.
    """
    maybe_msc.torch.save(train_state.state_dict(), filename)


def load_train_state(filename: str | PathLike[str]) -> TrainState:
    """Load a Bridge-compatible sidecar on rank zero and broadcast it.

    Args:
        filename: Source ``train_state.pt`` path.

    Returns:
        The restored mutable training state.

    Raises:
        RuntimeError: If the sidecar cannot be loaded.
    """
    distributed = torch.distributed.is_initialized()
    state_obj: list[dict[str, Any] | None] = [None]

    if not distributed or torch.distributed.get_rank() == 0:
        try:
            state_obj[0] = {
                "state_dict": maybe_msc.torch.load(filename, map_location="cpu", weights_only=True)
            }
        except Exception as error:
            state_obj[0] = {"error": f"Unable to load train state file {filename}: {error}"}

    if distributed:
        torch.distributed.broadcast_object_list(state_obj, src=0)

    payload = state_obj[0]
    if payload is None or "error" in payload:
        message = (
            "Train-state broadcast returned no payload" if payload is None else payload["error"]
        )
        raise RuntimeError(message)

    train_state = TrainState()
    train_state.load_state_dict(payload["state_dict"])
    return train_state
