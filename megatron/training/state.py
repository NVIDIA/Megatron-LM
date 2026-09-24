# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from megatron.core.msc_utils import maybe_msc

TRAIN_STATE_FILENAME = "train_state.pt"


@dataclass
class TrainState(Stateful):
    """Dataclass to hold the mutable and checkpointable state of the training process.

    Inherits from Stateful for distributed checkpointing compatibility.
    Tracks iteration count, consumed samples, flags for train/valid/test phases,
    and floating-point operations.
    """

    iteration: int = 0
    consumed_train_samples: int = 0
    skipped_train_samples: int = 0
    consumed_valid_samples: int = 0
    num_floating_point_operations_so_far: int = 0
    do_train: bool = False
    do_valid: bool = False
    do_test: bool = False

    def update_from_args(
        self, args: Any, iteration: int, num_floating_point_operations_so_far: int
    ) -> None:
        """Update the active state from legacy runtime fields during migration."""
        self.iteration = iteration
        self.consumed_train_samples = getattr(args, "consumed_train_samples", 0)
        self.skipped_train_samples = getattr(args, "skipped_train_samples", 0)
        self.consumed_valid_samples = getattr(args, "consumed_valid_samples", 0)
        self.num_floating_point_operations_so_far = num_floating_point_operations_so_far
        self.do_train = getattr(args, "do_train", False)
        self.do_valid = getattr(args, "do_valid", False)
        self.do_test = getattr(args, "do_test", False)

    def apply_to_args(self, args: Any) -> None:
        """Mirror loaded state to legacy runtime fields during migration."""
        args.consumed_train_samples = self.consumed_train_samples
        args.skipped_train_samples = self.skipped_train_samples
        args.consumed_valid_samples = self.consumed_valid_samples
        args.do_train = self.do_train
        args.do_valid = self.do_valid
        args.do_test = self.do_test

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serializes the training state into a dictionary of tensors.

        Conforms to the Stateful interface for distributed checkpointing.

        Returns:
            A dictionary where keys are state variable names and values are
            their corresponding tensor representations.
        """
        return {
            # TrainState comes from Megatron-Bridge, however that repo used 'step' instead of 'iteration'
            # for both the state dict and the dataclass attribute. 'iteration' is more consistent with
            # Megatron-LM, but using 'step' for the state dict will allow pre-unification Bridge checkpoints
            # to work without issue when loading in Megatron-LM after unification.
            # The same applies for 'floating_point_operations_so_far' (Megatron-Bridge) vs
            # 'num_floating_point_operations_so_far' (Megatron-LM).
            "step": torch.tensor(self.iteration, dtype=torch.int64),
            "consumed_train_samples": torch.tensor(self.consumed_train_samples, dtype=torch.int64),
            "skipped_train_samples": torch.tensor(self.skipped_train_samples, dtype=torch.int64),
            "consumed_valid_samples": torch.tensor(self.consumed_valid_samples, dtype=torch.int64),
            "floating_point_operations_so_far": torch.tensor(
                self.num_floating_point_operations_so_far, dtype=torch.int64
            ),
            "do_train": torch.tensor(self.do_train, dtype=torch.bool),
            "do_valid": torch.tensor(self.do_valid, dtype=torch.bool),
            "do_test": torch.tensor(self.do_test, dtype=torch.bool),
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the training state from a state dictionary.

        Args:
            state_dict: A dictionary containing the state variables as tensors.
        """
        self.iteration = state_dict["step"].item()
        self.consumed_train_samples = state_dict["consumed_train_samples"].item()
        self.skipped_train_samples = state_dict["skipped_train_samples"].item()
        self.consumed_valid_samples = state_dict["consumed_valid_samples"].item()
        self.num_floating_point_operations_so_far = state_dict[
            "floating_point_operations_so_far"
        ].item()
        self.do_train = state_dict["do_train"].item()
        self.do_valid = state_dict["do_valid"].item()
        self.do_test = state_dict["do_test"].item()


def save_train_state(train_state: TrainState, filename: str | PathLike[str]) -> None:
    """Write a Bridge-compatible train-state sidecar."""
    maybe_msc.torch.save(train_state.state_dict(), filename)


def load_train_state(filename: str | PathLike[str]) -> TrainState:
    """Load a train-state sidecar on rank zero and broadcast it to every rank."""
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
