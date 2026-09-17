# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import torch
from torch.distributed.checkpoint.stateful import Stateful


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
    num_floating_point_operations_so_far: float = 0.0
    do_train: bool = False
    do_valid: bool = False
    do_test: bool = False

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serializes the training state into a dictionary of tensors.

        Conforms to the Stateful interface for distributed checkpointing.

        Returns:
            A dictionary where keys are state variable names and values are
            their corresponding tensor representations.
        """
        return {
            "step": torch.tensor(self.iteration, dtype=torch.int64),
            "consumed_train_samples": torch.tensor(self.consumed_train_samples, dtype=torch.int64),
            "skipped_train_samples": torch.tensor(self.skipped_train_samples, dtype=torch.int64),
            "consumed_valid_samples": torch.tensor(self.consumed_valid_samples, dtype=torch.int64),
            "floating_point_operations_so_far": torch.tensor(
                self.num_floating_point_operations_so_far, dtype=torch.float64
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
        self.num_floating_point_operations_so_far = state_dict["floating_point_operations_so_far"].item()
        self.do_train = state_dict["do_train"].item()
        self.do_valid = state_dict["do_valid"].item()
        self.do_test = state_dict["do_test"].item()
