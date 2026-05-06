# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
import os
import time
import types
from dataclasses import dataclass
from typing import Any

from megatron.core._rank_utils import safe_get_rank
import torch
from megatron.core.timers import Timers
from megatron.core.dist_checkpointing.strategies.torch import get_async_strategy
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.tensorboard.writer import SummaryWriter

from megatron.training.config import PretrainConfigContainer
from megatron.training.dist_signal_handler import DistributedSignalHandler


@dataclass
class TrainState(Stateful):
    """Dataclass to hold the state of the training process.

    Inherits from Stateful for distributed checkpointing compatibility.
    Tracks iteration count, consumed samples, flags for train/valid/test phases,
    and floating-point operations.
    """

    step: int = 0
    consumed_train_samples: int = 0
    skipped_train_samples: int = 0
    consumed_valid_samples: int = 0
    floating_point_operations_so_far: int = 0
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
        """Load the training state from a state dictionary.

        Args:
            state_dict: A dictionary containing the state variables as tensors.
        """
        self.step = state_dict["step"].item()
        self.consumed_train_samples = state_dict["consumed_train_samples"].item()
        self.skipped_train_samples = state_dict["skipped_train_samples"].item()
        self.consumed_valid_samples = state_dict["consumed_valid_samples"].item()
        self.floating_point_operations_so_far = state_dict["floating_point_operations_so_far"].item()
        self.do_train = state_dict["do_train"].item()
        self.do_valid = state_dict["do_valid"].item()
        self.do_test = state_dict["do_test"].item()


class GlobalState:
    """Manages the global state of the training process.

    Provides access to configuration, loggers, timers,
    training state, fault tolerance state, and signal handler
    through properties with lazy initialization.
    """

    def __init__(self) -> None:
        """Initializes the GlobalState object."""
        # Prevent reinitialization in subsequent instantiations.
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._cfg: PretrainConfigContainer | None = None
        self._timers: Timers | None = None
        self._train_state: TrainState | None = None
        self._signal_handler: DistributedSignalHandler | None = None
        self.start_time: float = time.time()

    @property
    def cfg(self) -> PretrainConfigContainer | None:
        """The main configuration container object."""
        return self._cfg

    @cfg.setter
    def cfg(self, value: PretrainConfigContainer | None) -> None:
        """Sets the configuration container and initializes the signal handler.

        Args:
            value: The ConfigContainer instance to set.
        """
        self._cfg = value

        # This lazily initializes the signal handler when the config is set
        # in order to read the exit signal from the config.
        # This assumes the global state is first initialized and that the
        # config is immediately set on the global state after initialization.
        if value is not None:
            self._set_signal_handler()

    @property
    def timers(self) -> Timers:
        """The Megatron Timers instance used for tracking execution times."""
        if self._timers is None:
            self._timers = Timers(self.cfg.logger.timing_log_level, self.cfg.logger.timing_log_option)
        return self._timers

    @property
    def train_state(self) -> TrainState:
        """The TrainState object holding training progress information."""
        if self._train_state is None:
            self._train_state = TrainState()
        return self._train_state

    @train_state.setter
    def train_state(self, value: TrainState) -> None:
        """Sets the training state object.

        Args:
            value: The TrainState instance to set.
        """
        self._train_state = value

    @property
    def signal_handler(self) -> DistributedSignalHandler:
        """The DistributedSignalHandler instance for graceful shutdown."""
        if self._signal_handler is None:
            self._set_signal_handler()
        return self._signal_handler

    def _set_signal_handler(self) -> None:
        """Initializes the distributed signal handler based on the configuration."""
        if self.cfg.train is not None:
            self._signal_handler = DistributedSignalHandler(self.cfg.train.exit_signal).__enter__()
