# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Extension interface for injecting hooks into the pretrain loop.

`pretrain()` accepts a `pretrain_context_factory` kwarg and calls its methods at well-defined
points in the training loop. Callers that need to customize behavior at these points
(e.g. the RL training loop) must provide a factory whose hooks override the defaults.

The context has a two-phase lifecycle:

1. Setup phase (constructed before model/optimizer/dataloaders exist):
   policy hooks like `should_build_optimizer()` and `should_build_standard_dataloaders()`.

2. Training phase (after `attach_training_state()` is called):
   train-loop hooks like `begin_iteration()`, `run_eval()`, etc. have access to
   the model, optimizer, config, and data iterators.
"""

from typing import Any, Callable, List, Optional

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches


class PretrainContext:
    """Default pretrain hooks; standard pretrain behavior."""

    def __init__(self, *, args, model_provider, model_type):
        self.args = args
        self.model_provider = model_provider
        self.model_type = model_type

        # The following are populated by attach_training_state.
        self.model = None
        self.optimizer = None
        self.config = None
        self.train_data_iterator = None
        self.valid_data_iterator = None
        self.checkpointing_context = None
        self.forward_step_func = None
        self.process_non_loss_data_func = None
        self.non_loss_data_func = None
        self._attached = False

    def should_build_optimizer(self) -> bool:
        """Whether `setup_model_and_optimizer` should build the optimizer."""
        return not self.args.skip_train

    def should_build_standard_dataloaders(self) -> bool:
        """Whether `pretrain` should build the standard data iterators."""
        return True

    def attach_training_state(
        self,
        *,
        model,
        optimizer,
        config,
        train_data_iterator,
        valid_data_iterator,
        checkpointing_context,
        forward_step_func,
        process_non_loss_data_func,
        non_loss_data_func,
    ) -> None:
        """Populate the heavy training state after model/optimizer/dataloaders are built."""
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.train_data_iterator = train_data_iterator
        self.valid_data_iterator = valid_data_iterator
        self.checkpointing_context = checkpointing_context
        self.forward_step_func = forward_step_func
        self.process_non_loss_data_func = process_non_loss_data_func
        self.non_loss_data_func = non_loss_data_func
        self._attached = True

    def should_run_train_loop(self) -> bool:
        """Whether ``pretrain`` should enter the train loop at all."""
        return not self.args.skip_train

    def before_train_loop(self) -> None:
        """Called once from ``train()`` before iteration begins.

        The default wraps ``train_data_iterator`` in the hybrid-CP data loader
        when that option is set.
        """
        assert self._attached, "PretrainContext.attach_training_state() must be called first"
        if self.args.hybrid_context_parallel:
            from megatron.core.datasets.data_schedule import HybridCPDataLoaderWrapper

            self.train_data_iterator = iter(
                HybridCPDataLoaderWrapper(self.train_data_iterator, self.config)
            )

    def begin_iteration(self, iteration: int) -> None:
        """Called at the top of each outer iteration, before any step work."""
        pass

    def iteration_sequence_count(self) -> int:
        """Number of sequences consumed in the current iteration."""
        return (
            mpu.get_data_parallel_world_size() * self.args.micro_batch_size * get_num_microbatches()
        )

    def iteration_bin_count(self) -> Optional[int]:
        """Packed bin count for the current iteration, or `None` when unused."""
        return None

    def on_microbatch_change(
        self, iteration: int, old_num_microbatches: int, new_num_microbatches: int
    ) -> bool:
        """Called when the microbatch count changes mid-loop.

        Return `True` to run the standard ramp-up handling.
        Return `False` to suppress it, like with RL sequence packing.
        """
        return True

    def run_eval(
        self,
        *,
        prefix: str,
        iteration: int,
        verbose: bool,
        write_to_tensorboard: bool,
    ) -> None:
        """Run the validation pass. Default: `evaluate_and_print_results`."""
        assert self._attached, "PretrainContext.attach_training_state() must be called first"
        from megatron.training.training import evaluate_and_print_results

        evaluate_and_print_results(
            prefix,
            self.forward_step_func,
            self.valid_data_iterator,
            self.model,
            iteration,
            self.process_non_loss_data_func,
            self.config,
            verbose=verbose,
            write_to_tensorboard=write_to_tensorboard,
            non_loss_data_func=self.non_loss_data_func,
        )

    def extra_log_timers(self) -> List[str]:
        """Extra timer names to include in `timers_to_log`."""
        return []

    def log_tensorboard_metrics(self, writer, wandb_writer, iteration: int) -> None:
        """Write context-specific per-iteration metrics to tensorboard/wandb."""
        pass

    def extra_log_string(self) -> str:
        """Extra text appended to the per-iteration training log line."""
        return ""

    def shutdown(self) -> None:
        """Called once when training terminates (clean shutdown)."""
        pass


PretrainContextFactory = Callable[..., PretrainContext]


def default_pretrain_context_factory(**kwargs: Any) -> PretrainContext:
    """Default factory: construct a plain PretrainContext class."""
    return PretrainContext(**kwargs)
