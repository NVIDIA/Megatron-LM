# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any

from megatron.rl import rl_utils
from megatron.rl.rl_profiling import (
    RL_LOGGABLE_TIMER_NAMES,
    log_iteration_profile,
    maybe_initialize_rl_profiler,
    shutdown_rl_profiler,
)
from megatron.rl.sequence_packing_utils import (
    bin_batch_size,
    get_sequence_packing_log_info,
    get_sequence_packing_tensorboard_metrics,
)
from megatron.training.callbacks import Callback, CallbackContext
from megatron.training.global_vars import (
    get_args,
    get_tensorboard_writer,
    get_timers,
    get_train_state,
    get_wandb_writer,
)
from megatron.training.utils import print_rank_0


class RLLoggingCallback(Callback):
    """RL timers, packing counters, GRPO collection progress, rollout throughput and the RL
    profiler's lifecycle."""

    def __init__(self, rl_config: Any, *, global_batch_size: int) -> None:
        self._rl_config = rl_config
        # Training iterations served by one rollout collection.
        self._iterations_per_collection = rl_config.grpo_iterations * (
            rl_config.grpo_samples_per_iteration // global_batch_size
        )

    def on_train_start(self, context: CallbackContext) -> None:
        profile_dir = maybe_initialize_rl_profiler(
            self._rl_config,
            get_args().save,
            log_to_wandb=get_wandb_writer() is not None,
            log_to_tensorboard=get_tensorboard_writer() is not None,
        )
        if profile_dir is not None:
            print_rank_0(f'[RLProfiler] Profiling enabled, output: {profile_dir}')

    def on_train_step_end(self, context: CallbackContext) -> None:
        if self._rl_config.rl_use_sequence_packing:
            # Bins consumed by this step; the sequences inside them are counted by the loop.
            rl_utils.get_rl_runtime_state().consumed_train_bins += bin_batch_size(
                get_args().micro_batch_size
            )

    def on_tensorboard_log(self, context: CallbackContext) -> None:
        writer = get_tensorboard_writer()
        wandb_writer = get_wandb_writer()
        iteration = get_train_state().iteration
        if self._rl_config.rl_use_sequence_packing:
            metrics = get_sequence_packing_tensorboard_metrics(
                rl_utils.get_rl_runtime_state().consumed_train_bins, get_args().micro_batch_size
            )
            for name, value in metrics.items():
                writer.add_scalar(name, value, iteration)
            if wandb_writer and metrics:
                wandb_writer.log(metrics, iteration)
        grpo_collection_iteration = iteration // self._iterations_per_collection
        writer.add_scalar('grpo_collection_iteration', grpo_collection_iteration, iteration)
        if wandb_writer:
            wandb_writer.log({'grpo_collection_iteration': grpo_collection_iteration}, iteration)

    def on_log(self, context: CallbackContext) -> None:
        wandb_writer = get_wandb_writer()
        iteration = get_train_state().iteration
        context.timers_to_log.extend(RL_LOGGABLE_TIMER_NAMES)
        if self._rl_config.rl_use_sequence_packing:
            context.log_fragments.append(
                get_sequence_packing_log_info(rl_utils.get_rl_runtime_state().consumed_train_bins)
            )
        context.log_fragments.append(
            rl_utils.log_rl_throughput_metrics(
                get_args(),
                context.batch_size,
                context.elapsed_time_per_iteration,
                iteration,
                wandb_writer,
            )
        )
        if self._rl_config.rl_profile:
            # Before the loop resets its timers (that happens after on_log).
            log_iteration_profile(
                iteration=iteration,
                timers=get_timers(),
                elapsed_time_ms=context.elapsed_time_per_iteration * 1000.0,
                throughput_tflops=context.throughput,
                global_batch_size=context.batch_size,
                wandb_writer=wandb_writer,
                tb_writer=get_tensorboard_writer(),
            )

    def on_train_end(self, context: CallbackContext) -> None:
        if self._rl_config.rl_profile:
            shutdown_rl_profiler()


class RLCheckpointCallback(Callback):
    """Compacts the durable rollout bank once a checkpoint is durable (no-op without a bank)."""

    def on_checkpoint_saved(self, context: CallbackContext) -> None:
        rl_utils.maybe_compact_rollout_bank(context.durable_iteration)
