# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer and scheduler construction without training-loop global configuration."""

from collections.abc import Callable
from functools import partial

from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.training.config.optimizer_config import OptimizerConfigOverrides
from megatron.training.config.training_config import SchedulerConfig


def build_optimizer_param_scheduler(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    optimizer: MegatronOptimizer,
) -> OptimizerParamScheduler:
    """Build a scheduler from finalized configuration without reading global arguments.

    The caller owns validation and finalization, including converting iteration horizons
    into the scheduler's sample clock. No configuration is mutated or finalized again.
    """
    return OptimizerParamScheduler(
        optimizer,
        init_lr=scheduler_config.lr_warmup_init,
        max_lr=optimizer_config.lr,
        min_lr=optimizer_config.min_lr,
        lr_warmup_steps=scheduler_config.lr_warmup_steps,
        lr_decay_steps=scheduler_config.lr_decay_steps,
        lr_decay_style=scheduler_config.lr_decay_style,
        start_wd=scheduler_config.start_weight_decay,
        end_wd=scheduler_config.end_weight_decay,
        wd_incr_steps=scheduler_config.wd_incr_steps,
        wd_incr_style=scheduler_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=scheduler_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=scheduler_config.override_opt_param_scheduler,
        wsd_decay_steps=scheduler_config.wsd_decay_steps,
        lr_wsd_decay_style=scheduler_config.lr_wsd_decay_style,
        step_lr_rescale_schedule=scheduler_config.step_lr_rescale_schedule,
        step_lr_rescale_warmup_samples=scheduler_config.step_lr_rescale_warmup_samples,
    )


def setup_optimizer(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    model: MegatronModule | list[MegatronModule],
    *,
    config_overrides: OptimizerConfigOverrides | None = None,
    use_gloo_process_groups: bool = False,
    pg_collection: ProcessGroupCollection | None = None,
    optimizer_factory: Callable[..., MegatronOptimizer] | None = None,
) -> tuple[MegatronOptimizer, OptimizerParamScheduler]:
    """Construct an optimizer and scheduler from explicit, finalized inputs.

    Args:
        optimizer_config: Finalized optimizer configuration.
        scheduler_config: Finalized scheduler configuration, including sample horizons.
        model: Built model or model chunks. The standard factory receives a list.
        config_overrides: Resolved parameter-group policy, with the same None/empty
            semantics as get_megatron_optimizer. Recipe mappings and callables can be
            resolved with build_optimizer_config_overrides before this call. This
            boundary preserves legacy group layouts instead of normalizing them again.
        use_gloo_process_groups: Whether optimizer construction may use Gloo groups.
        pg_collection: The model's process groups, forwarded unchanged. None retains
            the standard factory's legacy MPU fallback; explicit groups avoid it.
        optimizer_factory: Optional callable with its configuration already bound,
            for example functools.partial(get_megatron_optimizer, config=...).
            Receives model_chunks, config_overrides, use_gloo_process_groups and
            pg_collection as keyword arguments. Custom factories receive the original
            model argument and must return a Megatron-compatible optimizer supporting
            param_groups, training updates and checkpoint state.

    Returns:
        The optimizer and its scheduler. Configuration finalization and recipe policy
        resolution remain caller-owned; this function reads no global training config.
    """
    if optimizer_factory is None:
        optimizer_factory = partial(get_megatron_optimizer, config=optimizer_config)
        model = model if isinstance(model, list) else [model]
    optimizer = optimizer_factory(
        model_chunks=model,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
        pg_collection=pg_collection,
    )
    return optimizer, build_optimizer_param_scheduler(optimizer_config, scheduler_config, optimizer)
