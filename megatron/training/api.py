# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Python entrypoint for config-driven GPT-family training."""

from collections.abc import Callable
from functools import partial

from megatron.core.enums import ModelType
from megatron.training.callbacks import Callback, CallbackManager
from megatron.training.config import PretrainConfigContainer
from megatron.training.datasets.providers import build_gpt_datasets
from megatron.training.global_vars import set_run_config
from megatron.training.models.gpt import GPTModelConfig
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.training import pretrain as pretrain_loop


def pretrain(
    config: PretrainConfigContainer,
    forward_step_func: Callable,
    callbacks: list[Callback] | CallbackManager | None = None,
    *,
    dataset_provider: Callable | None = None,
) -> None:
    """Run MLM's training loop directly from native configs, without parsing CLI args.

    Supports builder-backed GPT/Hybrid models (including their VLM subclasses)
    with datasets described by ``config.data`` or a dataset callback. Custom
    model providers use ``megatron.training.training.pretrain`` directly.
    Config-only checkpoint save/resume currently supports ``torch_dist``;
    legacy CLI MoE upcycling is not supported by this entrypoint.

    Args:
        config: Native MLM configuration, finalized by framework startup.
        forward_step_func: MLM forward step taking ``(data_iterator, model)`` and
            optionally ``return_schedule_plan``. Returns output and a loss callback.
        callbacks: Native MLM training callbacks.
        dataset_provider: Optional MLM train/validation/test dataset callback. Use a
            loader-only ``config.data`` for application-specific datasets.
    """
    if not isinstance(config, PretrainConfigContainer):
        raise TypeError("config must be an MLM PretrainConfigContainer")
    if not isinstance(config.model, (GPTModelConfig, HybridModelConfig)):
        raise TypeError("This entrypoint requires a GPTModelConfig or HybridModelConfig")
    if config.data.dataset is None and dataset_provider is None:
        raise ValueError("Config-driven pretraining requires config.data.dataset")
    if config.checkpoint.ckpt_format != "torch_dist":
        raise ValueError("Config-only pretraining currently supports torch_dist checkpoints")
    config.validate()
    set_run_config(config)
    provider = dataset_provider
    if provider is None:
        provider = partial(build_gpt_datasets, data_config=config.data)
        provider.is_distributed = True
    pretrain_loop(
        config, provider, ModelType.encoder_or_decoder, forward_step_func, callbacks=callbacks
    )
