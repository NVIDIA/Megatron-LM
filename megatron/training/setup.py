# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
from typing import Any

from megatron.core._rank_utils import safe_get_rank
from megatron.training.config import PretrainConfigContainer, CheckpointConfig
from megatron.training.utils import print_rank_0

def maybe_save_config(cfg: PretrainConfigContainer) -> None:
    """Save configuration to disk."""

    if safe_get_rank() != 0:
        return

    if cfg.logger.save_config_filepath is not None:
        try:
            cfg.to_yaml(cfg.logger.save_config_filepath)
        except Exception as e:
            print_rank_0(f"Error saving config to file {cfg.logger.save_config_filepath}: {e}")


def init_checkpointing_context(checkpoint_config: CheckpointConfig) -> dict[str, Any]:
    """Initialize the checkpointing context, primarily for local checkpointing support.

    If `non_persistent_ckpt_type` is set to "local", this function sets up
    the `LocalCheckpointManager` and replication strategy based on the provided
    `checkpoint_config`.

    Args:
        checkpoint_config: The checkpoint configuration object.

    Returns:
        A dictionary containing the checkpointing context. This will include
        a `local_checkpoint_manager` if local checkpointing is enabled,
        otherwise it will be an empty dictionary.

    Raises:
        RuntimeError: If local checkpointing is configured but the
                      `nvidia_resiliency_ext` module is not found.
    """
    if checkpoint_config.non_persistent_ckpt_type != "local":
        return {}

    try:
        from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
            LocalCheckpointManager,
        )
        from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import (
            GroupWrapper,
            parse_group_sequence,
        )
        from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
            CliqueReplicationStrategy,
        )
    except ModuleNotFoundError:
        raise RuntimeError(
            "The 'nvidia_resiliency_ext' module is required for local "
            "checkpointing but was not found. Please ensure it is installed."
        )

    if checkpoint_config.replication:
        repl_strategy = CliqueReplicationStrategy.from_replication_params(
            checkpoint_config.replication_jump,
            checkpoint_config.replication_factor,
        )
    else:
        repl_strategy = None

    checkpointing_context = {
        "local_checkpoint_manager": LocalCheckpointManager(
            checkpoint_config.non_persistent_local_ckpt_dir,
            repl_strategy=repl_strategy,
        )
    }
    return checkpointing_context


def validate_and_set_vocab_size(model_vocab_size: int | None, tokenizer_vocab_size: int) -> tuple[int, bool]:
    """Validate and determine the correct vocab size for the model.

    Args:
        model_vocab_size: Vocab size set in model config (can be None)
        tokenizer_vocab_size: Unpadded tokenizer vocab size

    Returns:
        tuple[int, bool]: The validated unpadded vocab size and padding flag
            - vocab_size: The validated unpadded vocab size to use for the model
            - should_pad_vocab: True if vocab should be padded, False otherwise

    Raises:
        ValueError: If model vocab size is invalid
    """
    if model_vocab_size is None:
        # If model vocab size is not set, use the tokenizer's vocab size
        # Enable padding since this came from tokenizer
        return tokenizer_vocab_size, True
    elif model_vocab_size < tokenizer_vocab_size:
        # Vocab size smaller than tokenizer
        raise ValueError(
            f"Model vocab_size ({model_vocab_size}) cannot be smaller than tokenizer's vocab_size "
            f"({tokenizer_vocab_size})."
        )
    else:
        # Model vocab size is explicitly set and is >= tokenizer vocab size
        # Disable padding since this was explicitly set
        if model_vocab_size > tokenizer_vocab_size:
            print_rank_0(
                f"Using preset vocab_size: {model_vocab_size} over the tokenizer vocab_size: {tokenizer_vocab_size}, dummy tokens:"
                f" {model_vocab_size - tokenizer_vocab_size}."
            )
        return model_vocab_size, False
