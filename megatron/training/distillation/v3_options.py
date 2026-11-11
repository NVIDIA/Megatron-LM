# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bind V3's CLI options to the run-owned configuration and mutable TrainState.

V3 flags remain in the argument namespace. Shared training/model options must read
their live configuration, and counters must never read the stale CLI namespace.
Before runtime initialization (argument validation and isolated codec callers), the
original namespace is used directly.
"""

from typing import Any


class _RunOptions:
    def __init__(self, args: Any, config: Any, state: Any):
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_state", state)

    def _owner(self, name: str) -> Any:
        if name in (
            "iteration",
            "consumed_train_samples",
            "consumed_valid_samples",
            "skipped_train_samples",
        ):
            if self._state is None:
                raise RuntimeError(f"V3 runtime option {name} requires TrainState")
            return self._state
        if name == "seq_length":
            return self._config.model
        if name == "padded_vocab_size":
            return self._config.tokenizer
        for section in ("train", "validation", "checkpoint", "rng"):
            owner = getattr(self._config, section)
            if hasattr(owner, name):
                return owner
        if hasattr(self._config.model.transformer, name):
            return self._config.model.transformer
        if hasattr(self._config.data.dataloader, name):
            return self._config.data.dataloader
        if self._config.data.dataset is not None and hasattr(self._config.data.dataset, name):
            return self._config.data.dataset
        return self._args

    def __getattr__(self, name: str) -> Any:
        if name == "transformer_config":
            return self._config.model.transformer
        return getattr(self._owner(name), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._owner(name), name, value)


def runtime_options(args: Any) -> Any:
    """Resolve shared V3 settings through the initialized run rather than args copies."""
    if isinstance(args, _RunOptions):
        return args
    try:
        from megatron.training.global_vars import (
            get_run_config,
            get_train_state,
            is_run_config_initialized,
        )
    except ImportError:
        # Standalone CPU consumers provide only an argument namespace.
        return args
    if not is_run_config_initialized():
        return args
    return _RunOptions(args, get_run_config(), get_train_state())


def get_args() -> Any:
    """Return V3 flags together with live run configuration and counters."""
    from megatron.training import get_args as get_namespace

    return runtime_options(get_namespace())
