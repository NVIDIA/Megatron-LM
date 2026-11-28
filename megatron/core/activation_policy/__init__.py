# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Activation memory policies: recompute and offloading of activations.

A module declares its :mod:`.strategies` in ``MegatronModule.activation_strategies``, built from
its config (for example ``TransformerConfig.recompute_selected``); they are installed once the
module is constructed. A strategy acts on a *point*: a child module or a method of its owner,
hooked by :mod:`.runtime`, which runs the installed strategies around those calls.

The mechanisms behind the strategies live in :mod:`.recompute` and :mod:`.offload`.

Attributes are imported on first use, so importing this package (for example from
``megatron.core.tensor_parallel``, which re-exports the checkpointing primitives) does not import
model code.
"""

import importlib

_EXPORTS = {
    "CheckpointStrategy": "strategies",
    "CheckpointWithoutOutputStrategy": "strategies",
    "Event": "runtime",
    "Strategy": "strategies",
    "has_strategy": "runtime",
    "install_activation_strategies": "runtime",
    "installed_strategies": "runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
