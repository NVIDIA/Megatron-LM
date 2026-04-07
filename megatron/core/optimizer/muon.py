# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backward-compatible shim — all code now lives in ``emerging_optimizers``."""

from typing import Any


def get_megatron_muon_optimizer(*args: Any, **kwargs: Any) -> Any:
    """Backward compatible muon optimizer getter.

    .. deprecated::
        Use :func:`megatron.core.optimizer.get_megatron_optimizer` instead.
    """
    from . import get_megatron_optimizer

    use_layer_wise = kwargs.pop('layer_wise_distributed_optimizer', False)

    if "config" in kwargs:
        config = kwargs["config"]
    else:
        config = args[0]

    if use_layer_wise:
        config.use_layer_wise_distributed_optimizer = True
    if use_layer_wise and not config.optimizer.startswith('dist_'):
        raise ValueError(
            "Layer-wise distributed optimizer is enabled by dist_ prefix in optimizer name."
        )

    return get_megatron_optimizer(*args, **kwargs)
