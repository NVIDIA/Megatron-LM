# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backward-compatible shim — all code now lives in ``emerging_optimizers``."""

from typing import Any


def get_megatron_muon_optimizer(*args: Any, **kwargs: Any) -> Any:
    """Backward compatible muon optimizer getter.

    .. deprecated::
        Use :func:`megatron.core.optimizer.get_megatron_optimizer` instead.
    """
    from . import get_megatron_optimizer

    if kwargs.pop('layer_wise_distributed_optimizer', False):
        config = args[0] if args else kwargs.get('config')
        if config is not None:
            config.use_layer_wise_distributed_optimizer = True

    return get_megatron_optimizer(*args, **kwargs)
