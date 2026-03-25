# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backward-compatible shim — all code now lives in ``emerging_optimizers``."""

from typing import Any

# TODO: Remove this separate try/except once the next version of emerging_optimizers
# (which includes Lion) is released. Then Lion can be imported in the block above.
try:
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import

    HAVE_LION = True
except ImportError:
    HAVE_LION = False


def get_megatron_muon_optimizer(*args: Any, **kwargs: Any) -> Any:
    """Backward compatible muon optimizer getter.

    .. deprecated::
        Use :func:`megatron.core.optimizer.get_megatron_optimizer` instead.
    """
    from . import get_megatron_optimizer

    kwargs.pop('layer_wise_distributed_optimizer', None)
    return get_megatron_optimizer(*args, **kwargs)
