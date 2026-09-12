# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings

import torch
from packaging import version

try:
    import triton

    TRITON_VERSION = version.parse(triton.__version__)
except ImportError:
    TRITON_VERSION = version.parse("0.0.0")

TRITON_HAS_CACHE_RESULTS = TRITON_VERSION >= version.parse("3.4.0")
_autotune_warning_issued = False

_deterministic_override = None


def use_deterministic_mode():
    """Use torch deterministic mode."""
    if _deterministic_override is not None:
        return _deterministic_override
    env = os.environ.get('MAMBA_DETERMINISTIC')
    if env:
        return env[0] == '1'
    return torch.are_deterministic_algorithms_enabled()


def set_deterministic_mode(value):
    """Set torch deterministic mode."""
    global _deterministic_override
    _deterministic_override = value


def _estimate_config_cost(cfg):
    """Estimate shared memory cost of a config. Lower is cheaper.

    Returns a tuple (block_cost, num_warps) so that ties in block cost
    are broken deterministically by warp count (fewer warps = cheaper).
    """
    block_product = 1
    for key, val in cfg.kwargs.items():
        if key.startswith('BLOCK') and isinstance(val, int):
            block_product *= val
    stages = getattr(cfg, 'num_stages', 1) or 1
    warps = getattr(cfg, 'num_warps', 1) or 1
    return (block_product * stages, warps)


def _filter_configs_by_block_sizes(configs):
    """Filter configs by TRITON_AUTOTUNE_BLOCK_* env vars.

    Scans environment for any variable matching TRITON_AUTOTUNE_BLOCK_*
    (e.g. TRITON_AUTOTUNE_BLOCK_SIZE_M, TRITON_AUTOTUNE_BLOCK_SIZE_H,
    TRITON_AUTOTUNE_BLOCK_T, TRITON_AUTOTUNE_BLOCK_C, TRITON_AUTOTUNE_BLOCK_SIZE)
    and maps them to the corresponding kernel kwarg (BLOCK_SIZE_M, BLOCK_SIZE_H,
    BLOCK_T, BLOCK_C, BLOCK_SIZE).
    """
    prefix = "TRITON_AUTOTUNE_"
    env_filters = {}
    for env_key, env_val in os.environ.items():
        if env_key.startswith(prefix + "BLOCK") and env_val:
            kwarg_name = env_key[len(prefix) :]
            env_filters[kwarg_name] = int(env_val)
    if not env_filters:
        return None
    matching = configs
    for key, target in sorted(env_filters.items()):
        matching = [c for c in matching if c.kwargs.get(key) == target]
    return matching[:1] if matching else None


def autotune_configs(configs):
    """Select autotune configs for deterministic mode.

    Uses cached autotuning (TRITON_CACHE_AUTOTUNING=1) if Triton >= 3.4.0,
    otherwise auto-selects the cheapest config by block size * stages.
    """
    if not configs or not use_deterministic_mode():
        return configs
    if TRITON_HAS_CACHE_RESULTS and os.environ.get("TRITON_CACHE_AUTOTUNING") == "1":
        return configs
    global _autotune_warning_issued
    if not _autotune_warning_issued:
        _autotune_warning_issued = True
        msg = (
            "Deterministic mode: set TRITON_CACHE_AUTOTUNING=1 for cached autotuning."
            if TRITON_HAS_CACHE_RESULTS
            else "Deterministic mode: upgrade to Triton >= 3.4.0 for cached autotuning."
        )
        warnings.warn(msg)
    filtered = _filter_configs_by_block_sizes(configs)
    if filtered:
        return filtered
    return [min(configs, key=_estimate_config_cost)]


def alloc_tile_workspace(base_shape, tile_dim, dtype, device, deterministic, *, zero_init=True):
    """Allocate buffer for deterministic per-program reductions."""
    if base_shape is None:
        return None, 0
    if deterministic:
        factory = torch.zeros if zero_init else torch.empty
        tensor = factory(*base_shape, tile_dim, device=device, dtype=dtype)
        return tensor, tensor.stride(-1)
    return torch.empty(*base_shape, device=device, dtype=dtype), 0


def finalize_tile_workspace(tensor, deterministic):
    """Finalize tile workspace."""
    if tensor is None:
        return None
    if deterministic:
        tensor = tensor.sum(dim=-1)
    return tensor
