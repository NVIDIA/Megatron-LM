# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

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

    Config selection must be value-deterministic (a pure function of the
    config list), never timing-derived. Cached autotuning
    (``TRITON_CACHE_AUTOTUNING=1``) is NOT sufficient: the first benchmark is
    still timing-based, so the cached winner can differ per process, per GPU,
    and per (ephemeral, container-local) Triton cache directory. Two
    otherwise-identical runs then execute different tile shapes, which changes
    the floating-point reduction order. Deterministic mode therefore always
    pins to the env-selected or cheapest config.
    """
    if not configs or not use_deterministic_mode():
        return configs
    filtered = _filter_configs_by_block_sizes(configs)
    if filtered:
        return filtered
    return [min(configs, key=_estimate_config_cost)]


_external_autotune_pinned = False


def pin_external_mamba_autotuners():
    """Pin timing-based Triton autotuners in the external ``mamba_ssm`` package.

    The Mamba memory-efficient training path (``use_mamba_mem_eff_path``, the
    default) calls ``mamba_split_conv1d_scan_combined`` from the external
    ``mamba_ssm`` package. Those kernels carry their own ``@triton.autotune``
    decorators that benchmark candidate configs at first call, so the winning
    config — and therefore the floating-point reduction order — can differ
    across processes and GPUs from clock jitter alone. Under deterministic
    mode, prune each such autotuner to a single value-deterministically chosen
    config (same rule as ``autotune_configs``). Scoped to ``mamba_ssm``
    functions so other Triton autotuners keep their tuned performance.

    Idempotent; a no-op when triton is unavailable.
    """
    global _external_autotune_pinned
    if _external_autotune_pinned:
        return
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return

    original_run = Autotuner.run

    def _kernel_module(autotuner):
        fn = getattr(autotuner, "base_fn", None) or getattr(autotuner, "fn", None)
        # Unwrap JITFunction and nested decorators to the underlying python fn.
        seen = 0
        while fn is not None and not hasattr(fn, "__module__") and hasattr(fn, "fn") and seen < 8:
            fn = fn.fn
            seen += 1
        return getattr(fn, "__module__", "") or ""

    def deterministic_run(self, *args, **kwargs):
        if (
            len(getattr(self, "configs", ())) > 1
            and use_deterministic_mode()
            and _kernel_module(self).startswith("mamba_ssm")
        ):
            filtered = _filter_configs_by_block_sizes(self.configs)
            self.configs = filtered or [min(self.configs, key=_estimate_config_cost)]
        return original_run(self, *args, **kwargs)

    Autotuner.run = deterministic_run
    _external_autotune_pinned = True


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
