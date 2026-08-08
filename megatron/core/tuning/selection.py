# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Choosing one Triton config without measuring anything.

Every rule here is a pure function of the candidate list and the call signature,
so each rank computes the same answer. That property, not which config wins, is
what makes a run reproducible: the cheapest config is as deterministic as the
fastest one, it is just slower.
"""

import hashlib
import os
import warnings

_untuned_kernels_warned: set = set()


def arch_tag() -> str:
    """``sm<major><minor>`` for the current device, or ``unknown`` off-GPU."""
    import torch

    try:
        major, minor = torch.cuda.get_device_capability()
    except (AssertionError, RuntimeError):
        # No CUDA device, or the driver is unavailable: the caller only needs a
        # stable label, and "unknown" simply never matches a recorded table.
        return "unknown"
    return f"sm{major}{minor}"


def kernel_name(autotuner) -> str:
    """Underlying kernel function name, unwrapping JIT and decorator layers."""
    fn = getattr(autotuner, "base_fn", None) or getattr(autotuner, "fn", None)
    seen = 0
    while fn is not None and not hasattr(fn, "__name__") and hasattr(fn, "fn") and seen < 8:
        fn = fn.fn
        seen += 1
    return getattr(fn, "__name__", "") or ""


def kernel_module(autotuner) -> str:
    """Module the kernel is defined in, used to scope which kernels are pinned."""
    fn = getattr(autotuner, "base_fn", None) or getattr(autotuner, "fn", None)
    seen = 0
    while fn is not None and not hasattr(fn, "__module__") and hasattr(fn, "fn") and seen < 8:
        fn = fn.fn
        seen += 1
    return getattr(fn, "__module__", "") or ""


def tuning_key(autotuner, args, kwargs) -> str:
    """Shape/dtype signature for one autotuner invocation.

    Built from the autotuner's own ``keys`` plus the dtypes of its tensor
    arguments, so it is at least as fine as what the tiling depends on. It does
    not have to match Triton's key byte for byte: a drift produces a table miss,
    which falls back to a deterministic choice, never to a timed one.
    """
    named = dict(zip(getattr(autotuner, "arg_names", ()) or (), args))
    named.update(kwargs)
    parts = [
        f"{name}={named[name]}" for name in getattr(autotuner, "keys", ()) or () if name in named
    ]
    parts.extend(str(value.dtype) for value in named.values() if hasattr(value, "dtype"))
    return "|".join(parts)


def config_signature(config) -> str:
    """Stable text form of one Triton config, for logs and cross-rank compare."""
    if config is None:
        return "none"
    kwargs = ",".join(f"{k}={v}" for k, v in sorted(getattr(config, "kwargs", {}).items()))
    return (
        f"{kwargs};warps={getattr(config, 'num_warps', None)}"
        f";stages={getattr(config, 'num_stages', None)}"
    )


def estimate_config_cost(cfg):
    """Estimate shared-memory cost of a config. Lower is cheaper.

    Returns ``(block_cost, num_warps)`` so ties in block cost break
    deterministically on warp count.
    """
    block_product = 1
    for key, val in cfg.kwargs.items():
        if key.startswith('BLOCK') and isinstance(val, int):
            block_product *= val
    stages = getattr(cfg, 'num_stages', 1) or 1
    warps = getattr(cfg, 'num_warps', 1) or 1
    return (block_product * stages, warps)


def filter_configs_by_block_sizes(configs):
    """Filter configs by ``TRITON_AUTOTUNE_BLOCK_*`` environment overrides.

    Each variable names a kernel kwarg: ``TRITON_AUTOTUNE_BLOCK_SIZE_M`` selects
    configs whose ``BLOCK_SIZE_M`` matches. Returns ``None`` when no override
    applies or nothing matches, so the caller falls through to its normal rule.
    """
    prefix = "TRITON_AUTOTUNE_"
    env_filters = {}
    for env_key, env_val in os.environ.items():
        if env_key.startswith(prefix + "BLOCK") and env_val:
            env_filters[env_key[len(prefix) :]] = int(env_val)
    if not env_filters:
        return None
    matching = configs
    for key, target in sorted(env_filters.items()):
        matching = [c for c in matching if c.kwargs.get(key) == target]
    return matching[:1] if matching else None


def cheapest(configs):
    """The deterministic fallback: cheapest config by static estimate."""
    return min(configs, key=estimate_config_cost)


def deterministic_choice(autotuner, candidates, args, kwargs, *, table=None, on_miss="min_cost"):
    """Pick one config, preferring a tuned entry, never measuring.

    Order: the tuned table entry for this kernel and shape, then an explicit
    ``TRITON_AUTOTUNE_BLOCK_*`` override, then the cheapest config.
    """
    name = kernel_name(autotuner)
    if table is not None:
        tuned = table.lookup(name, tuning_key(autotuner, args, kwargs), candidates)
        if tuned is not None:
            return tuned
    if on_miss == "error":
        raise RuntimeError(
            f"No tuned config for triton kernel {name!r} on {arch_tag()} and "
            "MCORE_AUTOTUNE_ON_MISS=error. Record a table, or allow the "
            "deterministic min-cost fallback."
        )
    if name not in _untuned_kernels_warned:
        _untuned_kernels_warned.add(name)
        warnings.warn(
            f"No pre-tuned config for triton kernel {name!r} on {arch_tag()}; using the "
            "cheapest config, which is deterministic but may be slower. Record a table "
            "with MCORE_AUTOTUNE_RECORD to recover the throughput."
        )
    filtered = filter_configs_by_block_sizes(candidates)
    return (filtered or [cheapest(candidates)])[0]


def chaos_choice(autotuner, candidates, args, kwargs):
    """Pick a different config per rank on purpose, reproducibly.

    A positive control: every other check is a negative one, and "the runs
    matched" cannot distinguish a working divergence detector from a blind one.
    """
    seed = (
        f"{os.environ.get('RANK', '0')}|{kernel_name(autotuner)}"
        f"|{tuning_key(autotuner, args, kwargs)}"
    )
    index = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % len(candidates)
    return candidates[index]


def autotune_configs(configs):
    """Reduce an in-tree kernel's config list under deterministic mode.

    Used by Megatron's own Triton kernels at decoration time, where there is no
    autotuner object to intercept. Cached autotuning
    (``TRITON_CACHE_AUTOTUNING=1``) is not sufficient here: the first benchmark
    is still timed, so the cached winner varies per process, per GPU and per
    cache directory.
    """
    from megatron.core.tuning.policy import use_deterministic_mode

    if not configs or not use_deterministic_mode():
        return configs
    filtered = filter_configs_by_block_sizes(configs)
    if filtered:
        return filtered
    return [cheapest(configs)]
