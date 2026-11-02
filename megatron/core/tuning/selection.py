# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Choosing one Triton config without measuring anything.

Every rule here is a pure function of the candidate list and the call signature,
so each rank computes the same answer. That property, not which config wins, is
what makes a run reproducible: the cheapest config is as deterministic as the
fastest one, it is just slower.
"""

import functools
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_untuned_kernels_warned: set = set()
_mamba_env_warned = False


def reset_warnings() -> None:
    """Forget which once-per-policy notices were issued, so a new policy reports again."""
    _untuned_kernels_warned.clear()


@functools.lru_cache(maxsize=None)
def _arch_for_device(index: int) -> str:
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    return f"sm{major}{minor}"


def arch_tag() -> str:
    """``sm<major><minor>`` for the current device, or ``unknown`` off-GPU."""
    import torch

    try:
        return _arch_for_device(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        # No CUDA device, or the driver is unavailable: the caller only needs a
        # stable label, and "unknown" simply never matches a recorded table.
        return "unknown"


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
    return json.dumps(config_data(config), sort_keys=True)


def config_data(config) -> dict:
    """Serializable launch options; hooks remain on the live Config object."""
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
        "num_ctas": getattr(config, "num_ctas", 1),
        "maxnreg": getattr(config, "maxnreg", None),
        "ir_override": getattr(config, "ir_override", None),
    }


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


def filter_configs_by_block_sizes(configs, block_sizes=()):
    """Match explicit kernel block-size arguments against live candidates."""
    if not block_sizes:
        return None
    matching = configs
    for key, target in sorted(dict(block_sizes).items()):
        matching = [c for c in matching if c.kwargs.get(key) == target]
    return matching[:1] if matching else None


def cheapest(configs):
    """The deterministic fallback: cheapest config by static estimate."""
    return min(configs, key=estimate_config_cost)


def deterministic_choice(
    autotuner, candidates, args, kwargs, *, table=None, on_miss="min_cost", block_sizes=()
):
    """Pick one config, preferring a tuned entry, never measuring.

    Order: the tuned table entry for this kernel and shape, then an explicit
    block-size override, then the cheapest config.
    """
    name = kernel_name(autotuner)
    qualified = f"{kernel_module(autotuner)}.{name}"
    if table is not None:
        key = tuning_key(autotuner, args, kwargs)
        # Recordings name kernels by module; older tables use the bare function name.
        tuned = table.lookup(qualified, key, candidates)
        if tuned is None:
            tuned = table.lookup(name, key, candidates)
        if tuned is not None:
            return tuned
    filtered = filter_configs_by_block_sizes(candidates, block_sizes)
    if filtered:
        return filtered[0]
    if on_miss == "error":
        raise RuntimeError(
            f"No tuned config for triton kernel {qualified!r} on {arch_tag()} and "
            "on_miss='error'. Record a table, or allow the "
            "deterministic min-cost fallback."
        )
    if qualified not in _untuned_kernels_warned:
        _untuned_kernels_warned.add(qualified)
        logger.warning(
            "No pre-tuned config for triton kernel %r on %s; using the cheapest config, "
            "which is deterministic but may be slower. Record a table with "
            "AutotunePolicy(mode='record', record_path=...) to recover the throughput.",
            qualified,
            arch_tag(),
        )
    return cheapest(candidates)


def chaos_choice(autotuner, candidates, args, kwargs):
    """Pick a different config per rank on purpose, reproducibly.

    A positive control: every other check is a negative one, and "the runs
    matched" cannot distinguish a working divergence detector from a blind one.
    """
    from megatron.core._rank_utils import safe_get_rank

    seed = f"{safe_get_rank()}|{kernel_name(autotuner)}|{tuning_key(autotuner, args, kwargs)}"
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
    from megatron.core.tuning.interception import active_policy
    from megatron.core.tuning.policy import use_deterministic_mode

    if not configs or not use_deterministic_mode():
        return configs
    policy = active_policy()
    filtered = filter_configs_by_block_sizes(configs, policy.block_sizes if policy else ())
    if filtered:
        return filtered
    return [cheapest(configs)]


def warn_if_mamba_env_ignored(policy) -> None:
    """Point out, once, that ``MAMBA_DETERMINISTIC`` does not pin Megatron's kernels.

    The variable still controls the external ``mamba_ssm`` package, but Megatron's
    own Triton kernels follow ``deterministic_mode`` and PyTorch's deterministic
    flag, so setting only the variable leaves them on timed autotuning.
    """
    global _mamba_env_warned

    if _mamba_env_warned or policy.mode != "auto":
        return
    if not os.environ.get("MAMBA_DETERMINISTIC", "").startswith("1"):
        return
    _mamba_env_warned = True
    logger.warning(
        "MAMBA_DETERMINISTIC=1 only affects the external mamba_ssm package. Megatron's "
        "Triton kernels, including megatron.core.ssm.ops, follow deterministic_mode "
        "(--deterministic-mode) or torch.use_deterministic_algorithms(True); set one of "
        "those, or AutotunePolicy(mode='pinned'), to pin their configurations."
    )
