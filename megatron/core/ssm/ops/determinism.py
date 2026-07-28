# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import os
import warnings

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
_untuned_kernels_warned = set()
# Set to a path prefix to run a TUNING pass: autotuning happens normally (so the
# run is NOT bit-exact) and the winning configs are written out for review.
_TUNE_RECORD_PATH = os.environ.get("MCORE_DET_TUNE_RECORD")
_tune_records: dict = {}


def _arch_tag():
    """``sm<major><minor>`` for the current device, or ``unknown`` off-GPU."""
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return "unknown"
    return f"sm{major}{minor}"


def _tuning_key(autotuner, args, kwargs):
    """Shape/dtype signature for one autotuner invocation.

    Mirrors the key Triton itself builds (the ``keys`` arguments plus the dtypes
    of the tensor arguments). It does not have to match Triton's byte for byte:
    if the layouts ever drift apart the only consequence is a table miss, which
    falls back to the deterministic min-cost config — never to timing-based
    selection.
    """
    named = dict(zip(getattr(autotuner, "arg_names", ()) or (), args))
    named.update(kwargs)
    parts = [
        f"{name}={named[name]}" for name in getattr(autotuner, "keys", ()) or () if name in named
    ]
    parts.extend(str(value.dtype) for value in named.values() if hasattr(value, "dtype"))
    return "|".join(parts)


def _kernel_name(autotuner):
    fn = getattr(autotuner, "base_fn", None) or getattr(autotuner, "fn", None)
    seen = 0
    while fn is not None and not hasattr(fn, "__name__") and hasattr(fn, "fn") and seen < 8:
        fn = fn.fn
        seen += 1
    return getattr(fn, "__name__", "") or ""


def _lookup_tuned_config(kernel_name, key, candidates):
    """Return the pre-tuned config for this kernel/shape, or ``None``.

    The stored entry is matched back against the kernel's own candidate list
    rather than rebuilt, so fields the table does not carry (``pre_hook``,
    ``num_ctas``, ``maxnreg``) are preserved exactly.
    """
    from megatron.core.ssm.ops.tuned_autotune_configs import TUNED_AUTOTUNE_CONFIGS

    entries = TUNED_AUTOTUNE_CONFIGS.get(_arch_tag(), {}).get(kernel_name)
    if not entries:
        return None
    wanted = entries.get(key) or entries.get("*")
    if not wanted:
        return None
    for config in candidates:
        if (
            config.kwargs == wanted.get("kwargs")
            and getattr(config, "num_warps", None) == wanted.get("num_warps")
            and getattr(config, "num_stages", None) == wanted.get("num_stages")
        ):
            return config
    return None


def _deterministic_choice(autotuner, candidates, args, kwargs):
    """Pick one config without measuring anything.

    Preference order: the pre-tuned winner for this kernel and shape, then an
    explicit ``TRITON_AUTOTUNE_BLOCK_*`` override, then the cheapest config.
    """
    kernel_name = _kernel_name(autotuner)
    tuned = _lookup_tuned_config(kernel_name, _tuning_key(autotuner, args, kwargs), candidates)
    if tuned is not None:
        return tuned
    if kernel_name not in _untuned_kernels_warned:
        _untuned_kernels_warned.add(kernel_name)
        warnings.warn(
            f"Deterministic mode: no pre-tuned config for triton kernel {kernel_name!r} on "
            f"{_arch_tag()}; using the cheapest config, which is deterministic but may be "
            "slower. Record a table with MCORE_DET_TUNE_RECORD to recover the throughput."
        )
    filtered = _filter_configs_by_block_sizes(candidates)
    return (filtered or [min(candidates, key=_estimate_config_cost)])[0]


def _record_tuned_winner(autotuner, args, kwargs):
    """Capture the config a real autotuning pass selected."""
    config = getattr(autotuner, "best_config", None)
    if config is None:
        return
    kernels = _tune_records.setdefault(_arch_tag(), {}).setdefault(_kernel_name(autotuner), {})
    kernels[_tuning_key(autotuner, args, kwargs)] = {
        "kwargs": dict(config.kwargs),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
    }


def _dump_tune_records():
    if not _tune_records:
        return
    import json

    rank = os.environ.get("RANK", "0")
    with open(f"{_TUNE_RECORD_PATH}.rank{rank}.json", "w") as handle:
        json.dump(_tune_records, handle, indent=1, sort_keys=True)


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
        if not (
            len(getattr(self, "configs", ())) > 1
            and use_deterministic_mode()
            and _kernel_module(self).startswith("mamba_ssm")
        ):
            return original_run(self, *args, **kwargs)

        if _TUNE_RECORD_PATH:
            # Tuning pass: let Triton benchmark as usual and record its winner.
            # This run is deliberately not bit-exact; its output is the table.
            result = original_run(self, *args, **kwargs)
            _record_tuned_winner(self, args, kwargs)
            return result

        # Restore the full list afterwards: the choice is per shape, so a later
        # call with different shapes must still see every candidate.
        candidates = self.configs
        try:
            self.configs = [_deterministic_choice(self, candidates, args, kwargs)]
            return original_run(self, *args, **kwargs)
        finally:
            self.configs = candidates

    Autotuner.run = deterministic_run
    if _TUNE_RECORD_PATH:
        atexit.register(_dump_tune_records)
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
