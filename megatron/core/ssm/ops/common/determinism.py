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


# ---------------------------------------------------------------------------
# Autotune choice log.
#
# Which config each kernel ends up running is the *cause* of the reduction-order
# class of nondeterminism; the diverging tensors are the effect. Recording the
# choice costs one dict insert per (kernel, shape) and no host sync, so it can
# stay on in production, and it answers "did two runs pick the same tiling?"
# without tracing a single tensor.
# ---------------------------------------------------------------------------

_choice_log: dict = {}

# Positive control. Every check we have is a negative one — "the runs matched" —
# which cannot distinguish a working instrument from a broken one. With
# DET_AUTOTUNE_CHAOS=1 each rank deliberately picks a different config, so a
# correct differ MUST report divergence localized to these kernels' consumers.
# The choice is a hash of (rank, kernel, shape), so the run stays reproducible.
_CHAOS_AUTOTUNE = os.environ.get("DET_AUTOTUNE_CHAOS", "0") == "1"


def _chaos_choice(autotuner, candidates, args, kwargs):
    """Pick a per-rank config on purpose, reproducibly."""
    import hashlib

    seed = (
        f"{os.environ.get('RANK', '0')}|{_kernel_name(autotuner)}"
        f"|{_tuning_key(autotuner, args, kwargs)}"
    )
    index = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % len(candidates)
    return candidates[index]


def _config_signature(config) -> str:
    """Stable text form of one Triton config."""
    if config is None:
        return "none"
    kwargs = ",".join(f"{k}={v}" for k, v in sorted(getattr(config, "kwargs", {}).items()))
    return (
        f"{kwargs};warps={getattr(config, 'num_warps', None)}"
        f";stages={getattr(config, 'num_stages', None)}"
    )


def _record_choice(autotuner, args, kwargs, config, pinned: bool) -> None:
    """Note the config chosen for one (kernel, shape), the first time only."""
    key = f"{_arch_tag()}|{_kernel_name(autotuner)}|{_tuning_key(autotuner, args, kwargs)}"
    if key in _choice_log:
        return
    _choice_log[key] = f"{_config_signature(config)};{'pinned' if pinned else 'timed'}"


def autotune_choice_digest() -> str:
    """Hash of every autotune choice this rank has made so far."""
    import hashlib

    payload = "\n".join(f"{key}={value}" for key, value in sorted(_choice_log.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def verify_autotune_choices(group=None) -> bool:
    """Assert every rank picked the same config for every kernel and shape.

    Call at a point all ranks reach, such as a step boundary — never at the
    moment of choice, because ranks reach a given kernel at different times and a
    collective there would deadlock. Returns True when the choices agree.

    A mismatch is the reduction-order bug caught at its cause, before it has
    turned into diverging tensors. ``DET_AUTOTUNE_VERIFY_STRICT=1`` raises
    instead of warning.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    digest = autotune_choice_digest()
    digests: list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(digests, digest, group=group)
    if len(set(digests)) == 1:
        return True

    # Only on mismatch: gather the full maps and name the offending kernels.
    maps: list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(maps, dict(_choice_log), group=group)
    offenders: dict = {}
    for key in {k for m in maps if m for k in m}:
        seen = {m.get(key) for m in maps if m}
        if len(seen) > 1:
            offenders[key] = sorted(str(value) for value in seen)
    lines = [
        f"  {key}\n    " + "\n    ".join(values) for key, values in sorted(offenders.items())[:10]
    ]
    message = (
        f"Deterministic mode: ranks disagree on {len(offenders)} autotune choice(s); "
        "the same kernel is running different tilings on different ranks, so results "
        "cannot be bit-reproducible.\n" + "\n".join(lines)
    )
    if os.environ.get("DET_AUTOTUNE_VERIFY_STRICT", "0") == "1":
        raise RuntimeError(message)
    warnings.warn(message)
    return False


def _dump_choice_log() -> None:
    path = os.environ.get("DET_AUTOTUNE_CHOICE_LOG")
    if not path or not _choice_log:
        return
    import json

    rank = os.environ.get("RANK", "0")
    with open(f"{path}.rank{rank}.json", "w") as handle:
        json.dump(_choice_log, handle, indent=1, sort_keys=True)


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


# Packages whose Triton autotuners are pinned under deterministic mode.
#
# ``mamba_ssm``: the Mamba memory-efficient path calls
# ``mamba_split_conv1d_scan_combined``, whose SSD kernels autotune by wall clock.
#
# ``transformer_engine``: ``common/triton/permutation.py`` autotunes
# ``_unpermute_kernel`` and ``_unpermute_bwd_with_merging_probs_kernel`` over
# seven ``BLOCK_SIZE`` values. Both sum each token's top-k expert contributions,
# so the block size sets the accumulation order — the same mechanism as the SSD
# kernels. These are reachable whenever ``moe_permute_fusion`` is on, which the
# nemotronh recipes set explicitly (the mcore default is off).
#
# Override with DET_AUTOTUNE_PIN_MODULES as a comma-separated prefix list.
_PINNED_AUTOTUNE_MODULES = tuple(
    prefix.strip()
    for prefix in os.environ.get("DET_AUTOTUNE_PIN_MODULES", "mamba_ssm,transformer_engine").split(
        ","
    )
    if prefix.strip()
)

# Set DET_AUTOTUNE_ENUMERATE=1 to report every multi-config autotuner Triton
# actually benchmarks, without changing which config is chosen. Answers "is any
# timing-selected kernel still unpinned on this path" from the run itself rather
# than from a static audit.
_ENUMERATE_AUTOTUNERS = os.environ.get("DET_AUTOTUNE_ENUMERATE", "0") == "1"
_ENUMERATED: set = set()


def _enumerate_autotuner(module: str, name: str, count: int, pinned: bool) -> None:
    """Log one line the first time a multi-config autotuner is seen."""
    key = (module, name)
    if key in _ENUMERATED:
        return
    _ENUMERATED.add(key)
    warnings.warn(
        f"[det-autotune] {'PINNED ' if pinned else 'UNPINNED'} "
        f"{module}.{name} ({count} configs)",
        stacklevel=2,
    )


def pin_external_mamba_autotuners():
    """Pin timing-based Triton autotuners in external packages.

    Kernels carrying ``@triton.autotune`` benchmark their candidate configs at
    first call, so the winner — and with it the floating-point reduction order —
    can differ across processes and GPUs from clock jitter alone. Under
    deterministic mode, prune each such autotuner in a pinned package to a single
    value-deterministically chosen config (same rule as ``autotune_configs``).
    Scoped by module prefix so unrelated Triton autotuners keep their tuned
    performance; see ``_PINNED_AUTOTUNE_MODULES``.

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
        count = len(getattr(self, "configs", ()))
        module = _kernel_module(self)
        pinned = module.startswith(_PINNED_AUTOTUNE_MODULES) if _PINNED_AUTOTUNE_MODULES else False
        if _ENUMERATE_AUTOTUNERS and count > 1:
            _enumerate_autotuner(module, _kernel_name(self), count, pinned)
        if not (count > 1 and use_deterministic_mode() and pinned):
            result = original_run(self, *args, **kwargs)
            if count > 1:
                # Timing-selected: read back what Triton picked. This is the case
                # that can differ between ranks, so it is the one worth logging.
                _record_choice(self, args, kwargs, getattr(self, "best_config", None), False)
            return result

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
            if _CHAOS_AUTOTUNE:
                chosen = _chaos_choice(self, candidates, args, kwargs)
            else:
                chosen = _deterministic_choice(self, candidates, args, kwargs)
            _record_choice(self, args, kwargs, chosen, True)
            self.configs = [chosen]
            return original_run(self, *args, **kwargs)
        finally:
            self.configs = candidates

    Autotuner.run = deterministic_run
    if _TUNE_RECORD_PATH:
        atexit.register(_dump_tune_records)
    if os.environ.get("DET_AUTOTUNE_CHOICE_LOG"):
        atexit.register(_dump_choice_log)
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
