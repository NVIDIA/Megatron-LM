# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Carrying out an :class:`AutotunePolicy` against Triton's autotuner.

This is the only module that touches Triton internals. Everything else works
against the policy and the table, so replacing this file is all that is needed
when Triton grows a supported hook, or when the upstream packages stop needing
the intervention at all.
"""

from __future__ import annotations

import atexit
import json
import os
import warnings
from functools import wraps

from megatron.core.tuning import selection
from megatron.core.tuning import table as table_mod
from megatron.core.tuning.policy import AutotunePolicy, use_deterministic_mode

_installed = False
_policy: AutotunePolicy | None = None
_explicit_policy = False
_tables: dict = {}

# (kernel, shape) -> chosen config. Which config a kernel runs is the *cause* of
# reduction-order nondeterminism; diverging tensors are the effect. Recording it
# costs one dict insert and no host sync, so it can stay on in production.
_choice_log: dict = {}
_tune_records: dict = {}
_enumerated: set = set()


def active_policy() -> AutotunePolicy | None:
    """The policy currently installed, if any."""
    return _policy


def _enumerate(module: str, name: str, count: int, pinned: bool) -> None:
    key = (module, name)
    if key in _enumerated:
        return
    _enumerated.add(key)
    warnings.warn(
        f"[autotune] {'PINNED  ' if pinned else 'UNPINNED'} {module}.{name} ({count} configs)",
        stacklevel=2,
    )


def _record_choice(autotuner, args, kwargs, config, pinned: bool) -> None:
    key = (
        f"{selection.arch_tag()}|{selection.kernel_module(autotuner)}."
        f"{selection.kernel_name(autotuner)}"
        f"|{selection.tuning_key(autotuner, args, kwargs)}"
    )
    _choice_log[key] = f"{selection.config_signature(config)};{'pinned' if pinned else 'timed'}"


def _record_winner(autotuner, args, kwargs) -> None:
    config = getattr(autotuner, "best_config", None)
    if config is None:
        return
    kernels = _tune_records.setdefault(selection.arch_tag(), {}).setdefault(
        selection.kernel_name(autotuner), {}
    )
    kernels[selection.tuning_key(autotuner, args, kwargs)] = selection.config_data(config)


def _dump_records() -> None:
    if not _tune_records or _policy is None or not _policy.record_path:
        return
    rank = os.environ.get("RANK", "0")
    path = f"{_policy.record_path}.rank{rank}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_tune_records, handle, indent=1, sort_keys=True)


def choice_digest() -> str:
    """Hash of every autotune choice this rank has made so far."""
    import hashlib

    payload = "\n".join(f"{key}={value}" for key, value in sorted(_choice_log.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def choice_log() -> dict:
    """The per-(kernel, shape) choices this rank has made."""
    return dict(_choice_log)


def verify_choices(group=None) -> bool:
    """Compare configs for kernel/shape keys observed on multiple ranks.

    Call where all ranks arrive, such as a step boundary. Calling it at the
    moment of choice would deadlock: ranks reach a given kernel at different
    times, so they would not agree on whether to take part in the collective.

    Pipeline stages and expert ranks can execute different kernels or shapes.
    An absent key is not a conflicting choice. Agreement only covers observed
    choices; it does not establish numerical determinism or equal coverage.
    """
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    # all_gather_object requires one slot per member of ``group``, which is not the
    # global world size once a caller verifies agreement within a DP or TP subgroup.
    world_size = torch.distributed.get_world_size(group=group)
    digests: list = [None] * world_size
    torch.distributed.all_gather_object(digests, choice_digest(), group=group)
    if len(set(digests)) == 1:
        return True

    maps: list = [None] * world_size
    torch.distributed.all_gather_object(maps, dict(_choice_log), group=group)
    offenders: dict = {}
    for key in {k for m in maps if m for k in m}:
        seen = {m[key] for m in maps if m and key in m}
        if len(seen) > 1:
            offenders[key] = sorted(str(v) for v in seen)
    if not offenders:
        return True
    lines = [f"  {k}\n    " + "\n    ".join(v) for k, v in sorted(offenders.items())[:10]]
    message = (
        f"Ranks disagree on {len(offenders)} autotune choice(s); the same kernel is "
        "running different configurations on different ranks, which may change "
        "floating-point results.\n" + "\n".join(lines)
    )
    if _policy is not None and _policy.verify_strict:
        raise RuntimeError(message)
    warnings.warn(message)
    return False


def maybe_verify_choices(iteration: int, group=None) -> bool | None:
    """Run :func:`verify_choices` if the policy asks for a check at ``iteration``.

    This is the cadence behind ``MCORE_AUTOTUNE_VERIFY``; the training loop calls
    it once per step and the policy decides whether anything happens. Returns
    ``None`` when no check ran, so a caller can tell "agreed" from "not asked".
    """
    if _policy is None or _policy.verify_every <= 0:
        return None
    if iteration % _policy.verify_every:
        return None
    return verify_choices(group=group)


def install(policy: AutotunePolicy | None = None) -> bool:
    """Apply a process-wide policy, replacing any previously selected policy.

    Install once during initialization, before kernels execute. An explicit
    policy takes precedence over subsequent framework ``install_from_env``
    calls. Repeated calls reuse the same adapter rather than nesting patches.
    """
    global _explicit_policy

    _explicit_policy = policy is not None
    return _install(policy or AutotunePolicy.from_env())


def _install(policy: AutotunePolicy) -> bool:
    global _installed, _policy

    if policy != _policy:
        _dump_records()
        _policy = policy
        _tables.clear()
        _choice_log.clear()
        _tune_records.clear()
        _enumerated.clear()

    if _installed:
        return True
    if not policy.intercepts:
        return False
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return False

    original_run = Autotuner.run

    @wraps(original_run)
    def policy_run(self, *args, **kwargs):
        policy = _policy
        if not policy.intercepts:
            return original_run(self, *args, **kwargs)
        count = len(getattr(self, "configs", ()))
        module = selection.kernel_module(self)
        in_scope = any(
            module == prefix or module.startswith(prefix + ".") for prefix in policy.modules
        )
        pinned = in_scope and policy.mode == "pinned"

        if policy.enumerate_autotuners and count > 1:
            _enumerate(module, selection.kernel_name(self), count, pinned)

        if count <= 1 or not pinned:
            result = original_run(self, *args, **kwargs)
            if in_scope and policy.mode == "record":
                _record_winner(self, args, kwargs)
            if count > 1 or in_scope:
                _record_choice(self, args, kwargs, getattr(self, "best_config", None), pinned)
            return result

        candidates = self.configs
        nargs = getattr(self, "nargs", None)
        try:
            # Preserve Triton's shape-dependent validity/performance pruning.
            # It expects positional arguments in self.nargs, as in Autotuner.run.
            self.nargs = dict(zip(self.arg_names, args))
            valid_configs = self.prune_configs(kwargs)
            if not valid_configs:
                raise RuntimeError(
                    f"No valid configs for Triton kernel {selection.kernel_name(self)!r}"
                )
            if policy.chaos:
                chosen = selection.chaos_choice(self, valid_configs, args, kwargs)
            else:
                # Framework configuration can precede CUDA device selection.
                # Load tables only at kernel execution, for the current device.
                arch = selection.arch_tag()
                if arch not in _tables:
                    _tables[arch] = table_mod.load(arch, policy.table_path)
                chosen = selection.deterministic_choice(
                    self, valid_configs, args, kwargs, table=_tables[arch], on_miss=policy.on_miss
                )
            # Triton only benchmarks when more than one candidate remains, so a
            # single-entry list skips the timing loop entirely.
            self.configs = [chosen]
            result = original_run(self, *args, **kwargs)
            _record_choice(self, args, kwargs, chosen, True)
            return result
        finally:
            # The choice is per shape: a later call must see every candidate again.
            self.configs = candidates
            self.nargs = nargs

    Autotuner.run = policy_run
    atexit.register(_dump_records)
    _installed = True
    return True


def install_from_env(*, deterministic: bool = False) -> bool:
    """Resolve framework defaults unless a caller supplied an explicit policy."""
    if _explicit_policy:
        return _installed
    # A later component's default False must not undo another model's request.
    # An explicit MCORE_AUTOTUNE_MODE still takes precedence in from_env().
    deterministic = deterministic or (_policy is not None and _policy.mode == "pinned")
    return _install(AutotunePolicy.from_env(deterministic=deterministic))


__all__ = [
    "active_policy",
    "choice_digest",
    "choice_log",
    "install",
    "install_from_env",
    "maybe_verify_choices",
    "use_deterministic_mode",
    "verify_choices",
]
