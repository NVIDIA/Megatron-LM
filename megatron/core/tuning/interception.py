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
import logging
import os
from collections.abc import Mapping
from functools import wraps
from weakref import WeakKeyDictionary, WeakSet

from megatron.core._rank_utils import log_single_rank, safe_get_rank
from megatron.core.tuning import selection
from megatron.core.tuning import table as table_mod
from megatron.core.tuning.policy import AutotunePolicy, coerce_policy

logger = logging.getLogger(__name__)

_installed = False
_policy: AutotunePolicy | None = None
_tables: dict = {}
# Keep each autotuner's live configs (and hooks) separate, even for equal names.
_selected_configs: WeakKeyDictionary = WeakKeyDictionary()
# Per-autotuner scope, computed once per policy: (module, name, qualified, in_scope, invariant).
_scopes: WeakKeyDictionary = WeakKeyDictionary()
# Autotuners whose single candidate has already been logged.
_logged_singletons: WeakSet = WeakSet()

# (arch, kernel, shape) -> chosen config. Which config a kernel runs is the *cause*
# of reduction-order nondeterminism; diverging tensors are the effect. A choice is
# logged once, when it is made, so steady-state launches pay nothing for it.
_choice_log: dict = {}
# Choices logged since the last agreement check, and those already agreed on.
_unverified: dict = {}
_verified: dict = {}
_tune_records: dict = {}
_record_rank: int | None = None
_enumerated: set = set()


def active_policy() -> AutotunePolicy | None:
    """The policy currently installed, if any."""
    return _policy


def _scope(autotuner, policy: AutotunePolicy):
    scope = _scopes.get(autotuner)
    if scope is None:
        module = selection.kernel_module(autotuner)
        name = selection.kernel_name(autotuner)
        qualified = f"{module}.{name}"
        in_scope = any(
            module == prefix or module.startswith(prefix + ".") for prefix in policy.modules
        )
        scope = (module, name, qualified, in_scope, qualified in policy.config_invariant)
        _scopes[autotuner] = scope
    return scope


def _enumerate(qualified: str, count: int, state: str) -> None:
    if qualified in _enumerated:
        return
    _enumerated.add(qualified)
    logger.warning("[autotune] %-8s %s (%d configs)", state, qualified, count)


def _record_choice(arch: str, qualified: str, key: str, config, pinned: bool) -> None:
    entry = f"{arch}|{qualified}|{key}"
    value = f"{selection.config_signature(config)};{'pinned' if pinned else 'timed'}"
    if _choice_log.get(entry) != value:
        _choice_log[entry] = value
        _unverified[entry] = value


def _record_winner(arch: str, qualified: str, key: str, config) -> None:
    global _record_rank
    if config is None:
        return
    if _record_rank is None:
        # Resolve the rank while the process group is alive; the capture is written
        # at exit, possibly after the group has been destroyed.
        _record_rank = safe_get_rank()
    _tune_records.setdefault(arch, {}).setdefault(qualified, {})[key] = selection.config_data(
        config
    )


def _record_file(record_path: str, rank: int) -> str:
    return f"{record_path}.rank{rank}.json"


def _check_record_path(record_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(record_path))
    os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(
            f"Cannot write Triton autotune recordings under {directory!r} "
            f"(record_path={record_path!r})"
        )


def _dump_records() -> None:
    if not _tune_records or _policy is None or not _policy.record_path:
        return
    rank = _record_rank if _record_rank is not None else safe_get_rank()
    path = _record_file(_policy.record_path, rank)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(_tune_records, handle, indent=1, sort_keys=True)
        # Readers never see a partially written capture.
        os.replace(temporary, path)
    except OSError as exc:
        logger.error("Could not write Triton autotune recording %s: %s", path, exc)
        return
    logger.info("Wrote Triton autotune recording %s", path)


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

    Each check exchanges only the choices made since the previous check and
    compares them with those already agreed on, so once every kernel and shape
    has been seen a check moves an empty payload. Pipeline stages and expert
    ranks can execute different kernels or shapes; an absent key is not a
    conflicting choice. Agreement only covers observed choices; it does not
    establish numerical determinism or equal coverage.
    """
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    # all_gather_object requires one slot per member of ``group``, which is not the
    # global world size once a caller verifies agreement within a DP or TP subgroup.
    world_size = torch.distributed.get_world_size(group=group)
    pending = dict(_unverified)
    _unverified.clear()
    gathered: list = [None] * world_size
    torch.distributed.all_gather_object(gathered, pending, group=group)

    observed: dict = {}
    for entries in gathered:
        for key, value in (entries or {}).items():
            observed.setdefault(key, set()).add(value)
    offenders: dict = {}
    for key, values in observed.items():
        if key in _verified:
            values = values | {_verified[key]}
        if len(values) > 1:
            offenders[key] = sorted(str(v) for v in values)
        else:
            _verified[key] = next(iter(values))
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
    # Every rank reaches the same verdict, so one copy of the report is enough.
    first_rank = 0 if group is None else torch.distributed.get_process_group_ranks(group)[0]
    log_single_rank(logger, logging.WARNING, message, rank=first_rank)
    return False


def maybe_verify_choices(iteration: int, group=None) -> bool | None:
    """Run :func:`verify_choices` if the policy asks for a check at ``iteration``.

    This is the cadence behind ``AutotunePolicy.verify_every``; the training loop calls
    it once per step and the policy decides whether anything happens. Returns
    ``None`` when no check ran, so a caller can tell "agreed" from "not asked".
    """
    if _policy is None or _policy.verify_every <= 0:
        return None
    if iteration % _policy.verify_every:
        return None
    return verify_choices(group=group)


def install(policy: AutotunePolicy | Mapping | None = None, *, deterministic: bool = False) -> bool:
    """Apply ``policy`` to the whole process, replacing the previous one.

    Call during initialization, before kernels run. Megatron's training
    initializer does this from the ``--triton-autotune-*`` arguments; library
    callers that build models directly call it themselves. ``None`` applies the
    default policy, and a mapping is converted. An omitted ``mode`` is derived:
    a recording path selects ``record``, ``deterministic`` or PyTorch's
    deterministic flag selects ``pinned``, and anything else ``auto``. A policy
    that fails to install, such as one whose recording path is not writable,
    leaves the previous one in place. Repeated calls reuse the same adapter
    rather than nesting patches.

    Returns whether the adapter is installed.
    """
    policy = coerce_policy(policy) or AutotunePolicy()
    return _install(policy.resolve(deterministic=deterministic))


def _reset_runtime_state() -> None:
    global _record_rank

    _tables.clear()
    _selected_configs.clear()
    _scopes.clear()
    _logged_singletons.clear()
    _choice_log.clear()
    _unverified.clear()
    _verified.clear()
    _tune_records.clear()
    _record_rank = None
    _enumerated.clear()
    selection.reset_warnings()


def _install(policy: AutotunePolicy) -> bool:
    global _installed, _policy

    if policy != _policy:
        if policy.mode == "record":
            # Fail now rather than when the recording is written at exit.
            _check_record_path(policy.record_path)
        _dump_records()
        _policy = policy
        _reset_runtime_state()
        selection.warn_if_mamba_env_ignored(policy)

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
        _, _, qualified, in_scope, invariant = _scope(self, policy)
        pinned = in_scope and not invariant and policy.mode == "pinned"

        if policy.enumerate_autotuners and count > 1:
            if pinned:
                state = "PINNED"
            elif in_scope and invariant and policy.mode == "pinned":
                state = "TIMED"
            else:
                state = "UNPINNED"
            _enumerate(qualified, count, state)

        if count <= 1 or not pinned:
            observed = in_scope and not invariant
            tuned_before = len(getattr(self, "cache", ()))
            result = original_run(self, *args, **kwargs)
            if not observed:
                return result
            if count > 1:
                # Triton caches its winner per key, so a choice is new exactly when
                # the cache grew; cached launches skip the bookkeeping.
                if len(getattr(self, "cache", ())) == tuned_before:
                    return result
                arch = selection.arch_tag()
                key = selection.tuning_key(self, args, kwargs)
                config = getattr(self, "best_config", None)
                if policy.mode == "record":
                    _record_winner(arch, qualified, key, config)
                _record_choice(arch, qualified, key, config, pinned=False)
            elif self not in _logged_singletons:
                # A single candidate is the same for every shape; nothing was
                # timed, so there is nothing to record as a winner either.
                _logged_singletons.add(self)
                _record_choice(
                    selection.arch_tag(), qualified, "*", getattr(self, "best_config", None), pinned
                )
            return result

        candidates = self.configs
        nargs = getattr(self, "nargs", None)
        # Device selection can follow framework initialization. Include the current
        # architecture and Triton's declared tuning inputs in each cache entry.
        arch = selection.arch_tag()
        key = selection.tuning_key(self, args, kwargs)
        cache = _selected_configs.get(self)
        if cache is None:
            cache = _selected_configs[self] = {}
        try:
            chosen = cache.get((arch, key))
            new_choice = chosen is None
            if new_choice:
                # Preserve Triton's pruning on the first invocation of each key.
                # It expects positional arguments in self.nargs, as in Autotuner.run.
                self.nargs = dict(zip(self.arg_names, args))
                valid_configs = self.prune_configs(kwargs)
                if not valid_configs:
                    raise RuntimeError(f"No valid configs for Triton kernel {qualified!r}")
                if policy.chaos:
                    chosen = selection.chaos_choice(self, valid_configs, args, kwargs)
                else:
                    if arch not in _tables:
                        _tables[arch] = table_mod.load(arch, policy.table_path)
                    chosen = selection.deterministic_choice(
                        self,
                        valid_configs,
                        args,
                        kwargs,
                        table=_tables[arch],
                        on_miss=policy.on_miss,
                        block_sizes=policy.block_sizes,
                    )
            # Triton only benchmarks when more than one candidate remains, so a
            # single-entry list skips the timing loop entirely.
            self.configs = [chosen]
            result = original_run(self, *args, **kwargs)
            if new_choice:
                cache[(arch, key)] = chosen
                _record_choice(arch, qualified, key, chosen, pinned=True)
            return result
        finally:
            # The choice is per shape: a later call must see every candidate again.
            self.configs = candidates
            self.nargs = nargs

    Autotuner.run = policy_run
    atexit.register(_dump_records)
    _installed = True
    return True


__all__ = [
    "active_policy",
    "choice_digest",
    "choice_log",
    "install",
    "maybe_verify_choices",
    "verify_choices",
]
