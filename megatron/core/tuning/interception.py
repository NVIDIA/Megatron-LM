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

from megatron.core.tuning import selection
from megatron.core.tuning import table as table_mod
from megatron.core.tuning.policy import AutotunePolicy, use_deterministic_mode

_installed = False
_policy: AutotunePolicy | None = None
_table = None

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
        f"{selection.arch_tag()}|{selection.kernel_name(autotuner)}"
        f"|{selection.tuning_key(autotuner, args, kwargs)}"
    )
    if key in _choice_log:
        return
    _choice_log[key] = f"{selection.config_signature(config)};{'pinned' if pinned else 'timed'}"


def _record_winner(autotuner, args, kwargs) -> None:
    config = getattr(autotuner, "best_config", None)
    if config is None:
        return
    kernels = _tune_records.setdefault(selection.arch_tag(), {}).setdefault(
        selection.kernel_name(autotuner), {}
    )
    kernels[selection.tuning_key(autotuner, args, kwargs)] = {
        "kwargs": dict(config.kwargs),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
    }


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
    """Assert every rank picked the same config for every kernel and shape.

    Call where all ranks arrive, such as a step boundary. Calling it at the
    moment of choice would deadlock: ranks reach a given kernel at different
    times, so they would not agree on whether to take part in the collective.

    A mismatch is the reduction-order bug caught at its cause, before it has
    become diverging tensors.
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
        seen = {m.get(key) for m in maps if m}
        if len(seen) > 1:
            offenders[key] = sorted(str(v) for v in seen)
    lines = [f"  {k}\n    " + "\n    ".join(v) for k, v in sorted(offenders.items())[:10]]
    message = (
        f"Ranks disagree on {len(offenders)} autotune choice(s); the same kernel is "
        "running different tilings on different ranks, so results cannot be "
        "bit-reproducible.\n" + "\n".join(lines)
    )
    if _policy is not None and _policy.verify_strict:
        raise RuntimeError(message)
    warnings.warn(message)
    return False


def install(policy: AutotunePolicy | None = None) -> bool:
    """Apply ``policy`` to Triton's autotuner. Idempotent.

    Returns whether an interception is active. A no-op when the policy does not
    need one, or when triton is unavailable.
    """
    global _installed, _policy, _table

    policy = policy or AutotunePolicy.from_env()
    if _installed:
        return True
    _policy = policy
    if not policy.intercepts:
        return False
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return False

    _table = table_mod.load(selection.arch_tag(), policy.table_path)
    original_run = Autotuner.run

    def policy_run(self, *args, **kwargs):
        count = len(getattr(self, "configs", ()))
        module = selection.kernel_module(self)
        in_scope = module.startswith(policy.modules) if policy.modules else False

        if policy.enumerate_autotuners and count > 1:
            _enumerate(module, selection.kernel_name(self), count, in_scope)

        if count <= 1 or not in_scope or policy.mode == "auto":
            result = original_run(self, *args, **kwargs)
            if count > 1:
                # Timed selection: read back what Triton picked. This is the case
                # that can differ between ranks, so it is the one worth logging.
                _record_choice(self, args, kwargs, getattr(self, "best_config", None), False)
            return result

        if policy.mode == "record":
            # Let Triton benchmark as usual and capture its winner. A recording
            # run is deliberately not reproducible; its output is the table.
            result = original_run(self, *args, **kwargs)
            _record_winner(self, args, kwargs)
            _record_choice(self, args, kwargs, getattr(self, "best_config", None), False)
            return result

        candidates = self.configs
        try:
            if policy.chaos:
                chosen = selection.chaos_choice(self, candidates, args, kwargs)
            else:
                chosen = selection.deterministic_choice(
                    self, candidates, args, kwargs, table=_table, on_miss=policy.on_miss
                )
            _record_choice(self, args, kwargs, chosen, True)
            # Triton only benchmarks when more than one candidate remains, so a
            # single-entry list skips the timing loop entirely.
            self.configs = [chosen]
            return original_run(self, *args, **kwargs)
        finally:
            # The choice is per shape: a later call must see every candidate again.
            self.configs = candidates

    Autotuner.run = policy_run
    if policy.mode == "record":
        atexit.register(_dump_records)
    _installed = True
    return True


def install_from_env() -> bool:
    """Convenience wrapper used by framework initialization."""
    return install(AutotunePolicy.from_env())


__all__ = [
    "active_policy",
    "choice_digest",
    "choice_log",
    "install",
    "install_from_env",
    "use_deterministic_mode",
    "verify_choices",
]
