# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Carrying out an :class:`AutotunePolicy` against cudnn-frontend's cuTile launch autotune.

The linear-attention cuTile kernels (GDN, KDA) time ``occupancy x num_worker_warps``
candidates per process in ``cudnn.linear_attention.cutile.kernels.common.autotuned_launch``.
The worker-warp count changes the summation tree of a kernel that reduces a tile with
``ct.sum``, so two processes that settle on different winners write different bytes.
cudnn-frontend exposes two hooks on that launch (NVIDIA/cudnn-frontend#1735):
``select_hints(kernel, cache_key, configs)`` returns the candidate to run and skips the
timing, and ``record_hint(kernel, cache_key, config)`` is told each timed winner.

This module binds them to the installed policy, with the same modes, tables and
diagnostics as the Triton adapter: ``pinned`` runs the table entry, or the first
candidate of the sweep when the table misses; ``record`` captures the timed winners into
the per-rank recording so ``python -m megatron.core.tuning merge`` builds a table from
them; every choice is logged for ``verify_choices``. Kernels are keyed by their
qualified name (``cudnn.linear_attention.cutile.kernels.gdn.chunk_fwd_kernel_o``) and
the shape key the launch passes, which carries no device, so the entries match on every
rank. A cudnn-frontend without the hooks is left alone, with one warning when a policy
would have pinned it.
"""

from __future__ import annotations

import logging

from megatron.core.tuning import selection
from megatron.core.tuning import table as table_mod
from megatron.core.tuning.policy import AutotunePolicy

logger = logging.getLogger(__name__)

HOOKS = ("select_hints", "record_hint")
_unhookable_warned = False
# Per hook, what this adapter displaced (a host's callback, or None), put back when the policy lets go.
_previous: dict = {name: None for name in HOOKS}


def _common():
    """cudnn-frontend's cuTile launch module when it carries the hooks, else ``None``."""
    global _unhookable_warned
    try:
        from cudnn.linear_attention.cutile.kernels import common
    except ImportError:
        return None
    if all(hasattr(common, name) for name in HOOKS):
        return common
    if not _unhookable_warned:
        _unhookable_warned = True
        logger.warning(
            "This cudnn-frontend has no cuTile autotune hooks (NVIDIA/cudnn-frontend#1735); "
            "its linear-attention cuTile kernels keep their timed launch picks."
        )
    return None


def qualified_name(kernel) -> tuple[str, str]:
    """``(module, qualified)`` of a ``@ct.kernel``, from the Python function it wraps."""
    fn = getattr(kernel, "_pyfunc", None)
    module = getattr(fn, "__module__", None) or "cudnn"
    name = getattr(fn, "__name__", None) or repr(kernel)
    return module, f"{module}.{name}"


def shape_key(cache_key) -> str:
    """The launch's shape key as table text."""
    if isinstance(cache_key, (tuple, list)):
        return "|".join(str(part) for part in cache_key)
    return str(cache_key)


def config_data(config) -> dict:
    """Serializable launch hints of one candidate (``occupancy``, ``num_worker_warps``, ...)."""
    return dict(sorted(vars(config).items()))


def config_signature(config) -> str:
    import json

    return json.dumps(config_data(config), sort_keys=True)


def select(kernel, cache_key, configs):
    """``select_hints``: the pinned candidate, or ``None`` to let cudnn-frontend time them."""
    from megatron.core.tuning import interception

    policy = interception.active_policy()
    if policy is None or not policy.intercepts:
        return None
    module, qualified = qualified_name(kernel)
    pinned = policy.mode == "pinned" and policy.governs(module, qualified)
    if policy.enumerate_autotuners and len(configs) > 1:
        interception._enumerate(qualified, len(configs), "PINNED" if pinned else "UNPINNED")
    if not pinned:
        return None
    arch = selection.arch_tag()
    key = shape_key(cache_key)
    if policy.chaos:
        chosen = selection.chaos_pick(configs, qualified, key)
    else:
        if arch not in interception._tables:
            interception._tables[arch] = table_mod.load(arch, policy.table_path)
        entries = interception._tables[arch].kernels.get(qualified) or {}
        wanted = entries.get(key) or entries.get("*")
        chosen = next((c for c in configs if config_data(c) == wanted), None) if wanted else None
        if chosen is None:
            if policy.on_miss == "error":
                raise RuntimeError(
                    f"No tuned config for cuTile kernel {qualified!r} at shape {key!r} on {arch} "
                    "and on_miss='error'; record a table or relax on_miss"
                )
            if qualified not in selection._untuned_kernels_warned:
                selection._untuned_kernels_warned.add(qualified)
                logger.warning(
                    "No pre-tuned config for cuTile kernel %r on %s; using the sweep's first "
                    "candidate %s, which is deterministic but may be slower. Record a table "
                    "with AutotunePolicy(mode='record', record_path=...) to recover the "
                    "throughput.",
                    qualified,
                    arch,
                    config_data(configs[0]),
                )
            chosen = configs[0]
    interception._log_choice(arch, qualified, key, config_signature(chosen), pinned=True)
    return chosen


def record(kernel, cache_key, config):
    """``record_hint``: capture a timed winner, and log it for the agreement check."""
    from megatron.core.tuning import interception

    policy = interception.active_policy()
    if policy is None or not policy.intercepts:
        return
    module, qualified = qualified_name(kernel)
    if not policy.governs(module, qualified):
        return
    arch = selection.arch_tag()
    key = shape_key(cache_key)
    if policy.mode == "record":
        interception._note_winner(arch, qualified, key, config_data(config))
    interception._log_choice(arch, qualified, key, config_signature(config), pinned=False)


def apply(policy: AutotunePolicy) -> bool:
    """Take the hooks for an intercepting ``policy``, or give back what was displaced; ``True`` when held.

    Ownership is tracked per hook: a host that replaces one of them while the policy holds
    both keeps that one, and the other is still restored. Taking a hook over drops the launch
    cache, since the picks in it were made by someone else.
    """
    common = _common()
    if common is None:
        return False
    ours = dict(zip(HOOKS, (select, record)))
    taken = False
    for name in HOOKS:
        current = getattr(common, name)
        if policy.intercepts:
            if current is not ours[name]:
                _previous[name] = current
                taken = True
                if current is not None:
                    logger.warning(
                        "Megatron's autotune policy takes over cudnn-frontend's cuTile hook %s "
                        "that another host installed; it returns when the policy stops "
                        "intercepting.",
                        name,
                    )
            setattr(common, name, ours[name])
        elif current is ours[name]:
            setattr(common, name, _previous[name])
            _previous[name] = None
    if taken:
        common.launch_hint_cache.clear()
    return policy.intercepts


def invalidate() -> None:
    """Drop cudnn-frontend's cached launch picks, so the next launch of every shape consults the hooks.

    cudnn-frontend asks ``select_hints`` only when a shape misses its launch cache; a pick
    timed under the previous policy would otherwise outlive the policy change on every rank
    that made it.
    """
    common = _common()
    if common is not None:
        common.launch_hint_cache.clear()


__all__ = ["apply", "config_data", "invalidate", "qualified_name", "record", "select", "shape_key"]
