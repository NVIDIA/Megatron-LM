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


def in_scope(module: str, policy: AutotunePolicy) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in policy.modules)


def select(kernel, cache_key, configs):
    """``select_hints``: the pinned candidate, or ``None`` to let cudnn-frontend time them."""
    from megatron.core.tuning import interception

    policy = interception.active_policy()
    if policy is None or not policy.intercepts:
        return None
    module, qualified = qualified_name(kernel)
    pinned = (
        policy.mode == "pinned"
        and in_scope(module, policy)
        and qualified not in policy.config_invariant
    )
    if policy.enumerate_autotuners and len(configs) > 1:
        interception._enumerate(qualified, len(configs), "PINNED" if pinned else "UNPINNED")
    if not pinned:
        return None
    arch = selection.arch_tag()
    key = shape_key(cache_key)
    if policy.chaos:
        import random

        chosen = random.choice(configs)
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
    if not in_scope(module, policy):
        return
    arch = selection.arch_tag()
    key = shape_key(cache_key)
    if policy.mode == "record":
        interception._note_winner(arch, qualified, key, config_data(config))
    interception._log_choice(arch, qualified, key, config_signature(config), pinned=False)


def apply(policy: AutotunePolicy) -> bool:
    """Bind or clear the hooks for ``policy``; ``True`` when cudnn-frontend is hooked."""
    common = _common()
    if common is None:
        return False
    hooked = policy.intercepts
    common.select_hints = select if hooked else None
    common.record_hint = record if hooked else None
    return hooked


__all__ = ["apply", "config_data", "qualified_name", "record", "select", "shape_key"]
