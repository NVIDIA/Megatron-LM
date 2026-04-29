# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for the *local replica* save/load mode.

Background
----------

In the standard ``FullyParallelSaveStrategyWrapper`` flow, every parameter that
is replicated across the parallelization group (the ``ep_dp`` group in our
production setup) is written by exactly one rank inside the group. The other
ranks have ``replica_id != 0`` and the underlying torch_dist save drops their
write items via ``keep_only_main_replica``. The metadata records a single
``(file, offset, length)`` triple per replicated chunk, and at load time every
peer that needs the chunk opens that single file — producing read-side
contention on the file system when the world is large.

The *local replica* mode opts into trading extra disk for read locality:
every rank that holds a replica writes its own copy. So that the metadata
can record one storage entry per copy without breaking PyT DCP's
``MetadataIndex``-keyed dedup (``MetadataIndex.index`` is declared
``compare=False``, so two copies under the same ``(fqn, offset)`` collapse on
write), each non-main rank's copy is published under a per-rank renamed FQN:
``__shadow_<global_rank>__<original_fqn>``. The double-underscore prefix is
unique enough that no real Megatron parameter starts with it.

At load time the rank simply rewrites its requested FQNs to point at its own
shadow key when one is in the metadata; the ``_StorageInfo`` for the shadow
key was authored by this rank during save, so the file the reader opens is
its local ``__<rank>_*.distcp``. No cross-rank reads.

This module hosts the FQN-rename helpers used by both
``FullyParallelSaveStrategyWrapper`` (for save-time renaming) and
``TorchDistLoadShardedStrategy`` (for load-time redirection). Keeping them
in one place makes the contract between the two sides explicit and easy to
audit.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor

SHADOW_PREFIX = "__shadow_"
"""Stable prefix used by every shadow FQN.

Decision: the prefix is purely cosmetic for external tooling (any reader that
walks ``Metadata.state_dict_metadata`` will see the extra entries and ignore
them when its state dict does not request them). We avoid the more compact
``__rR__`` style because ``__shadow_<rank>__`` is grep-friendly and
self-documenting in a binary checkpoint dump.
"""

_SHADOW_RE = re.compile(rf"^{re.escape(SHADOW_PREFIX)}(\d+)__(.+)$")


def shadow_key(global_rank: int, original_fqn: str) -> str:
    """Build the shadow FQN under which ``global_rank`` stores its local copy.

    Args:
        global_rank: ``torch.distributed.get_rank()`` at save time.
        original_fqn: the FQN the user-facing Megatron module produced
            (e.g. ``decoder.layers.0.norm.weight``).

    Returns:
        ``"__shadow_<rank>__<fqn>"`` — a string guaranteed to be unique per
        ``(rank, fqn)`` pair so that PyT DCP's ``MetadataIndex`` dedup does
        not collapse copies that come from different ranks.
    """
    return f"{SHADOW_PREFIX}{global_rank}__{original_fqn}"


def parse_shadow_key(maybe_shadow: str) -> Optional[Tuple[int, str]]:
    """Inverse of :func:`shadow_key`.

    Returns:
        ``(global_rank, original_fqn)`` if ``maybe_shadow`` is a shadow key,
        else ``None``. Used only by ad-hoc inspection / tests; the live save
        and load paths never need to reverse a shadow key.
    """
    m = _SHADOW_RE.match(maybe_shadow)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def is_shadow_key(fqn: str) -> bool:
    """True iff ``fqn`` was produced by :func:`shadow_key`.

    The save planner uses this to skip the global-plan volume validator for
    shadow entries (whose ``chunks`` list intentionally only contains the
    one rank's local chunk(s); the validator's "chunks must cover the global
    tensor" rule does not apply).
    """
    return _SHADOW_RE.match(fqn) is not None


def rewrite_replicas_to_shadow(
    sharded_state_dict: ShardedStateDict, global_rank: int
) -> int:
    """Promote every non-main local replica to a shadow saver in place.

    Walks every :class:`ShardedTensor` in ``sharded_state_dict``. If the
    tensor's ``replica_id`` is non-zero — meaning the wrapper's
    parallelization step picked another rank in the group as the main saver
    — we set ``replica_id = 0`` *and* rename ``key`` to a per-rank shadow
    FQN. After this pass every local ShardedTensor is a "saver" from the
    perspective of the underlying ``TorchDistSaveShardedStrategy``: main
    keeps the original key, shadow uses ``shadow_key(global_rank, key)``.

    The state dict is mutated in place. Callers are expected to pass the
    *save-only view* used by :meth:`FullyParallelSaveStrategyWrapper.async_save`,
    not the live training state, so this mutation does not leak to the user.

    Decision: we don't deep-copy the ShardedTensor because the underlying
    ``data`` tensor is the same object the rank already holds — a copy would
    double the host memory footprint of the save state dict for no benefit.

    Returns:
        Number of ShardedTensors that were renamed to a shadow key. Useful
        for assertions in tests.
    """
    n_renamed = 0
    for sh in nested_values(sharded_state_dict):
        if not isinstance(sh, ShardedTensor):
            continue
        if sh.replica_id == 0:
            continue
        print(f"[DEBUG shadow keys | {global_rank}] {sh.key} -> {shadow_key(global_rank, sh.key)}")
        sh.key = shadow_key(global_rank, sh.key)
        sh.replica_id = 0
        n_renamed += 1
    return n_renamed


def redirect_pyt_state_dict_to_shadows(
    pyt_state_dict: Dict[str, object],
    metadata,  # torch.distributed.checkpoint.Metadata
    global_rank: int,
) -> Dict[str, str]:
    """Route a load's PyT state dict at fqn X to ``__shadow_<rank>__X`` when present.

    Iterates the keys currently in ``pyt_state_dict``. For each one whose
    matching shadow FQN exists in ``metadata.state_dict_metadata``, renames
    the entry in ``pyt_state_dict`` to the shadow FQN. The returned mapping
    ``{shadow_fqn: original_fqn}`` is consumed by
    :func:`restore_pyt_state_dict_from_shadows` to undo the rename after
    ``checkpoint.load`` has populated the tensor data.

    Decision: we rename inside ``pyt_state_dict`` rather than on the
    upstream MCore ShardedTensor because (a) ``pyt_state_dict`` is a
    transient dict owned by the load strategy — mutating it has no
    user-visible effect, and (b) the rename has to be reversed before the
    strategy's downstream code (``_unwrap_pyt_sharded_tensor`` /
    ``_replace_sharded_keys_with_state_dict_keys``) walks the dict, since
    those routines expect the original keys.

    Returns:
        ``{shadow_fqn: original_fqn}`` for every key that was rerouted; an
        empty dict if no shadow keys are present (e.g. when loading a
        checkpoint saved without local-replica mode).
    """
    renames: Dict[str, str] = {}
    sd_md = getattr(metadata, "state_dict_metadata", None)
    if sd_md is None:
        return renames
    for fqn in list(pyt_state_dict.keys()):
        sk = shadow_key(global_rank, fqn)
        if sk in sd_md:
            pyt_state_dict[sk] = pyt_state_dict.pop(fqn)
            renames[sk] = fqn
    return renames


def restore_pyt_state_dict_from_shadows(
    pyt_state_dict: Dict[str, object], renames: Dict[str, str]
) -> None:
    """Reverse the rename produced by :func:`redirect_pyt_state_dict_to_shadows`.

    After ``checkpoint.load`` has populated ``pyt_state_dict[shadow_fqn]``
    with the loaded tensor data, swap the key back to ``original_fqn`` so
    the rest of ``TorchDistLoadShardedStrategy.load`` (which walks the
    dict by original keys) sees the data under the user's expected FQN.
    """
    for shadow_fqn, original_fqn in renames.items():
        pyt_state_dict[original_fqn] = pyt_state_dict.pop(shadow_fqn)


def filter_non_shadow_keys(
    state_dict_metadata: Dict[str, object],
) -> Dict[str, object]:
    """Return a copy of ``state_dict_metadata`` with shadow entries removed.

    Used by :class:`MCoreSavePlanner` to bypass the default
    "chunks must cover the global tensor" volume check on shadow entries —
    a check that does not apply because each shadow tensor only carries
    the local rank's chunks and its global shape is the original (full)
    shape.
    """
    return {k: v for k, v in state_dict_metadata.items() if not is_shadow_key(k)}
