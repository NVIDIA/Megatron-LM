# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""High-level orchestration for async checkpoint loading.

:class:`AsyncCheckpointLoader` combines the async-load entrypoints from
``serialization`` with a :class:`~megatron.core.dist_checkpointing.cpu_shadow.ShadowBufferPool`
and a per-topology plan cache, so repeated loads of same-topology checkpoints
only pay for disk reads.
"""

from __future__ import annotations

import enum
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .cpu_shadow import ShadowBufferPool

logger = logging.getLogger(__name__)

# Same tracker filename as megatron.training.checkpointing (not imported here:
# megatron.core must not depend on megatron.training).
_CHECKPOINT_TRACKER_FILENAME = 'latest_checkpointed_iteration.txt'


def resolve_checkpoint_iter_dir(load_dir: str) -> str:
    """Resolve a Megatron load dir to its latest ``iter_*`` / ``release`` subdir.

    Args:
        load_dir (str): base checkpoint directory (or an iter dir, returned as-is).

    Returns: path of the checkpoint directory for the latest iteration.
    """
    tracker = Path(load_dir) / _CHECKPOINT_TRACKER_FILENAME
    if not tracker.is_file():
        if Path(load_dir).name == "release" or Path(load_dir).name.startswith("iter_"):
            return load_dir
        raise FileNotFoundError(f"No tracker file at {tracker} and {load_dir} is not an iter dir")
    token = tracker.read_text().strip()
    if token == "release":
        return str(Path(load_dir) / "release")
    if not token.isdigit():
        raise ValueError(f"Unexpected tracker contents at {tracker}: {token!r}")
    return str(Path(load_dir) / f"iter_{int(token):07d}")


def compute_checkpoint_save_topology_fingerprint(checkpoint_dir: str) -> str:
    """sha256 of the per-FQN chunk layout in ``checkpoint_dir/.metadata``.

    Checkpoints saved with the same parallel topology produce the same
    fingerprint and can share one async load plan.
    """
    from .strategies.cached_metadata_filesystem_reader import CachedMetadataFileSystemReader

    metadata = CachedMetadataFileSystemReader(checkpoint_dir).read_metadata()
    h = hashlib.sha256()
    for fqn in sorted(metadata.state_dict_metadata.keys()):
        entry = metadata.state_dict_metadata[fqn]
        h.update(fqn.encode("utf-8"))
        h.update(b"|" + type(entry).__name__.encode("utf-8") + b"|")
        chunks = getattr(entry, "chunks", None)
        if chunks is None:
            continue
        size = getattr(entry, "size", None)
        if size is not None:
            h.update(repr(tuple(size)).encode("utf-8"))
        props = getattr(entry, "properties", None)
        if props is not None and getattr(props, "dtype", None) is not None:
            h.update(repr(props.dtype).encode("utf-8"))
        # DCP doesn't guarantee chunk order; sort so the hash is reproducible.
        h.update(repr(sorted((tuple(c.offsets), tuple(c.sizes)) for c in chunks)).encode("utf-8"))
    return h.hexdigest()


class TopologyPlanCache:
    """One ``AsyncLoadPlan`` per distinct save-time topology fingerprint.

    Sharing a template plan across checkpoints with mismatched save-time
    layouts would read wrong file bytes, so plans are keyed by
    :func:`compute_checkpoint_save_topology_fingerprint`.
    """

    def __init__(self):
        self._plans: dict[str, Any] = {}
        self._dir_to_fingerprint: dict[str, str] = {}
        self._n_hits = 0
        self._n_misses = 0

    def __len__(self) -> int:
        return len(self._plans)

    def get_or_build(self, checkpoint_iter_dir: str, pool: ShadowBufferPool):
        """Return the cached plan for this checkpoint's topology, building it
        (a collective) on first sight of a new fingerprint.

        Returns: tuple of (plan, fingerprint).
        """
        fingerprint = self._dir_to_fingerprint.get(checkpoint_iter_dir)
        if fingerprint is None:
            fingerprint = compute_checkpoint_save_topology_fingerprint(checkpoint_iter_dir)
            self._dir_to_fingerprint[checkpoint_iter_dir] = fingerprint
        cached = self._plans.get(fingerprint)
        if cached is not None:
            self._n_hits += 1
            return cached, fingerprint

        self._n_misses += 1
        tmpl_shadow = pool.acquire()
        try:
            # Call-time import to avoid a circular import with the package init.
            from megatron.core import dist_checkpointing

            plan = dist_checkpointing.prepare_async_load(tmpl_shadow, checkpoint_iter_dir)
        finally:
            pool.release(tmpl_shadow)
        self._plans[fingerprint] = plan
        return plan, fingerprint

    def stats(self) -> dict[str, int]:
        """Cache statistics for logging."""
        return {
            "distinct_plans": len(self._plans),
            "distinct_checkpoints": len(self._dir_to_fingerprint),
            "hits": self._n_hits,
            "misses": self._n_misses,
        }


class LoadState(enum.Enum):
    """Lifecycle state of an :class:`AsyncLoadHandle`."""

    KICKED = "kicked"
    FINALIZED = "finalized"
    RELEASED = "released"


@dataclass
class AsyncLoadHandle:
    """One in-flight (or finalized) async load.

    The handle owns the leased shadow from ``kick`` until ``release``; the
    finalized tree returned by ``poll``/``finalize`` aliases that shadow's
    pinned storage and is only valid until ``release``.
    """

    key: str
    request: Any  # AsyncLoadRequest
    shadow: dict
    call_idx: int
    state: LoadState = LoadState.KICKED


class AsyncCheckpointLoader:
    """Stream torch_dist checkpoints into pinned-CPU shadows asynchronously.

    Bound to one ``model`` for its lifetime (the shadow shape depends on it).
    Loads are identified by an opaque ``key``; the loader owns the shadow pool,
    the topology plan cache and the monotonic ``call_idx``. What happens with
    the finalized tensors is the caller's concern.
    """

    def __init__(self, model, *, num_shadow_buffers: int = 1):
        self._model = model
        self._pool = ShadowBufferPool(model, num_buffers=num_shadow_buffers)
        self._plan_cache = TopologyPlanCache()
        self._dir_cache: dict[str, str] = {}
        self._call_idx = 0

    @property
    def call_idx(self) -> int:
        """Monotonic sequence number of the last kicked load."""
        return self._call_idx

    def num_free_shadows(self) -> int:
        """Number of currently free shadow buffers."""
        return self._pool.num_free()

    def num_shadow_buffers(self) -> int:
        """Total number of shadow buffers."""
        return self._pool.num_buffers()

    def plan_cache_stats(self) -> dict[str, int]:
        """Statistics of the topology plan cache."""
        return self._plan_cache.stats()

    def resolve_iter_dir(self, load_dir: str) -> str:
        """Resolve and cache ``load_dir``'s latest iteration directory.

        The result is cached: a tracker file updated later (e.g. a new
        iteration saved into the same load_dir) is not re-read.
        """
        iter_dir = self._dir_cache.get(load_dir)
        if iter_dir is None:
            iter_dir = resolve_checkpoint_iter_dir(load_dir)
            self._dir_cache[load_dir] = iter_dir
        return iter_dir

    @torch.no_grad()
    def load_finalized_to_model(self, finalized: dict, *, strict: bool = False) -> None:
        """Load a finalized state dict directly into the bound model chunks.

        ``finalized`` is keyed like Megatron checkpoints: ``finalized['model']``
        for a single chunk, ``finalized[f'model{i}']`` per chunk with virtual
        pipeline parallelism. With ``strict=False``, entries absent from the
        checkpoint (e.g. TE ``extra_state``) keep their current values.
        """
        if len(self._model) == 1:
            self._model[0].load_state_dict(finalized["model"], strict=strict)
        else:
            for i, model_i in enumerate(self._model):
                key = f"model{i}"
                if key in finalized:
                    model_i.load_state_dict(finalized[key], strict=strict)

    def prewarm_plans(self, load_dirs: Iterable[str]) -> None:
        """Build every topology plan up front while the pool is idle.

        Every rank must call this with the same ``load_dirs`` in the same
        order (plan building is a collective).
        """
        for load_dir in load_dirs:
            iter_dir = self.resolve_iter_dir(load_dir)
            self._plan_cache.get_or_build(iter_dir, self._pool)

    def kick(self, key: str, load_dir: str) -> AsyncLoadHandle:
        """Start an async load of ``load_dir`` into a leased shadow.

        Collective ordered operation: every rank must kick the same ``key`` in
        the same order so ``call_idx`` stays in lockstep. Raises if no shadow
        buffer is free.
        """
        iter_dir = self.resolve_iter_dir(load_dir)
        # Plan build may itself acquire+release a shadow, so it must happen
        # before the acquire below — nesting would self-deadlock at
        # num_buffers=1.
        plan, _ = self._plan_cache.get_or_build(iter_dir, self._pool)

        from megatron.core import dist_checkpointing

        shadow = self._pool.acquire()
        try:
            prepared_plan = dist_checkpointing.prepare_async_load_reusing_topology(
                shadow, iter_dir, plan
            )
        except BaseException:
            self._pool.release(shadow)
            raise

        self._call_idx += 1
        call_idx = self._call_idx
        request = dist_checkpointing.start_async_load_from_plan(prepared_plan, call_idx=call_idx)
        return AsyncLoadHandle(key=key, request=request, shadow=shadow, call_idx=call_idx)

    def poll(self, handle: AsyncLoadHandle):
        """Non-blocking collective poll.

        Returns: the finalized tree (aliasing the handle's shadow storage,
            valid until ``release``) when every rank's read finished, else None.
        """
        if handle.state is not LoadState.KICKED:
            raise RuntimeError(f"poll() on handle in state {handle.state}; expected KICKED.")
        finalized = handle.request.maybe_finalize(blocking=False)
        if finalized is None:
            return None
        handle.state = LoadState.FINALIZED
        return finalized

    def finalize(self, handle: AsyncLoadHandle):
        """Blocking collective finalize.

        Returns: the finalized tree (aliasing the handle's shadow storage,
            valid until ``release``).
        """
        if handle.state is not LoadState.KICKED:
            raise RuntimeError(f"finalize() on handle in state {handle.state}; expected KICKED.")
        finalized = handle.request.maybe_finalize(blocking=True)
        handle.state = LoadState.FINALIZED
        return finalized

    def release(self, handle: AsyncLoadHandle) -> None:
        """Return the handle's shadow to the pool.

        Call only after the finalized tree has been fully consumed — it
        aliases the shadow's pinned storage.
        """
        if handle.state is not LoadState.FINALIZED:
            raise RuntimeError(
                f"release() on handle in state {handle.state}; finalize/poll the load "
                "before releasing (the finalized tree aliases the shadow's pinned storage)."
            )
        self._pool.release(handle.shadow)
        handle.state = LoadState.RELEASED
