# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, Iterable, List, Optional, Tuple


class PrefixCacheRegistry:
    """Pure-CPU prefix-cache hash registry.

    Owns the host-side ``hash -> block_id`` mappings for both KV cache
    blocks and (for hybrid models) Mamba state blocks, and exposes the
    match / register / evict primitives that drive prefix caching.

    The interface is deliberately torch-free: only Python ``int`` /
    ``list`` / ``set`` cross the boundary. That's what makes the registry
    transport-friendly — the same surface can be implemented in process,
    in a sidecar, or as an RPC stub on a separate machine.

    Physical resources (the GPU ``block_hashes`` shadow and ref counts on
    the KV allocator, the slot pool and state tensors on the Mamba
    allocator) stay where they are; their owners call into this registry
    at register / evict time to keep the host view consistent with their
    physical state.
    """

    def __init__(self) -> None:
        self.kv_hash_to_block_id: Dict[int, int] = {}
        self.mamba_hash_to_block_id: Dict[int, int] = {}

        self._on_mamba_evicted: Optional[Callable[[List[int]], None]] = None

    def set_mamba_evict_callback(self, cb: Callable[[List[int]], None]) -> None:
        """Wire a callback fired when Mamba dict entries are evicted.

        The callback receives the list of block IDs whose Mamba state
        should be released. Used by the Mamba allocator to free GPU
        slots; remote deployments leave this unset.
        """
        self._on_mamba_evicted = cb

    # =========================================================================
    # KV
    # =========================================================================

    def match_kv_prefix(self, hashes: List[int]) -> Tuple[List[int], int]:
        """Longest cached prefix of ``hashes``.

        Parent-chained hashing guarantees that if ``hashes[N]`` is cached,
        all earlier hashes are too. Backwards scan finds the longest
        cached prefix in one pass.

        Returns ``(matched_block_ids, parent_hash)`` — ``parent_hash`` is
        the hash of the last matched block, ``0`` if no matches.
        """
        if not hashes:
            return [], 0
        d = self.kv_hash_to_block_id
        for i in range(len(hashes) - 1, -1, -1):
            if hashes[i] in d:
                num = i + 1
                return [d[hashes[j]] for j in range(num)], hashes[num - 1]
        return [], 0

    def register_kv(self, block_ids: List[int], hashes: List[int]) -> None:
        """Bulk add ``(hash -> block_id)`` entries."""
        if not block_ids:
            return
        self.kv_hash_to_block_id.update(zip(hashes, block_ids))

    def evict_kv(self, hashes: Iterable[int]) -> set:
        """Drop entries whose hashes appear in ``hashes`` (filtering -1).

        Cascades to the Mamba dict: a deregistered KV block can never
        retain valid Mamba state. Returns the set of hashes that were
        actually removed from the KV dict.
        """
        keys = set(hashes) - {-1}
        if not keys:
            return keys
        present = keys & self.kv_hash_to_block_id.keys()
        for h in present:
            self.kv_hash_to_block_id.pop(h)
        self._cascade_mamba_evict(present)
        return present

    # =========================================================================
    # Mamba
    # =========================================================================

    def match_mamba_farthest(self, hashes: List[int]) -> int:
        """Index (one past) the farthest cached Mamba block in ``hashes``.

        Mamba state is cumulative — only the farthest cached block needs
        to be restored. Backwards scan finds it in one pass.
        Returns 0 if nothing matches.
        """
        if not hashes:
            return 0
        d = self.mamba_hash_to_block_id
        for i in range(len(hashes) - 1, -1, -1):
            if hashes[i] in d:
                return i + 1
        return 0

    def find_mamba_backoff(self, hashes: List[int], search_len: int) -> int:
        """Farthest cached Mamba block within ``hashes[:search_len]``.

        Used by ``add_request`` when the raw Mamba match overruns the
        chunk and we need to back off to a previous cached state. The
        normal ``match_mamba_farthest`` scans the whole list; this scans
        a bounded prefix.
        """
        if search_len <= 0:
            return 0
        d = self.mamba_hash_to_block_id
        for j in range(search_len - 1, -1, -1):
            if hashes[j] in d:
                return j + 1
        return 0

    def register_mamba(self, block_ids: List[int], hashes: List[int]) -> None:
        """Bulk add ``(hash -> block_id)`` entries; skips ``hash <= 0``."""
        updates = {h: bid for bid, h in zip(block_ids, hashes) if h > 0}
        if updates:
            self.mamba_hash_to_block_id.update(updates)

    def evict_mamba(self, hashes: Iterable[int]) -> set:
        """Drop entries whose hashes appear in ``hashes`` (filtering -1).

        Returns the set of hashes that were actually removed. Fires the
        Mamba-evict callback (with the corresponding block IDs) so the
        allocator can free GPU slots.
        """
        if not self.mamba_hash_to_block_id:
            return set()
        keys = set(hashes) - {-1}
        present = keys & self.mamba_hash_to_block_id.keys()
        if not present:
            return present
        block_ids = [self.mamba_hash_to_block_id.pop(h) for h in present]
        if self._on_mamba_evicted is not None:
            self._on_mamba_evicted(block_ids)
        return present

    def _cascade_mamba_evict(self, hashes: set) -> None:
        """Internal: KV eviction implies Mamba eviction for the same hashes."""
        if not self.mamba_hash_to_block_id:
            return
        present = hashes & self.mamba_hash_to_block_id.keys()
        if not present:
            return
        block_ids = [self.mamba_hash_to_block_id.pop(h) for h in present]
        if self._on_mamba_evicted is not None:
            self._on_mamba_evicted(block_ids)

    # =========================================================================
    # Reset
    # =========================================================================

    def clear_kv(self) -> None:
        """Drop all KV entries."""
        self.kv_hash_to_block_id.clear()

    def clear_mamba(self) -> None:
        """Drop all Mamba entries."""
        self.mamba_hash_to_block_id.clear()

    def reset(self) -> None:
        """Drop all entries (KV and Mamba)."""
        self.kv_hash_to_block_id.clear()
        self.mamba_hash_to_block_id.clear()
