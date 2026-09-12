# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import heapq
from collections import deque
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy

# Block deregistration observers are currently registered only by DynamoHelper.
BlocksDeregisteredObserver = Callable[[list[int], set[int]], None]

# Selected prompt scores plus top-N float32 values and int32 token IDs consume
# roughly ``(1 + 2 * N) * 4`` bytes per token. A limit of 100 keeps a 256-token
# sidecar below 1 MiB while covering practical API requests.
MAX_CACHED_PROMPT_TOP_N_LOGPROBS = 100


class PromptLogprobsKey(NamedTuple):
    """Exact semantic identity for reusable prompt log probabilities.

    Raw log probabilities do not depend on the sampling backend or sampling
    filters, so those fields are normalized to ``None`` in raw mode. Processed
    log probabilities include the exact backend and normalized request sampling
    values. ``top_n`` is always exact: a sidecar created for one value of N is
    never reused for another.
    """

    mode: str
    top_n: int
    sampling_backend: Optional[str]
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]

    @classmethod
    def create(
        cls,
        mode: str,
        top_n: int,
        sampling_backend: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> "PromptLogprobsKey":
        """Create a normalized key from model and request configuration."""
        if mode not in ("raw_logprobs", "processed_logprobs"):
            raise ValueError(f"unsupported logprobs mode: {mode}")
        if isinstance(top_n, bool) or int(top_n) != top_n or top_n < 0:
            raise ValueError("top_n must be a non-negative integer")
        if top_n > MAX_CACHED_PROMPT_TOP_N_LOGPROBS:
            raise ValueError(
                f"cached prompt top-N must not exceed {MAX_CACHED_PROMPT_TOP_N_LOGPROBS}"
            )

        if mode == "raw_logprobs":
            return cls(mode, int(top_n), None, None, None, None)

        if not sampling_backend:
            raise ValueError("processed_logprobs requires a sampling backend identity")
        normalized_temperature = 1.0 if temperature is None else float(temperature)
        normalized_top_k = 0 if top_k is None else int(top_k)
        normalized_top_p = 0.0 if top_p is None else float(top_p)
        # Canonicalize negative zero without otherwise rounding exact values.
        normalized_temperature += 0.0
        normalized_top_p += 0.0
        return cls(
            mode,
            int(top_n),
            str(sampling_backend),
            normalized_temperature,
            normalized_top_k,
            normalized_top_p,
        )


class PromptLogprobsBlock:
    """Mutable prompt-logprob sidecar bound to one physical KV block.

    Allocator mappings retain only the latest settings variant. A request may
    keep a strong reference to an older object until its result is materialized.
    """

    __slots__ = (
        "logical_block_index",
        "key",
        "block_id",
        "selected_logprobs",
        "top_n_logprobs",
        "top_n_token_ids",
        "valid",
    )

    def __init__(
        self, block_size: int, logical_block_index: int, key: PromptLogprobsKey, block_id: int
    ) -> None:
        self.block_id = block_id
        self.logical_block_index = logical_block_index
        self.key = key
        self.selected_logprobs = np.empty((block_size,), dtype=np.float32)
        self.top_n_logprobs = np.empty((block_size, key.top_n), dtype=np.float32)
        self.top_n_token_ids = np.empty((block_size, key.top_n), dtype=np.int32)
        self.valid = np.zeros((block_size,), dtype=np.bool_)

    def matches(self, logical_block_index: int, key: PromptLogprobsKey) -> bool:
        """Whether newly stored rows belong to this same semantic sidecar."""
        return self.logical_block_index == logical_block_index and self.key == key

    def store(
        self,
        target_positions: np.ndarray,
        selected_logprobs: np.ndarray,
        top_n_logprobs: Optional[np.ndarray],
        top_n_token_ids: Optional[np.ndarray],
    ) -> None:
        """Fill rows indexed by target-token position without replacing cached scores."""
        positions = np.asarray(target_positions)
        if positions.ndim != 1:
            raise ValueError("target_positions must be one-dimensional")
        if not np.issubdtype(positions.dtype, np.integer):
            raise ValueError("target_positions must contain integers")
        positions = positions.astype(np.int32, copy=False)
        if positions.size and (
            int(positions.min()) < 0 or int(positions.max()) >= self.valid.shape[0]
        ):
            raise ValueError("target position is outside the KV block")
        if np.unique(positions).size != positions.size:
            raise ValueError("target_positions must not contain duplicates")

        selected = np.asarray(selected_logprobs)
        if selected.shape != (positions.size,):
            raise ValueError(
                f"selected_logprobs shape {selected.shape} does not match "
                f"{positions.size} target positions"
            )
        selected = selected.astype(np.float32, copy=False)

        if self.key.top_n == 0:
            if top_n_logprobs is not None and np.asarray(top_n_logprobs).size:
                raise ValueError("top_n_logprobs must be empty when top_n is zero")
            if top_n_token_ids is not None and np.asarray(top_n_token_ids).size:
                raise ValueError("top_n_token_ids must be empty when top_n is zero")
            top_values = np.empty((positions.size, 0), dtype=np.float32)
            top_ids = np.empty((positions.size, 0), dtype=np.int32)
        else:
            if top_n_logprobs is None or top_n_token_ids is None:
                raise ValueError("top-N values and token IDs are required when top_n is nonzero")
            top_values_array = np.asarray(top_n_logprobs)
            top_ids_array = np.asarray(top_n_token_ids)
            expected_shape = (positions.size, self.key.top_n)
            if top_values_array.shape != expected_shape or top_ids_array.shape != expected_shape:
                raise ValueError(
                    "top-N arrays must both have shape "
                    f"{expected_shape}, got {top_values_array.shape} and {top_ids_array.shape}"
                )
            if top_ids_array.size and (
                int(top_ids_array.min()) < np.iinfo(np.int32).min
                or int(top_ids_array.max()) > np.iinfo(np.int32).max
            ):
                raise ValueError("top-N token ID does not fit in int32")
            top_values = top_values_array.astype(np.float32, copy=False)
            top_ids = top_ids_array.astype(np.int32, copy=False)

        # A hybrid model can replay the final matched block after restoring the
        # preceding Mamba state so it can compute the first uncached score. Keep
        # the original scores for replayed positions: small numeric differences do
        # not make cached scores stale. Newly missing rows remain mutable.
        missing = ~self.valid[positions]
        missing_positions = positions[missing]
        self.selected_logprobs[missing_positions] = selected[missing]
        self.top_n_logprobs[missing_positions] = top_values[missing]
        self.top_n_token_ids[missing_positions] = top_ids[missing]
        self.valid[missing_positions] = True

    def has_rows(self, required_positions: np.ndarray) -> bool:
        """Whether every requested target-token position has been stored."""
        required = np.asarray(required_positions)
        if required.ndim != 1 or not np.issubdtype(required.dtype, np.integer):
            raise ValueError("required_positions must be a one-dimensional integer array")
        required = required.astype(np.int32, copy=False)
        if required.size and (
            int(required.min()) < 0 or int(required.max()) >= self.valid.shape[0]
        ):
            raise ValueError("required target position is outside the KV block")
        if np.unique(required).size != required.size:
            raise ValueError("required_positions must not contain duplicates")
        return bool(self.valid[required].all())

    def extract(self, required_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy requested rows in caller-provided target-token order."""
        required = np.asarray(required_positions, dtype=np.int32)
        if not self.has_rows(required):
            missing = required[~self.valid[required]]
            raise ValueError(
                f"prompt-logprob sidecar is missing target positions {missing.tolist()}"
            )
        return (
            np.array(self.selected_logprobs[required], dtype=np.float32, copy=True),
            np.array(self.top_n_logprobs[required], dtype=np.float32, copy=True),
            np.array(self.top_n_token_ids[required], dtype=np.int32, copy=True),
        )


class KVBlockAllocator:
    """Allocator that manages blocks of memory for the KV cache.

    This allocator is responsible for:
    - Initializing a pool of block IDs
    - Allocating blocks from the pool
    - Releasing blocks back to the pool

    Args:
        context (DynamicInferenceContext): Dynamic inference context.
        pool_size (int): Number of blocks in the pool, including the dummy block.
        paused_limit (int): Paused-request block retention limit. Must leave at
            least one non-dummy block outside the limit.
    """

    def __init__(
        self,
        context: "DynamicInferenceContext",
        pool_size: int,
        paused_limit: int,
        enable_prefix_caching: bool = False,
        prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
            PrefixCachingEvictionPolicy.REF_ZERO
        ),
    ):

        self.context = context
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_caching_eviction_policy = prefix_caching_eviction_policy
        self.on_blocks_deregistered: Optional[Callable] = None
        self._blocks_deregistered_observers: list[BlocksDeregisteredObserver] = []

        # Handoff blocks remain pinned until decode finishes pulling them.
        # Pinning at request finish only happens on engines with KV transfer
        # configured (setup_kv_transfer flips this on); other engines have no
        # release path for the pins.
        self.enable_handoff_pinning = False

        assert (
            0 <= paused_limit <= pool_size - 2
        ), "paused block limit must leave at least one usable block outside the limit"

        self.pool_size = pool_size
        self.pool_avail = pool_size - 1  # Raw free-pool count; -1 for dummy_block_idx.
        self.paused_limit = paused_limit
        self.dummy_block_idx = self.pool_size - 1

        # Initialize block pool as a "stack" data structure (CPU for bookkeeping).
        self.block_bag = torch.arange(self.pool_size, dtype=torch.int32, device='cpu')

        if self.enable_prefix_caching:
            # Block hash tracking for prefix caching: -1 = uncomputed, positive = valid hash
            self.block_hashes = torch.full((self.pool_size,), -1, dtype=torch.int64, device='cpu')

            # Hash-to-block mapping for O(1) prefix lookup
            self.kv_hash_to_block_id: Dict[int, int] = {}

            # Reference count per block: 0 = cached (evictable), >0 = actively used
            self.block_ref_counts = torch.zeros((self.pool_size,), dtype=torch.int32, device='cpu')

            # LRU timestamps for eviction ordering (higher = more recently used)
            # Only needed in LRU mode; RZ mode evicts immediately on ref_count==0
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps = torch.zeros(
                    (self.pool_size,), dtype=torch.int64, device='cpu'
                )

                # Persisted prefix-chain bookkeeping for LRU eviction, maintained
                # incrementally on register/deregister. Block hashes are
                # parent-chained: a cached block that is another cached block's
                # parent must not be evicted before its child (see evict_lru_blocks).
                #
                # block_parent_id[b] = block id of b's parent in the prefix chain,
                #   or -1 when b is a root block or its parent is not registered.
                self.block_parent_id = torch.full(
                    (self.pool_size,), -1, dtype=torch.int64, device='cpu'
                )
                # block_child_count[b] = number of currently-registered children of b.
                # For a cached block all of its children are cached too, so this
                # equals its cached-child count and b is an evictable leaf exactly
                # when it reaches 0.
                self.block_child_count = torch.zeros(
                    (self.pool_size,), dtype=torch.int64, device='cpu'
                )

        # Per-block MoE routing storage (populated when routing replay is enabled)
        self.block_routing: Dict[int, np.ndarray] = {}

        # Prompt-logprob sidecars share the lifetime and identity of physical KV
        # blocks. Request-owned state carries scores until their target block is
        # allocated, so this mapping contains only physically bound sidecars.
        self.block_prompt_logprobs: Dict[int, PromptLogprobsBlock] = {}

    def __str__(self):
        return (
            f"blocks: occupied {self.get_total_used()}/{self.pool_size - 1}"
            f"; allocatable {self.get_allocatable_count()}"
            f"; active-used {self.get_active_used()}"
            f"; paused-used {self.get_paused_used()}/{self.paused_limit}"
        )

    def get_total_used(self):
        """Compute number of physical blocks outside the free pool."""
        return self.pool_size - self.pool_avail - 1

    def get_active_used(self):
        """Compute number of active blocks used."""
        if not self.enable_prefix_caching:
            return (
                self.context.request_kv_block_counts[
                    self.context.paused_request_count : self.context.total_request_count
                ]
                .sum()
                .item()
            )

        active_start = self.context.paused_request_count
        active_end = self.context.total_request_count
        if active_end > active_start:
            active_rows = self.context.request_to_kv_block_ids[active_start:active_end]
            valid_ids = active_rows[active_rows >= 0]
            if valid_ids.numel() > 0:
                return int(torch.unique(valid_ids).numel())
        return 0

    def get_paused_used(self):
        """Compute number of paused blocks used."""
        if not self.enable_prefix_caching:
            return (
                self.context.request_kv_block_counts[: self.context.paused_request_count]
                .sum()
                .item()
            )

        if self.context.paused_request_count > 0:
            paused_rows = self.context.request_to_kv_block_ids[: self.context.paused_request_count]
            valid_ids = paused_rows[paused_rows >= 0]
            if valid_ids.numel() > 0:
                return int(torch.unique(valid_ids).numel())
        return 0

    def is_memory_available(self, num_blocks: int, potential_matched_count: int = 0) -> bool:
        """Check if memory blocks are available.

        Includes both free pool blocks and registered, evictable cached blocks.

        Args:
            num_blocks (int): Number of blocks to check.
            potential_matched_count (int): Number of currently-evictable cached
                blocks to subtract from the evictable count because the caller
                will pin them before allocating (e.g. prefix-matched blocks that
                get their ref counts bumped in add_request). These blocks are
                ref_count == 0 now, so they are included in the evictable count,
                but they will be protected from eviction, so they cannot supply
                the requested ``num_blocks``.

        Return:
            (bool) Is memory available?
        """
        # Fast path: avoid computing the evictable count when the free pool
        # suffices. Soon-to-be-pinned matches do not affect raw free capacity.
        if self.pool_avail >= num_blocks:
            return True
        return self.get_allocatable_count() - potential_matched_count >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Will attempt LRU eviction of cached blocks if the free pool is insufficient.

        Args:
            num_blocks (int): Number of blocks to allocate.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        # Try to evict cached blocks if free pool is insufficient
        if self.pool_avail < num_blocks:
            if (
                not self.enable_prefix_caching
                or self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO
            ):
                return None  # RZ: no eviction path; disabled: no cached blocks
            blocks_needed_from_eviction = num_blocks - self.pool_avail
            if not self.evict_lru_blocks(blocks_needed_from_eviction):
                return None  # Not enough blocks even after eviction

        # Now allocate from the free pool
        self.pool_avail -= num_blocks
        block_ids = self.block_bag[self.pool_avail : (self.pool_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        if self.enable_prefix_caching:
            # Initialize ref counts for newly allocated blocks
            self.block_ref_counts[block_ids] = 1
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.update_timestamps(block_ids)

        # Clear stale per-block data for re-allocated blocks.
        for bid in block_ids.tolist():
            self.block_routing.pop(bid, None)
            self.block_prompt_logprobs.pop(bid, None)

        return block_ids

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks by decrementing reference counts.

        Blocks with ref_count == 0 remain cached (in hash map) for potential reuse.
        They will be evicted via LRU when space is needed.

        Args:
            blocks (Tensor): Block IDs to release.

        Return:
            None
        """
        if blocks.numel() == 0:
            return

        if self.enable_prefix_caching:
            unique_blocks, release_counts = torch.unique(blocks, return_counts=True)
            remaining_ref_counts = self.block_ref_counts[unique_blocks] - release_counts.to(
                dtype=self.block_ref_counts.dtype
            )
            assert torch.all(
                remaining_ref_counts >= 0
            ), "released more KV block references than the allocator owns"
            self.block_ref_counts[unique_blocks] = remaining_ref_counts
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
                zero_mask = remaining_ref_counts == 0
                if zero_mask.any():
                    self._deregister_blocks(unique_blocks[zero_mask])
            elif self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                # Unregistered blocks (hash == -1, ref_count == 0) have no hash
                # entry to preserve for reuse (e.g., partial blocks at the end of
                # a request). Return them directly to the free pool so they are not
                # leaked.
                unreg_mask = (remaining_ref_counts == 0) & (self.block_hashes[unique_blocks] == -1)
                if unreg_mask.any():
                    unreg_blocks = unique_blocks[unreg_mask]
                    num_unreg = unreg_blocks.numel()
                    for bid in unreg_blocks.tolist():
                        self.block_prompt_logprobs.pop(bid, None)
                    self.block_bag[self.pool_avail : self.pool_avail + num_unreg] = unreg_blocks
                    self.pool_avail += num_unreg
        else:
            num_blocks = blocks.numel()
            for bid in blocks.tolist():
                self.block_prompt_logprobs.pop(bid, None)
            self.block_bag[self.pool_avail : self.pool_avail + num_blocks] = blocks
            self.pool_avail += num_blocks

    def retain_memory_blocks(self, block_ids: list[int]) -> None:
        """Add one prefix-cache reference to each block.

        Args:
            block_ids: Blocks retained by a new owner.
        """
        assert self.enable_prefix_caching, "retaining KV blocks requires prefix caching"
        if block_ids:
            blocks = torch.tensor(block_ids, dtype=torch.int32, device='cpu')
            unique_blocks, retain_counts = torch.unique(blocks, return_counts=True)
            self.block_ref_counts[unique_blocks] += retain_counts.to(
                dtype=self.block_ref_counts.dtype
            )
            self.update_timestamps(unique_blocks)

    def reset(self) -> None:
        """Reset the allocator to initial state.

        This resets the available block count to the entire memory pool
        (except for the dummy block).
        """

        # Reset block bag to so we start consuming from the beginning of the pool
        # for UVM performance.
        # *Note*: Resetting the block bag is essential because if engine has been
        # suspended, then the block bag contains non-unique IDs since the
        # right-most IDs have been 'popped' off and are owned by the context.
        # Without resetting the block bag, context request memory will clash and
        # requests will point to each other's memory blocks, resulting in faulty
        # generations.
        # Refill the existing buffer so it remains mutable when reset runs under
        # torch.inference_mode(), such as during CUDA graph setup.
        torch.arange(self.pool_size, out=self.block_bag)

        self.pool_avail = self.pool_size - 1

        if self.enable_prefix_caching:
            # Reset all block hashes
            self.block_hashes.fill_(-1)

            # Reset prefix caching state
            self.kv_hash_to_block_id.clear()
            self.block_ref_counts.fill_(0)
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.block_timestamps.fill_(0)
                self.block_parent_id.fill_(-1)
                self.block_child_count.fill_(0)

        # Clear per-block routing storage
        self.block_routing.clear()
        self.block_prompt_logprobs.clear()

    # =========================================================================
    # Prefix caching methods
    # =========================================================================

    def register_kv_block_hashes(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_hashes: Optional[list[int]] = None,
    ) -> None:
        """Register blocks in the hash-to-block mapping for discovery (batch).

        Registration is idempotent: a block that already carries the hash being
        registered is skipped. Callers may legitimately re-offer an already
        registered block (a cache-matched block whose block-table slot a later
        prefill chunk also spans), and the bookkeeping below is one-shot per
        block — applying it twice adds a second child entry to the block's
        parent that no deregistration can ever cancel, leaving that parent
        permanently short of ``child_count == 0`` and therefore never an
        evictable leaf (see ``evict_lru_blocks``).

        Re-registering a live block under a *different* hash would instead
        overwrite its recorded parent while leaving the previous parent's child
        count raised, so that case is rejected rather than absorbed.

        This method never touches reference counts. New blocks are pinned at
        ``ref_count == 1`` by ``allocate_memory_blocks``, and additional owners
        of an already registered block are pinned by the caller that matched it.

        Args:
            block_ids: List of block IDs.
            block_hashes: List of computed hash values (same length as block_ids).
            parent_hashes: Parent hash for each block in the prefix chain (same
                length as block_ids); 0 marks a root block with no parent. Used
                by LRU eviction to avoid evicting a parent before its children.
                If None, parents default to 0.
        """
        if not block_ids:
            return
        if parent_hashes is not None:
            assert len(parent_hashes) == len(block_ids)
        # Tensor views of the batch, used to index the per-block state arrays.
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=self.block_hashes.device)
        hash_tensor = torch.tensor(block_hashes, dtype=torch.int64, device=self.block_hashes.device)

        # Drop blocks that already carry this hash, and reject hash changes on a
        # block that is still registered. Read the stored hashes before writing
        # them below, so this sees each block's pre-call state.
        # Hash each block holds right now; -1 means it is not registered.
        current_hashes = self.block_hashes[id_tensor]
        # Per-entry: this exact (block, hash) pair is already registered -> skip it.
        already_registered = current_hashes == hash_tensor
        # Per-entry: block is registered, but under some other hash -> illegal.
        conflict_mask = (current_hashes != -1) & ~already_registered
        # Batch positions of the illegal entries, for the failure message.
        conflicting = torch.nonzero(conflict_mask, as_tuple=True)[0].tolist()
        assert not conflicting, "block re-registered under a different hash: " + ", ".join(
            f"block {block_ids[i]} holds {int(current_hashes[i])}, given {block_hashes[i]}"
            for i in conflicting
        )
        if already_registered.any():
            # Batch positions of the entries that still need registering. Every
            # list and tensor below is narrowed to these so that the writes, the
            # hash-map update and the child-count bumps all see the same subset.
            keep = torch.nonzero(~already_registered, as_tuple=True)[0]
            if keep.numel() == 0:
                return
            keep_list = keep.tolist()
            block_ids = [block_ids[i] for i in keep_list]
            block_hashes = [block_hashes[i] for i in keep_list]
            if parent_hashes is not None:
                parent_hashes = [parent_hashes[i] for i in keep_list]
            id_tensor = id_tensor[keep]
            hash_tensor = hash_tensor[keep]

        self.block_hashes[id_tensor] = hash_tensor
        # Add the new blocks to the hash map first so that a block whose parent is
        # elsewhere in this same batch (block k's parent is block k-1) resolves.
        # Skipped blocks are already in the map, so they resolve as parents too.
        self.kv_hash_to_block_id.update(zip(block_hashes, block_ids))

        if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
            # Persist the resolved parent block id and bump each parent's child count.
            # Parents are earlier in the prefix chain and already registered
            # (a matched block or a prior chunk / earlier entry in this batch),
            # so a valid parent hash resolves; 0 marks a root and an unknown hash
            # falls back to -1.
            if parent_hashes is None:
                parent_hashes = [0] * len(block_ids)
            # Parent hashes resolved to block ids, aligned with block_ids; -1 for
            # a root block and for a parent hash that is no longer cached.
            parent_ids = [
                self.kv_hash_to_block_id.get(ph, -1) if ph != 0 else -1 for ph in parent_hashes
            ]
            parent_id_tensor = torch.tensor(parent_ids, dtype=torch.int64, device=id_tensor.device)
            self.block_parent_id[id_tensor] = parent_id_tensor
            # Per-entry: this block has a resolved parent whose count to bump.
            has_parent = parent_id_tensor >= 0
            if has_parent.any():
                self.block_child_count.scatter_add_(
                    0,
                    parent_id_tensor[has_parent],
                    torch.ones(int(has_parent.sum()), dtype=torch.int64),
                )

    def add_blocks_deregistered_observer(self, observer: BlocksDeregisteredObserver) -> None:
        """Register a callback invoked when cached blocks are deregistered.

        Currently used only by DynamoHelper.
        """
        self._blocks_deregistered_observers.append(observer)

    def _deregister_blocks(self, block_ids: Tensor) -> None:
        """Remove blocks from prefix caching state and return to free pool.

        Shared cleanup logic for both LRU eviction and RZ proactive eviction.

        Args:
            block_ids: Tensor of block IDs to deregister.
        """
        num_blocks = block_ids.numel()
        if num_blocks == 0:
            return

        # Gather hashes via batched tensor indexing
        block_ids_i64 = block_ids.to(torch.int64)
        block_ids_list = block_ids.tolist()
        hashes = self.block_hashes[block_ids_i64].tolist()

        # Request-held PromptLogprobsBlock objects remain valid after these
        # allocator-owned mappings are removed.
        for bid in block_ids_list:
            self.block_prompt_logprobs.pop(bid, None)

        # Remove from kv_hash_to_block_id dict (set ops + C-level map, no Python loop)
        keys_to_delete = set(hashes) - {-1}
        deque(
            map(self.kv_hash_to_block_id.pop, keys_to_delete & self.kv_hash_to_block_id.keys()),
            maxlen=0,
        )

        # Reset block state (batched tensor ops)
        if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
            # Drop these blocks from their parents' child counts before clearing
            # their own bookkeeping, keeping block_child_count in sync so a parent
            # becomes an evictable leaf once its last child is deregistered.
            parent_ids = self.block_parent_id[block_ids_i64]
            has_parent = parent_ids >= 0
            if has_parent.any():
                self.block_child_count.scatter_add_(
                    0,
                    parent_ids[has_parent],
                    torch.full((int(has_parent.sum()),), -1, dtype=torch.int64),
                )
            self.block_parent_id[block_ids] = -1
            self.block_child_count[block_ids] = 0
            self.block_timestamps[block_ids] = 0
        self.block_hashes[block_ids] = -1
        self.block_ref_counts[block_ids] = 0

        # Return blocks to free pool
        self.block_bag[self.pool_avail : self.pool_avail + num_blocks] = block_ids
        self.pool_avail += num_blocks

        # Notify dependent allocators and external observers only after KV allocator
        # bookkeeping commits, so callback failures cannot leave this allocator partial.
        if self.on_blocks_deregistered is not None:
            self.on_blocks_deregistered(block_ids_list, keys_to_delete)
        for observer in tuple(self._blocks_deregistered_observers):
            observer(block_ids_list, keys_to_delete)

    def update_timestamps(self, block_ids: Tensor) -> None:
        """Update LRU timestamps for accessed blocks. No-op in RZ mode.

        Args:
            block_ids: Tensor of block IDs that were accessed.
        """
        if (
            self.prefix_caching_eviction_policy != PrefixCachingEvictionPolicy.LRU
            or block_ids.numel() == 0
        ):
            return
        self.block_timestamps[block_ids] = self.context.prefix_cache_lru_clock

    def get_evictable_block_count(self) -> Tensor:
        """Get count of cached blocks that can be evicted (ref_count == 0, hash set).

        Returns:
            Scalar tensor with the number of evictable cached blocks.
        """
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        return cached_mask.sum()

    def get_allocatable_count(self) -> int:
        """Compute the number of blocks available for allocation.

        Includes both blocks in the free pool and, under LRU prefix caching,
        registered ref-zero blocks that can be evicted.

        Returns:
            Number of blocks that can currently be allocated.
        """
        if (
            self.enable_prefix_caching
            and self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU
        ):
            return self.pool_avail + int(self.get_evictable_block_count())
        return self.pool_avail

    def evict_lru_blocks(self, num_blocks_needed: int) -> bool:
        """Evict LRU cached blocks to free up space in the pool.

        Evicts blocks with ref_count == 0, least-recently-used first, while never
        evicting a parent before its children. Block hashes are parent-chained,
        and ``_find_kv_match_count`` relies on the invariant that a cached child
        block always has all of its ancestors cached too. A naive oldest-first
        eviction breaks this: with chunked prefill, earlier chunks are allocated
        first (older timestamps) yet are ancestors of later chunks (newer
        timestamps), so once the request finishes and its blocks are cached, an
        ancestor can be older than its descendant and get evicted first, leaving a
        dangling child.

        To preserve the invariant while staying optimal we peel the cached forest
        from its leaves inward with a min-heap: only a leaf (a cached block with
        no cached children) is ever evictable, and among the currently-evictable
        leaves we always take the one with the oldest *own* timestamp. Evicting a
        leaf can turn its parent into a leaf, which is then pushed onto the heap.
        Repeating ``num_blocks_needed`` times gives, at each step, the globally
        least-recently-used block that can be removed without orphaning a child —
        the natural generalization of LRU to the parent-chain constraint. Keying
        each block by its *own* recency (and only reconsidering a parent once its
        children are gone) is what makes this optimal: a block is retained purely
        because it is recently used, never because a hot descendant props it up,
        so a colder evictable block is always evicted before a hotter one.

        Worked example, evicting 3 from::

            A(ts 1) -> B(ts 2) -> C(ts 5)   (C, F are leaves under B)
                              +-> F(ts 3)
                    +-> D(ts 3) -> E(ts 5)   (E is a leaf under D)

        Leaf-peel evicts F(3), then C(5); B is now childless so it joins the
        leaves with its own ts=2 and is evicted next -> retains {A, D, E}, keeping
        the hottest block E(5) rather than the colder interior block B(2).

        Note: because a request holds a contiguous block prefix [0..k], any in-use
        (ref_count > 0) block keeps all of its ancestors in use too. Hence a cached
        (ref_count == 0) block can only have cached children, and considering the
        cached set alone is sufficient to avoid dangling children.

        The parent block id of each block and its live child count are maintained
        incrementally on register/deregister (``block_parent_id`` /
        ``block_child_count``), so this method reads the prefix forest directly
        rather than rebuilding it from hashes with a per-eviction sort. Only the
        inherently-sequential leaf peel below is per-element.

        The parent graph is assumed acyclic (a forest), which holds for any hashes
        produced by the prefix-chain builder; an assertion guards against a
        pathological hash collision wedging the peel.

        Args:
            num_blocks_needed: Number of blocks to evict.

        Returns:
            True if enough blocks were evicted, False otherwise.
        """
        # Find all cached blocks (ref_count == 0, hash != -1)
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]

        num_cached = cached_block_ids.numel()
        if num_cached < num_blocks_needed:
            return False  # Not enough cached blocks to evict
        if num_blocks_needed <= 0:
            return True

        ts = self.block_timestamps[cached_block_ids].tolist()
        bid = cached_block_ids.tolist()
        parent_global = self.block_parent_id[cached_block_ids].tolist()
        child_count = self.block_child_count[cached_block_ids].tolist()

        # Map a cached block's global id to its local index so the peel can find a
        # parent's slot to decrement. Parents that are not cached (root, or a
        # parent still in use) are absent and are simply treated as peel roots.
        global_to_local = {bid[i]: i for i in range(num_cached)}
        parent_local = [global_to_local.get(p, -1) for p in parent_global]

        # Min-heap of currently-evictable leaves keyed by (own timestamp, block
        # id). Block ids are unique, so the tie-break is total and deterministic.
        heap = [(ts[i], bid[i], i) for i in range(num_cached) if child_count[i] == 0]
        heapq.heapify(heap)

        evicted_local = []
        while heap and len(evicted_local) < num_blocks_needed:
            _, _, i = heapq.heappop(heap)
            evicted_local.append(i)
            p = parent_local[i]
            if p >= 0:
                child_count[p] -= 1
                if child_count[p] == 0:
                    heapq.heappush(heap, (ts[p], bid[p], p))

        # A forest is always fully peelable, so the heap always exposes enough
        # leaves to collect num_blocks_needed (guaranteed by the num_cached >=
        # num_blocks_needed check above). Falling short means the parent graph is
        # cyclic — only possible under a hash collision, which we treat as a bug.
        assert len(evicted_local) == num_blocks_needed, (
            f"leaf peel evicted {len(evicted_local)} of {num_blocks_needed} "
            f"requested from {num_cached} cached blocks; parent graph is not a "
            f"forest (likely a block-hash collision)"
        )

        blocks_to_evict = cached_block_ids[torch.tensor(evicted_local, dtype=torch.int64)]
        self._deregister_blocks(blocks_to_evict)

        return True

    # =========================================================================
    # Per-block prompt-logprob sidecars
    # =========================================================================

    def _validate_prompt_logprobs_block_id(self, block_id: int) -> int:
        """Normalize and validate a non-dummy physical block ID."""
        block_id = int(block_id)
        if block_id < 0 or block_id >= self.dummy_block_idx:
            raise ValueError(f"invalid prompt-logprob KV block ID: {block_id}")
        return block_id

    def _validate_prompt_logprobs_hash(
        self, block_id: int, expected_block_hash: Optional[int]
    ) -> None:
        """Reject attachment to a physical block registered for other content."""
        if not self.enable_prefix_caching or expected_block_hash is None:
            return
        actual_hash = int(self.block_hashes[block_id])
        if actual_hash != -1 and actual_hash != int(expected_block_hash):
            raise ValueError(
                f"KV block {block_id} has hash {actual_hash}, expected {expected_block_hash}"
            )

    def store_prompt_logprobs(
        self,
        logical_block_index: int,
        block_id: int,
        key: PromptLogprobsKey,
        target_positions: np.ndarray,
        selected_logprobs: np.ndarray,
        top_n_logprobs: Optional[np.ndarray] = None,
        top_n_token_ids: Optional[np.ndarray] = None,
        expected_block_hash: Optional[int] = None,
        block: Optional[PromptLogprobsBlock] = None,
    ) -> PromptLogprobsBlock:
        """Store prompt scores in a mutable sidecar on a physical KV block.

        Only the latest settings variant is discoverable. Existing strong
        references remain valid when a new variant replaces the mapping.
        """
        logical_block_index = int(logical_block_index)
        if logical_block_index < 0:
            raise ValueError("logical_block_index must be non-negative")
        block_id = self._validate_prompt_logprobs_block_id(block_id)
        self._validate_prompt_logprobs_hash(block_id, expected_block_hash)
        if block is not None:
            if block.block_id != block_id or not block.matches(logical_block_index, key):
                raise ValueError("prompt-logprob block reference does not match the store target")
            entry = block
        else:
            entry = self.block_prompt_logprobs.get(block_id)
        if entry is None or not entry.matches(logical_block_index, key):
            entry = PromptLogprobsBlock(
                self.context.block_size_tokens, logical_block_index, key, block_id
            )

        self.block_prompt_logprobs[block_id] = entry
        entry.store(target_positions, selected_logprobs, top_n_logprobs, top_n_token_ids)
        return entry

    def get_prompt_logprobs_block(
        self,
        block_id: int,
        key: PromptLogprobsKey,
        expected_block_hash: Optional[int],
        required_positions: Optional[np.ndarray] = None,
    ) -> Optional[PromptLogprobsBlock]:
        """Return a complete sidecar for the exact key and live KV hash."""
        block_id = self._validate_prompt_logprobs_block_id(block_id)
        if (
            not self.enable_prefix_caching
            or expected_block_hash is None
            or int(self.block_hashes[block_id]) != int(expected_block_hash)
        ):
            return None
        entry = self.block_prompt_logprobs.get(block_id)
        if entry is None or entry.key != key:
            return None
        if required_positions is None:
            required_positions = np.arange(
                1 if entry.logical_block_index == 0 else 0,
                self.context.block_size_tokens,
                dtype=np.int32,
            )
        if not entry.has_rows(required_positions):
            return None
        return entry

    def materialize_prompt_logprobs(
        self,
        block_refs: Sequence[PromptLogprobsBlock],
        key: PromptLogprobsKey,
        prompt_token_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct numeric prompt results in target-token order.

        A prompt of ``P`` tokens has ``P - 1`` prompt scores. Logical block zero
        therefore contributes local target positions ``1..B-1``; every later
        block contributes ``0..B-1``, trimmed at the final prompt token.
        """
        prompt_token_count = int(prompt_token_count)
        if prompt_token_count < 0:
            raise ValueError("prompt_token_count must be non-negative")
        expected_count = max(prompt_token_count - 1, 0)
        if expected_count == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0, key.top_n), dtype=np.float32),
                np.empty((0, key.top_n), dtype=np.int32),
            )

        block_size = self.context.block_size_tokens
        num_blocks = (prompt_token_count + block_size - 1) // block_size
        if len(block_refs) < num_blocks:
            raise ValueError(f"materialization needs {num_blocks} block refs")

        selected_parts = []
        top_values_parts = []
        top_ids_parts = []
        for logical_block_index in range(num_blocks):
            ref = block_refs[logical_block_index]
            if ref.key != key or ref.logical_block_index != logical_block_index:
                raise ValueError("prompt-logprob block reference has the wrong key or order")
            global_start = logical_block_index * block_size
            local_start = 1 if logical_block_index == 0 else 0
            local_stop = min(block_size, prompt_token_count - global_start)
            required = np.arange(local_start, max(local_start, local_stop), dtype=np.int32)
            if not required.size:
                continue

            selected, top_values, top_ids = ref.extract(required)
            selected_parts.append(selected)
            top_values_parts.append(top_values)
            top_ids_parts.append(top_ids)

        selected = np.concatenate(selected_parts).astype(np.float32, copy=False)
        top_values = np.concatenate(top_values_parts).astype(np.float32, copy=False)
        top_ids = np.concatenate(top_ids_parts).astype(np.int32, copy=False)
        if selected.shape != (expected_count,):
            raise ValueError(
                f"materialized {selected.size} prompt scores, expected {expected_count}"
            )
        return selected, top_values, top_ids

    # =========================================================================
    # Per-block routing storage methods (for MoE routing replay)
    # =========================================================================

    def store_routing_per_block(self, flat_routing: Optional[np.ndarray]) -> None:
        """Scatter flat routing indices into per-block storage.

        Uses the context's token-to-block mapping to distribute each token's
        routing data into the appropriate block. Matched (prefix-cached) blocks
        already have routing from the original request and are not overwritten
        here since their tokens are not in the active token layout.

        Args:
            flat_routing: ndarray of shape [active_token_count, num_layers, topk]
                aligned with the context's active-token layout, or None.
        """
        if flat_routing is None:
            return

        context = self.context
        token_count = context.active_token_count
        if token_count == 0:
            return

        assert (
            flat_routing.shape[0] == token_count
        ), f"Routing token count {flat_routing.shape[0]} != active token count {token_count}"

        # Token-to-block mapping for all active tokens
        block_ids_np = context.token_to_block_idx[:token_count].cpu().numpy()
        positions_np = context.token_to_local_position_within_kv_block[:token_count].cpu().numpy()

        dummy = self.dummy_block_idx

        # Group tokens by block_id using sort for efficient scatter
        unique_blocks, inverse, counts = np.unique(
            block_ids_np, return_inverse=True, return_counts=True
        )
        sorted_indices = np.argsort(inverse, kind='stable')
        sorted_positions = positions_np[sorted_indices]
        sorted_routing = flat_routing[sorted_indices]

        offset = 0
        for bid, count in zip(unique_blocks, counts):
            bid = int(bid)
            count = int(count)
            if bid == dummy:
                offset += count
                continue
            block_pos = sorted_positions[offset : offset + count]
            block_rout = sorted_routing[offset : offset + count]
            self.store_block_routing(bid, block_pos, block_rout)
            offset += count

    def reconstruct_routing_from_blocks(
        self, block_ids: list[int], total_routing_tokens: int
    ) -> Optional[np.ndarray]:
        """Reconstruct routing indices from per-block storage.

        Concatenates per-block routing ndarrays in block order, trimming the
        last block to exactly ``total_routing_tokens`` entries.

        Args:
            block_ids: Ordered list of block IDs for the request.
            total_routing_tokens: Expected number of routing tokens
                (total_tokens - 1, since the last generated token has no
                forward-pass routing).

        Returns:
            ndarray [total_routing_tokens, num_layers, topk] or None if any
            block is missing routing data.
        """
        block_size = self.context.block_size_tokens
        routing_parts = []
        tokens_collected = 0

        for bid in block_ids:
            routing = self.get_block_routing(bid)
            if routing is None:
                return None  # Missing routing data for this block
            remaining = total_routing_tokens - tokens_collected
            if remaining <= 0:
                break
            take = min(block_size, remaining)
            routing_parts.append(routing[:take])
            tokens_collected += take

        if not routing_parts or tokens_collected != total_routing_tokens:
            return None

        return np.concatenate(routing_parts, axis=0)

    def store_block_routing(
        self, block_id: int, positions: np.ndarray, routing: np.ndarray
    ) -> None:
        """Store routing indices for specific token positions in a block.

        Args:
            block_id: The block ID.
            positions: ndarray of token positions within the block (1D, int).
            routing: ndarray of routing data [num_positions, num_layers, topk].
        """
        if block_id not in self.block_routing:
            self.block_routing[block_id] = np.zeros(
                (self.context.block_size_tokens, routing.shape[-2], routing.shape[-1]),
                dtype=routing.dtype,
            )
        self.block_routing[block_id][positions] = routing

    def get_block_routing(self, block_id: int) -> Optional[np.ndarray]:
        """Get routing indices for a block.

        Args:
            block_id: The block ID.

        Returns:
            ndarray [block_size_tokens, num_layers, topk] or None if not stored.
        """
        return self.block_routing.get(block_id)
