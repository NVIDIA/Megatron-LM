# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .dynamic_context import DynamicInferenceContext


class MoERoutingReplayCache:
    """Per-block storage for MoE routing replay.

    Stores the expert-routing decisions made during a forward pass so they
    can be replayed bit-exactly when a paused-and-resumed request continues
    its prefill. The storage is host-only — a Python ``dict`` keyed by
    block ID, with ndarray values shaped
    ``[block_size_tokens, num_layers, topk]``. There is no GPU shadow.

    Owned by :class:`KVBlockAllocator` as ``allocator.routing_replay``,
    which is the only attachment point. The base allocator never imports
    or refers to routing-replay state directly: it calls
    :meth:`clear_for_reallocated` and :meth:`reset` at the appropriate
    points and is otherwise oblivious. Deployments that don't use routing
    replay leave the dict empty; the empty-dict branches in this class
    are O(1) no-ops.
    """

    def __init__(
        self,
        context: "DynamicInferenceContext",
        dummy_block_idx: int,
        block_size_tokens: int,
    ):
        # ``context`` is held only for per-step reads in
        # :meth:`store_routing_per_block` (active token count, token-to-block
        # / token-to-local-position mappings). The dict-keyed routing storage
        # is fully owned here.
        self.context = context
        self.dummy_block_idx = dummy_block_idx
        self.block_size_tokens = block_size_tokens

        self.block_routing: Dict[int, np.ndarray] = {}

    def __bool__(self) -> bool:
        """Truthy when at least one block has stored routing data."""
        return bool(self.block_routing)

    def reset(self) -> None:
        """Clear all stored routing data."""
        self.block_routing.clear()

    # =========================================================================
    # Re-allocation cleanup
    # =========================================================================

    def clear_for_reallocated(self, block_ids: List[int]) -> None:
        """Drop routing data for blocks that were just popped from the free pool.

        Routing-replay correctness model: a re-allocated block's prior
        routing data is stale because the new request will overwrite only
        the positions it actually fills, leaving the tail with the previous
        owner's data. Clearing on re-allocate (rather than on release /
        deregister) lets a just-finished request still call
        :meth:`reconstruct_routing_from_blocks` over its now-released
        blocks before they are handed to the next owner.

        No-op when no blocks have stored routing — the empty-dict guard
        makes this safe to call from the allocator's hot allocate path
        regardless of whether routing replay is in use.
        """
        if not self.block_routing:
            return
        for bid in block_ids:
            self.block_routing.pop(bid, None)

    # =========================================================================
    # Bulk store / reconstruct
    # =========================================================================

    def store_routing_per_block(self, flat_routing: Optional[np.ndarray]) -> None:
        """Scatter flat routing indices into per-block storage.

        Uses the context's token-to-block mapping to distribute each
        token's routing data into the appropriate block. Matched
        (prefix-cached) blocks already have routing from the original
        request and are not overwritten here since their tokens are not
        in the active token layout.

        Args:
            flat_routing: ndarray of shape ``[active_token_count, num_layers, topk]``
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

        # Token-to-block mapping for all active tokens.
        block_ids_np = context.token_to_block_idx[:token_count].cpu().numpy()
        positions_np = context.token_to_local_position_within_kv_block[:token_count].cpu().numpy()

        dummy = self.dummy_block_idx

        # Group tokens by block_id using sort for efficient scatter.
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
        self, block_ids: List[int], total_routing_tokens: int
    ) -> Optional[np.ndarray]:
        """Reconstruct routing indices from per-block storage.

        Concatenates per-block routing ndarrays in block order, trimming
        the last block to exactly ``total_routing_tokens`` entries.

        Args:
            block_ids: Ordered list of block IDs for the request.
            total_routing_tokens: Expected number of routing tokens
                (``total_tokens - 1``, since the last generated token has
                no forward-pass routing).

        Returns:
            ndarray ``[total_routing_tokens, num_layers, topk]`` or
            ``None`` if any block is missing routing data.
        """
        block_size = self.block_size_tokens
        routing_parts = []
        tokens_collected = 0

        for bid in block_ids:
            routing = self.get_block_routing(bid)
            if routing is None:
                return None  # Missing routing data for this block.
            remaining = total_routing_tokens - tokens_collected
            if remaining <= 0:
                break
            take = min(block_size, remaining)
            routing_parts.append(routing[:take])
            tokens_collected += take

        if not routing_parts or tokens_collected != total_routing_tokens:
            return None

        return np.concatenate(routing_parts, axis=0)

    # =========================================================================
    # Per-block primitives
    # =========================================================================

    def store_block_routing(
        self, block_id: int, positions: np.ndarray, routing: np.ndarray
    ) -> None:
        """Store routing indices for specific token positions in a block.

        Args:
            block_id: The block ID.
            positions: ndarray of token positions within the block (1D, int).
            routing: ndarray of routing data ``[num_positions, num_layers, topk]``.
        """
        if block_id not in self.block_routing:
            self.block_routing[block_id] = np.zeros(
                (self.block_size_tokens, routing.shape[-2], routing.shape[-1]),
                dtype=routing.dtype,
            )
        self.block_routing[block_id][positions] = routing

    def get_block_routing(self, block_id: int) -> Optional[np.ndarray]:
        """Get routing indices for a block.

        Args:
            block_id: The block ID.

        Returns:
            ndarray ``[block_size_tokens, num_layers, topk]`` or ``None``
            if not stored.
        """
        return self.block_routing.get(block_id)
