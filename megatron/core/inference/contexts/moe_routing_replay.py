# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Optional

import numpy as np
import torch


class MoERoutingReplayCache:
    """Per-block storage for MoE routing replay.

    Stores the expert-routing decisions made during a forward pass so they can be replayed
    bit-exactly when a paused-and-resumed request continues its prefill.
    The storage is host-only: a Python `dict` keyed by block ID, with ndarray values shaped
    `[block_size_tokens, num_layers, topk]`.

    The class owns only the dict-keyed storage and has no reference to the inference context.
    `scatter_routing_to_blocks` (write) and `reconstruct_routing_from_blocks` (read) are the bulk
    operations; both take the per-step values they need (token-to-block mappings, block IDs)
    as explicit arguments rather than reading them off a stored context.

    Owned by `KVBlockAllocator` as `allocator.routing_replay`.
    """

    def __init__(self, block_size_tokens: int):
        self.block_size_tokens = block_size_tokens
        self.block_routing: Dict[int, np.ndarray] = {}

    def has_data(self) -> bool:
        """Return True when at least one block has stored routing data."""
        return bool(self.block_routing)

    def reset(self) -> None:
        """Clear all stored routing data."""
        self.block_routing.clear()

    # =========================================================================
    # Re-allocation cleanup
    # =========================================================================

    def clear_for_reallocated(self, block_ids: List[int]) -> None:
        """Drop routing data for blocks that were just popped from the free pool."""
        if not self.block_routing:
            return
        for bid in block_ids:
            self.block_routing.pop(bid, None)

    # =========================================================================
    # Bulk store / reconstruct
    # =========================================================================

    def scatter_routing_to_blocks(
        self,
        flat_routing: Optional[np.ndarray],
        *,
        active_token_count: int,
        token_to_block_idx: torch.Tensor,
        token_to_local_position: torch.Tensor,
        dummy_block_idx: int,
    ) -> None:
        """Scatter flat routing indices into per-block storage.

        Uses the caller-supplied token-to-block mapping to distribute each token's routing data
        into the appropriate block. Prefix-cached blocks already have routing from the original
        request and are not overwritten here.

        Args:
            flat_routing: ndarray of shape `[active_token_count, num_layers, topk]`
                aligned with the context's active-token layout, or `None`.
            active_token_count: Number of active tokens this step. Slices token-to-block tensors.
            token_to_block_idx: 1-D tensor mapping each active token to the block ID it belongs to.
            token_to_local_position: 1-D tensor mapping each active token to its block position.
            dummy_block_idx: Block ID used for padded / dummy tokens.
        """
        if flat_routing is None:
            return

        if active_token_count == 0:
            return

        assert flat_routing.shape[0] == active_token_count, (
            f"Routing token count {flat_routing.shape[0]} != "
            f"active token count {active_token_count}"
        )

        block_ids_np = token_to_block_idx[:active_token_count].cpu().numpy()
        positions_np = token_to_local_position[:active_token_count].cpu().numpy()

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
            if bid == dummy_block_idx:
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

        Args:
            block_ids: Ordered list of block IDs for the request.
            total_routing_tokens: Expected number of routing tokens.

        Returns:
            ndarray of shape `[total_routing_tokens, num_layers, topk]` or `None`.
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
            routing: ndarray of routing data `[num_positions, num_layers, topk]`.
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
            ndarray `[block_size_tokens, num_layers, topk]` or `None` if not stored.
        """
        return self.block_routing.get(block_id)
