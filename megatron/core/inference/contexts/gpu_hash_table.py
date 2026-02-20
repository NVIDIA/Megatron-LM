# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# Maximum linear probe distance for hash table operations. At 50% load factor,
# expected probe length is ~2; 256 is far beyond any realistic collision chain.
MAX_PROBE_DISTANCE = tl.constexpr(256)


# =========================================================================
# Triton kernels
# =========================================================================


@triton.jit
def _hash_table_insert_kernel(
    TABLE_KEYS,
    TABLE_VALUES,
    INSERT_KEYS,
    INSERT_VALUES,
    CAPACITY_MASK: tl.constexpr,
    EMPTY: tl.constexpr,
):
    """Insert N key-value pairs with linear probing.

    Each program handles one key-value pair. Collisions are resolved via
    linear probing with non-atomic stores. A Python-level retry loop in
    insert_batch() re-inserts any keys lost to races between threads.
    """
    pid = tl.program_id(0)
    key = tl.load(INSERT_KEYS + pid)
    val = tl.load(INSERT_VALUES + pid)
    # Cast EMPTY to int64 to match TABLE_KEYS pointer type
    empty_i64 = tl.full([], EMPTY, dtype=tl.int64)

    # Skip empty sentinel keys
    if key != empty_i64:
        slot = (key & CAPACITY_MASK).to(tl.int64)  # Fast modulo for power-of-2
        done: tl.int32 = 0

        # Linear probing with bounded search
        for _ in range(MAX_PROBE_DISTANCE):
            if done == 0:
                existing = tl.load(TABLE_KEYS + slot)
                if existing == empty_i64 or existing == key:
                    tl.store(TABLE_KEYS + slot, key)
                    tl.store(TABLE_VALUES + slot, val)
                    done = 1
                else:
                    slot = ((slot + 1) & CAPACITY_MASK).to(tl.int64)


@triton.jit
def _hash_table_lookup_kernel(
    TABLE_KEYS,
    TABLE_VALUES,
    QUERY_KEYS,
    RESULTS,
    CAPACITY_MASK: tl.constexpr,
    EMPTY: tl.constexpr,
):
    """Lookup N keys, write block_id or -1 to RESULTS.

    Each program handles one query key.
    """
    pid = tl.program_id(0)
    key = tl.load(QUERY_KEYS + pid)
    # Cast EMPTY to int64 to match TABLE_KEYS element type
    empty_i64 = tl.full([], EMPTY, dtype=tl.int64)

    slot = (key & CAPACITY_MASK).to(tl.int64)
    result: tl.int32 = -1
    done = 0

    for _ in range(MAX_PROBE_DISTANCE):
        if done == 0:
            existing_key = tl.load(TABLE_KEYS + slot)
            if existing_key == key:
                result = tl.load(TABLE_VALUES + slot)
                done = 1
            elif existing_key == empty_i64:
                done = 1
            else:
                slot = ((slot + 1) & CAPACITY_MASK).to(tl.int64)

    tl.store(RESULTS + pid, result.to(tl.int32))


@triton.jit
def _prefix_match_kernel(
    # Hash table
    TABLE_KEYS,
    TABLE_VALUES,
    CAPACITY_MASK: tl.constexpr,
    # Per-request hashes (flattened)
    REQUEST_HASHES,
    REQUEST_OFFSETS,
    # Pending bitmap
    PENDING_BITMAP,
    # Outputs (per request)
    NUM_MATCHED,
    HAS_PENDING,
    MATCHED_BLOCK_IDS,
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    EMPTY: tl.constexpr,
):
    """Per-request prefix matching with stop-at-first-miss + pending detection.

    Each program handles one request. Lookups within a request are sequential
    (required by parent-chain integrity). Requests are processed in parallel.

    Args:
        TABLE_KEYS/TABLE_VALUES: Hash table storage [capacity]
        CAPACITY_MASK: capacity - 1 for fast modulo
        REQUEST_HASHES: [total_hashes] flattened hashes for all requests
        REQUEST_OFFSETS: [num_requests+1] offsets into REQUEST_HASHES
        PENDING_BITMAP: [total_blocks] bool: True if block is pending
        NUM_MATCHED: [num_requests] output: matched block count per request
        HAS_PENDING: [num_requests] output: True if depends on pending block
        MATCHED_BLOCK_IDS: [num_requests, MAX_BLOCKS_PER_REQ] output: matched block IDs
        MAX_BLOCKS_PER_REQ: constexpr max blocks per request
        EMPTY: empty sentinel value (-1)
    """
    req_id = tl.program_id(0)
    start = tl.load(REQUEST_OFFSETS + req_id)
    end = tl.load(REQUEST_OFFSETS + req_id + 1)
    # Cast EMPTY to int64 to match TABLE_KEYS element type
    empty_i64 = tl.full([], EMPTY, dtype=tl.int64)

    matched: tl.int32 = 0
    has_pending: tl.int32 = 0
    block_id: tl.int32 = -1
    found: tl.int32 = 0

    stopped = 0
    for i in tl.static_range(MAX_BLOCKS_PER_REQ):
        if start + i >= end:
            stopped = 1

        # Reset per-iteration state
        found = 0
        block_id = -1

        if stopped == 0:
            hash_key = tl.load(REQUEST_HASHES + start + i)

            # Hash table lookup (inline linear probe)
            slot = (hash_key & CAPACITY_MASK).to(tl.int64)
            probe_done = 0
            for _ in range(MAX_PROBE_DISTANCE):
                if probe_done == 0:
                    existing = tl.load(TABLE_KEYS + slot)
                    if existing == hash_key:
                        block_id = tl.load(TABLE_VALUES + slot)
                        found = 1
                        probe_done = 1
                    elif existing == empty_i64:
                        probe_done = 1
                    else:
                        slot = ((slot + 1) & CAPACITY_MASK).to(tl.int64)

            if found == 0:
                stopped = 1  # Stop at first miss

        if stopped == 0:
            # Check if this block is pending
            if tl.load(PENDING_BITMAP + block_id.to(tl.int64)):
                has_pending = 1

            tl.store(
                MATCHED_BLOCK_IDS + (req_id * MAX_BLOCKS_PER_REQ + matched).to(tl.int64),
                block_id.to(tl.int32),
            )
            matched += 1

    tl.store(NUM_MATCHED + req_id, matched.to(tl.int32))
    tl.store(HAS_PENDING + req_id, has_pending.to(tl.int32))


# =========================================================================
# GPUHashTable class
# =========================================================================


class GPUHashTable:
    """GPU-resident open-addressing hash table with linear probing.

    Keys: int64 (block hashes), Values: int32 (block IDs).
    Empty sentinel: key = -1.
    Capacity: next_power_of_2(2 * max_entries) for ~50% load factor.

    Deletion strategy: Rather than implementing complex tombstone-free
    backward-shift deletion in Triton, we use a "rebuild after eviction"
    approach. Inserts happen individually via the insert kernel, but
    deletions trigger a full rebuild from the source-of-truth tensors
    (block_hashes in BlockAllocator). Rebuild cost is O(num_blocks):
    sub-millisecond for up to ~100K blocks on H100, approaching ~1ms
    near 1M blocks. In LRU mode, rebuilds only occur under memory
    pressure (free pool exhausted). In REF_ZERO mode, rebuilds occur
    on every release but remain cheap at typical block counts (1K-100K).

    Args:
        max_entries: Maximum number of entries the table can hold.
        device: CUDA device for tensor allocation.
    """

    def __init__(self, max_entries: int, device: torch.device):
        self.capacity = next_power_of_2(2 * max(max_entries, 16))
        self.capacity_mask = self.capacity - 1
        self.EMPTY = -1

        self.keys = torch.full(
            (self.capacity,), self.EMPTY, dtype=torch.int64, device=device
        )
        self.values = torch.full(
            (self.capacity,), -1, dtype=torch.int32, device=device
        )
        # Normalize device from the tensor (handles "cuda" vs torch.device("cuda", 0))
        self.device = self.keys.device
        self.size = 0  # CPU counter for fast "is empty?" check

    def insert_batch(self, keys: Tensor, values: Tensor) -> None:
        """Insert a batch of key-value pairs into the hash table.

        If a key already exists, its value is updated. Keys equal to -1
        (the empty sentinel) are skipped.

        The insert kernel uses non-atomic stores, so concurrent threads that
        hash to the same slot can race. We retry with only the missing keys
        until all are inserted. Collisions are astronomically rare in practice
        (64-bit hash space) so the retry loop almost never executes.

        Args:
            keys: int64 tensor of hash keys to insert.
            values: int32 tensor of block IDs (same length as keys).
        """
        n = keys.numel()
        if n == 0:
            return

        assert keys.dtype == torch.int64 and keys.device == self.device
        assert values.dtype == torch.int32 and values.device == self.device

        remaining_keys = keys
        remaining_values = values

        for _ in range(8):
            m = remaining_keys.numel()
            _hash_table_insert_kernel[(m,)](
                self.keys,
                self.values,
                remaining_keys,
                remaining_values,
                CAPACITY_MASK=self.capacity_mask,
                EMPTY=self.EMPTY,
            )

            # Verify all non-sentinel keys were inserted with correct values.
            # Non-atomic stores can cause key-value tears: two threads racing on
            # the same slot may leave one key with the other's value. Check both
            # presence (results >= 0) and value correctness (results == expected).
            valid = remaining_keys != self.EMPTY
            if not valid.any():
                break
            results = torch.full((m,), -1, dtype=torch.int32, device=self.device)
            _hash_table_lookup_kernel[(m,)](
                self.keys,
                self.values,
                remaining_keys,
                results,
                CAPACITY_MASK=self.capacity_mask,
                EMPTY=self.EMPTY,
            )
            wrong = valid & (results != remaining_values)
            if not wrong.any():
                break
            remaining_keys = remaining_keys[wrong]
            remaining_values = remaining_values[wrong]

        self.size += n  # Approximate; may overcount on duplicate inserts

    def lookup_batch(self, query_keys: Tensor, results: Tensor) -> None:
        """Look up a batch of keys, writing block_id or -1 to results.

        Args:
            query_keys: int64 tensor of hash keys to look up.
            results: int32 output tensor (same length as query_keys).
                     Filled with block_id if found, -1 if not found.
        """
        n = query_keys.numel()
        if n == 0:
            return

        assert query_keys.dtype == torch.int64 and query_keys.device == self.device
        assert results.dtype == torch.int32 and results.device == self.device

        _hash_table_lookup_kernel[(n,)](
            self.keys,
            self.values,
            query_keys,
            results,
            CAPACITY_MASK=self.capacity_mask,
            EMPTY=self.EMPTY,
        )

    def lookup_batch_alloc(self, query_keys: Tensor) -> Tensor:
        """Look up a batch of keys, returning results tensor.

        Convenience wrapper around lookup_batch that allocates the output.

        Args:
            query_keys: int64 tensor of hash keys to look up.

        Returns:
            int32 tensor with block_id if found, -1 if not found.
        """
        results = torch.full(
            (query_keys.numel(),), -1, dtype=torch.int32, device=self.device
        )
        self.lookup_batch(query_keys, results)
        return results

    def rebuild(self, valid_keys: Tensor, valid_values: Tensor) -> None:
        """Rebuild the hash table from scratch with the given key-value pairs.

        Used after batch eviction instead of per-element deletion.
        Clears the table and re-inserts all valid entries.

        Args:
            valid_keys: int64 tensor of all valid hash keys.
            valid_values: int32 tensor of corresponding block IDs.
        """
        self.keys.fill_(self.EMPTY)
        self.values.fill_(-1)
        self.size = 0
        self.insert_batch(valid_keys, valid_values)

    def prefix_match_batch(
        self,
        request_hashes: Tensor,
        request_offsets: Tensor,
        pending_bitmap: Tensor,
        max_blocks_per_req: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Batch prefix matching for multiple requests on GPU.

        For each request, looks up its sequence of hashes in the hash table,
        stopping at the first miss. Also checks if any matched block is pending.

        Args:
            request_hashes: [total_hashes] int64, flattened hashes for all requests.
            request_offsets: [num_requests+1] int32, offsets into request_hashes.
            pending_bitmap: [total_blocks] bool, True if block is pending computation.
            max_blocks_per_req: Maximum blocks any single request can have.

        Returns:
            Tuple of:
            - num_matched: [num_requests] int32, matched block count per request.
            - has_pending: [num_requests] int32, 1 if depends on pending block.
            - matched_block_ids: [num_requests, max_blocks_per_req] int32, matched IDs.
        """
        num_requests = request_offsets.numel() - 1
        if num_requests == 0:
            return (
                torch.empty(0, dtype=torch.int32, device=self.device),
                torch.empty(0, dtype=torch.int32, device=self.device),
                torch.empty((0, max_blocks_per_req), dtype=torch.int32, device=self.device),
            )

        num_matched = torch.zeros(num_requests, dtype=torch.int32, device=self.device)
        has_pending = torch.zeros(num_requests, dtype=torch.int32, device=self.device)
        matched_block_ids = torch.full(
            (num_requests, max_blocks_per_req), -1, dtype=torch.int32, device=self.device
        )

        _prefix_match_kernel[(num_requests,)](
            self.keys,
            self.values,
            CAPACITY_MASK=self.capacity_mask,
            REQUEST_HASHES=request_hashes,
            REQUEST_OFFSETS=request_offsets,
            PENDING_BITMAP=pending_bitmap,
            NUM_MATCHED=num_matched,
            HAS_PENDING=has_pending,
            MATCHED_BLOCK_IDS=matched_block_ids,
            MAX_BLOCKS_PER_REQ=max_blocks_per_req,
            EMPTY=self.EMPTY,
        )

        return num_matched, has_pending, matched_block_ids

    def clear(self) -> None:
        """Clear all entries from the hash table."""
        self.keys.fill_(self.EMPTY)
        self.values.fill_(-1)
        self.size = 0
