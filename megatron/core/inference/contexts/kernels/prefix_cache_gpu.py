# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

# Maximum linear probe depth. At 50% load with uniform hashing, expected probe
# length is ~2. A bound of 128 is astronomically unlikely to be exceeded.
MAX_PROBE: int = 128


@triton.jit
def _hash_table_insert_kernel(
    hash_keys_ptr,
    hash_values_ptr,
    input_hashes_ptr,
    input_block_ids_ptr,
    num_entries: tl.int32,
    table_size_mask: tl.int64,
    MAX_PROBE: tl.constexpr,
):
    """Insert (hash, block_id) pairs into open-addressing hash table. Grid: (num_entries,)."""
    i = tl.program_id(0)
    if i < num_entries:
        h = tl.load(input_hashes_ptr + i).to(tl.int64)
        bid = tl.load(input_block_ids_ptr + i).to(tl.int32)
        slot = h & table_size_mask
        done = 0
        for _ in tl.static_range(MAX_PROBE):
            if done == 0:
                key = tl.load(hash_keys_ptr + slot).to(tl.int64)
                if key == 0 or key == -1 or key == h:
                    tl.store(hash_keys_ptr + slot, h)
                    tl.store(hash_values_ptr + slot, bid)
                    done = 1
                slot = (slot + 1) & table_size_mask


@triton.jit
def _hash_table_lookup_kernel(
    hash_keys_ptr,
    hash_values_ptr,
    query_hashes_ptr,
    output_block_ids_ptr,
    num_queries: tl.int32,
    table_size_mask: tl.int64,
    MAX_PROBE: tl.constexpr,
):
    """Batch lookup hashes, write block_ids to output (-1 for misses). Grid: (num_queries,)."""
    i = tl.program_id(0)
    if i < num_queries:
        h = tl.load(query_hashes_ptr + i).to(tl.int64)
        slot = h & table_size_mask
        result = tl.cast(-1, tl.int32)
        done = 0
        for _ in tl.static_range(MAX_PROBE):
            if done == 0:
                key = tl.load(hash_keys_ptr + slot).to(tl.int64)
                if key == h:
                    result = tl.load(hash_values_ptr + slot).to(tl.int32)
                    done = 1
                if key == 0:
                    done = 1
                slot = (slot + 1) & table_size_mask
        tl.store(output_block_ids_ptr + i, result)


@triton.jit
def _hash_table_delete_kernel(
    hash_keys_ptr,
    hash_values_ptr,
    block_hashes_ptr,
    delete_block_ids_ptr,
    num_entries: tl.int32,
    table_size_mask: tl.int64,
    MAX_PROBE: tl.constexpr,
):
    """Delete entries by block_id. Looks up hash from block_hashes tensor. Grid: (num_entries,)."""
    i = tl.program_id(0)
    if i < num_entries:
        bid = tl.load(delete_block_ids_ptr + i).to(tl.int64)
        h = tl.load(block_hashes_ptr + bid).to(tl.int64)
        if h > 0:
            slot = h & table_size_mask
            done = 0
            for _ in tl.static_range(MAX_PROBE):
                if done == 0:
                    key = tl.load(hash_keys_ptr + slot).to(tl.int64)
                    if key == h:
                        tl.store(hash_keys_ptr + slot, tl.cast(-1, tl.int64))
                        tl.store(hash_values_ptr + slot, tl.cast(-1, tl.int32))
                        done = 1
                    if key == 0:
                        done = 1
                    slot = (slot + 1) & table_size_mask


@triton.jit
def _hash_table_clear_kernel(
    hash_keys_ptr,
    hash_values_ptr,
    table_size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero all keys, set all values to -1. Grid: (cdiv(table_size, BLOCK_SIZE),)."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < table_size
    tl.store(hash_keys_ptr + offset, tl.zeros([BLOCK_SIZE], dtype=tl.int64), mask=mask)
    tl.store(
        hash_values_ptr + offset, tl.full([BLOCK_SIZE], -1, dtype=tl.int32), mask=mask
    )


@triton.jit
def _hash_table_count_kernel(
    hash_keys_ptr,
    count_ptr,
    table_size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Count non-empty, non-tombstone entries. Grid: (1,)."""
    total = tl.cast(0, tl.int32)
    for offset in range(0, table_size, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < table_size
        keys = tl.load(hash_keys_ptr + idx, mask=mask, other=0)
        total += tl.sum((keys > 0).to(tl.int32))
    tl.store(count_ptr, total)


class GPUHashTable:
    """GPU-resident open-addressing hash table for prefix cache hash-to-block mapping.

    No CPU shadow dict. All operations go through GPU Triton kernels.
    """

    def __init__(self, total_count: int, device: torch.device):
        self.table_size = triton.next_power_of_2(max(total_count * 2, 16))
        self.table_size_mask = self.table_size - 1
        self.hash_keys = torch.zeros(self.table_size, dtype=torch.int64, device=device)
        self.hash_values = torch.full(
            (self.table_size,), -1, dtype=torch.int32, device=device
        )
        self._device = device
        self._count_buf = torch.zeros(1, dtype=torch.int32, device=device)

    def insert(self, hashes: List[int], block_ids: List[int]) -> None:
        """Insert hash→block_id pairs into GPU table."""
        if not hashes:
            return
        h_tensor = torch.tensor(hashes, dtype=torch.int64, device=self._device)
        b_tensor = torch.tensor(block_ids, dtype=torch.int32, device=self._device)
        n = len(hashes)
        _hash_table_insert_kernel[(n,)](
            self.hash_keys, self.hash_values,
            h_tensor, b_tensor,
            n, self.table_size_mask,
            MAX_PROBE=MAX_PROBE,
        )

    def lookup(self, query_hashes: Tensor) -> Tensor:
        """Batch lookup hashes on GPU. Returns int32 tensor of block_ids (-1 for misses)."""
        n = query_hashes.shape[0]
        if n == 0:
            return torch.empty(0, dtype=torch.int32, device=self._device)
        output = torch.full((n,), -1, dtype=torch.int32, device=self._device)
        _hash_table_lookup_kernel[(n,)](
            self.hash_keys, self.hash_values,
            query_hashes, output,
            n, self.table_size_mask,
            MAX_PROBE=MAX_PROBE,
        )
        return output

    def lookup_single(self, hash_val: int) -> int:
        """Single-element GPU lookup. Returns block_id or -1."""
        h = torch.tensor([hash_val], dtype=torch.int64, device=self._device)
        result = self.lookup(h)
        return result.item()

    def delete_by_block_ids(self, block_ids: Tensor, block_hashes_tensor: Tensor) -> None:
        """Delete entries by block_id using the block_hashes tensor to find hashes."""
        n = block_ids.shape[0]
        if n == 0:
            return
        _hash_table_delete_kernel[(n,)](
            self.hash_keys, self.hash_values,
            block_hashes_tensor, block_ids,
            n, self.table_size_mask,
            MAX_PROBE=MAX_PROBE,
        )

    def clear(self) -> None:
        """Reset entire table to empty."""
        block_size = 1024
        grid = ((self.table_size + block_size - 1) // block_size,)
        _hash_table_clear_kernel[grid](
            self.hash_keys, self.hash_values,
            self.table_size,
            BLOCK_SIZE=block_size,
        )

    def __contains__(self, hash_val: int) -> bool:
        """Membership test via GPU single lookup."""
        return self.lookup_single(hash_val) != -1

    def get(self, hash_val: int, default=None):
        """Dict-like get via GPU single lookup."""
        result = self.lookup_single(hash_val)
        return result if result != -1 else default

    def __len__(self) -> int:
        """Count entries via GPU kernel."""
        _hash_table_count_kernel[(1,)](
            self.hash_keys, self._count_buf,
            self.table_size,
            BLOCK_SIZE=1024,
        )
        return self._count_buf.item()

    def keys(self):
        """Return set of all valid hash keys (transfers to CPU)."""
        valid_mask = self.hash_keys > 0
        return set(self.hash_keys[valid_mask].tolist())
