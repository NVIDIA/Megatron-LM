"""Unit tests for TemporaryBucketAllocator. Pure CPU, no torch.distributed."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard_rewrite.allocator import (
    Bucket,
    TemporaryBucketAllocator,
)


def _run_allocator_tests(allocator: TemporaryBucketAllocator) -> None:
    """Three-phase test covering allocate, free, and re-allocate for any allocator.

    Phase 1 -- allocate:
      - Returned bucket has correct size, dtype, and is tracked internally.
      - Duplicate allocate on the same id returns the same object (no realloc).
      - Different ids produce independent buckets.

    Phase 2 -- free:
      - Freed id is removed from internal tracking.
      - Underlying tensor storage is resized to 0.
      - Freeing a non-existent id is silently ignored.
      - Freeing one id does not affect others.

    Phase 3 -- re-allocate after free:
      - Re-allocating a freed id returns a usable bucket with correct size.
    """

    # ---- Phase 1: allocate ----
    b0 = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0.data.numel() == 1024
    assert b0.data.dtype == torch.float32
    assert 0 in allocator.buckets

    b0_again = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0_again is b0
    assert b0_again.data.data_ptr() == b0.data.data_ptr()

    b1 = allocator.allocate(
        param_group_id=1, size=512, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert b1 is not b0
    assert b1.data.numel() == 512
    assert b1.data.dtype == torch.bfloat16

    # ---- Phase 2: free ----
    tensor_ref = b0.data
    allocator.free(0)
    assert 0 not in allocator.buckets
    assert tensor_ref._typed_storage()._size() == 0

    assert 1 in allocator.buckets
    assert b1.data.numel() == 512

    allocator.free(999)

    # ---- Phase 3: re-allocate ----
    b0_new = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0_new.data.numel() == 1024
    assert 0 in allocator.buckets

    # cleanup
    allocator.free(0)
    allocator.free(1)


class TestTemporaryBucketAllocator:

    def test_full_lifecycle(self):
        _run_allocator_tests(TemporaryBucketAllocator())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
