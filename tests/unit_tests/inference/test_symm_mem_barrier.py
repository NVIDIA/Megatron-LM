# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The symmetric-memory barrier (``symm_mem_sync``) on groups of any size, including 3."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from tests.unit_tests.test_utilities import Utils

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _exchange_kernel(
        buffer_ptrs, signal_pad_ptrs, out_ptr, value, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr
    ):
        """Publish ``value`` in this rank's buffer, barrier, read every rank's value, barrier."""
        bases = buffer_ptrs.to(tl.pointer_type(tl.uint64))
        own = tl.load(bases + RANK).to(tl.pointer_type(tl.int32))
        tl.store(own, value)
        sync_threads()
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )
        sync_threads()
        for r in tl.static_range(WORLD_SIZE):
            peer = tl.load(bases + r).to(tl.pointer_type(tl.int32))
            tl.store(out_ptr + r, tl.load(peer, volatile=True))
        sync_threads()
        # Every rank has read the values before any rank overwrites its own next step.
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=False,
        )


@pytest.mark.skipif(
    not HAVE_TRITON
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9
    or Utils.world_size < 4,
    reason="needs 4 GPUs with symmetric memory (sm90+) and triton",
)
@pytest.mark.parametrize("group_size", [2, 3, 4])
def test_barrier_orders_writes_across_the_group(group_size):
    """Each step every rank sees every other rank's value of that step: the barrier, including
    its rank vector padded to a power of two at group size 3, orders writes before reads.

    Every rank allocates the same world-group buffer (the rendezvous is collective over the
    whole world, so no rank's allocation sequence diverges); the first ``group_size`` ranks
    then run the barrier among themselves on its first ``group_size`` buffers and signal pads.
    """
    Utils.initialize_model_parallel(1, 1)
    try:
        hdl = SymmetricMemoryManager.get_buffer(
            "barrier_test", process_group=dist.group.WORLD, size_mb=1
        ).symm_mem_hdl
        assert hdl is not None
        rank = dist.get_rank()
        if rank < group_size:
            out = torch.full((group_size,), -1, dtype=torch.int32, device="cuda")
            for step in range(50):
                _exchange_kernel[(1,)](
                    hdl.buffer_ptrs_dev,
                    hdl.signal_pad_ptrs_dev,
                    out,
                    step * 16 + rank,
                    RANK=rank,
                    WORLD_SIZE=group_size,
                )
                expected = [step * 16 + r for r in range(group_size)]
                assert out.tolist() == expected, (step, out.tolist())
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
