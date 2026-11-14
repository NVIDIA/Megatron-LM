# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Launch limits shared by the symmetric-memory Triton collectives.

Every kernel that calls symm_mem_sync gives each (block, peer) pair one int32 slot in
the symmetric-memory signal pad, at block_id * world_size + rank. SymmetricMemoryBuffer
sizes the pad from max_num_blocks when each buffer is allocated, so no launch may use
more than max_num_blocks blocks.

A block that waits in symm_mem_sync keeps its SM until the blocks with the same block_id
on every peer arrive, so those blocks must be able to start. Up to max_concurrent_kernels
barrier kernels can be in flight on a GPU at once: with shared-expert overlap, the shared
expert's TP collectives on SharedExpertMLP.stream run while the MoE dispatcher's run on
the current stream. If one of them could fill the GPU, one rank could fill its SMs with
the first kernel's blocks and a peer with the second's, each waiting on blocks the other
cannot schedule, and the ranks would deadlock. grid_limit() therefore gives each kernel at
most an equal share of the SMs, so all the kernels in flight fit on the GPU together, in
any order the hardware schedules their blocks.

Do not autotune the number of blocks (the grid size, or BLOCK_SIZE where the grid is
derived from it) with triton.autotune or any other per-rank tuning:

  1. A grid larger than max_num_blocks writes barrier slots past the end of the
     signal pad.
  2. Autotuning picks a config from local timings, so ranks can choose different
     grids. A rank then waits on barrier slots that its peers never signal, and the
     collective hangs.
  3. The autotune benchmark launches the kernel a timing-dependent number of times on
     each rank, so barrier calls no longer pair up across ranks.

Grid sizes must be a deterministic function of values that are identical on every
rank, capped at grid_limit(). grid_limit() reads the GPU's SM count, which is the same
on every rank: every GPU of an NVLink multicast group is the same model.
"""

import functools
from dataclasses import dataclass

import torch


@functools.lru_cache(maxsize=None)
def _multi_processor_count(device_index: int) -> int:
    """Number of SMs of a CUDA device, queried once per device."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@dataclass(frozen=True)
class SymmMemKernelConfig:
    """Launch limits for the symmetric-memory collectives.

    Frozen because the signal pads are sized from max_num_blocks at allocation time;
    raising it afterwards would let launches overflow pads that already exist.
    """

    max_num_blocks: int = 128
    """Blocks the signal pads are sized for; grid_limit() never exceeds it."""

    max_block_size: int = 1024
    """Threads per block cap; each thread moves one 64- or 128-bit chunk."""

    warp_size: int = 32

    max_concurrent_kernels: int = 2
    """Barrier kernels that can be in flight on one GPU at once (shared-expert overlap)."""

    def signal_pad_bytes(self, world_size: int) -> int:
        """Signal pad bytes symm_mem_sync needs for a group of world_size ranks."""
        # One int32 slot per (block, peer), at block_id * world_size + rank.
        return self.max_num_blocks * world_size * 4

    def grid_limit(self) -> int:
        """Grid cap on the current device for every kernel that synchronizes through
        symm_mem_sync: max_num_blocks, and an equal share of the SMs for each of the
        max_concurrent_kernels barrier kernels that can be in flight at once."""
        num_sms = _multi_processor_count(torch.cuda.current_device())
        return min(self.max_num_blocks, num_sms // self.max_concurrent_kernels)

    def check_num_blocks(self, num_blocks: int) -> None:
        """Assert a launch fits in the signal pads and leaves the SMs the other barrier
        kernels in flight need."""
        limit = self.grid_limit()
        assert num_blocks <= limit, (
            f"Launching {num_blocks} blocks, but symmetric-memory barrier kernels may use at "
            f"most {limit}: the signal pads are sized for max_num_blocks={self.max_num_blocks}, "
            f"and {self.max_concurrent_kernels} such kernels in flight at once must fit on the "
            f"GPU's SMs together."
        )


SYMM_MEM_KERNEL_CONFIG = SymmMemKernelConfig()
