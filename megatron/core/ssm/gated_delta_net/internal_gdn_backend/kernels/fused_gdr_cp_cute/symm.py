# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""Symmetric-memory buffer set shared by the fused CP kernels (CuteDSL).

Allocates, zeroes and rendezvous the three symmetric tensors of the protocol
and builds the per-process peer pointer tables:

  hm_sym [2, R, HV, NSTRIPS, DK, 64] fp32   strip-major payload, parity-buffered
  flags  [R, HV, NSTRIPS]            uint32 monotonic step_id, producer->consumer
  acks   [R, HV, NV]                 uint32 monotonic step_id, consumer->producer
"""

import torch

BT = 64


class CPSymmBuffers:
    def __init__(self, group, HV, DK, DV, device):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.R = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.NV, self.NK = DV // 64, DK // 64
        self.NSTRIPS = self.NV + self.NK
        self.device = device

        self.hm_sym = symm_mem.empty(
            (2, self.R, HV, self.NSTRIPS, DK, BT), dtype=torch.float32, device=device
        )
        self.flags = symm_mem.empty((self.R, HV, self.NSTRIPS), dtype=torch.uint32, device=device)
        self.acks = symm_mem.empty((self.R, HV, self.NV), dtype=torch.uint32, device=device)
        self.flags.zero_()
        self.acks.zero_()
        hdls = [symm_mem.rendezvous(t, group) for t in (self.hm_sym, self.flags, self.acks)]
        dtypes = (torch.float32, torch.uint32, torch.uint32)
        tensors = (self.hm_sym, self.flags, self.acks)
        self.hm_ptrs, self.fl_ptrs, self.ak_ptrs = (
            torch.tensor(
                [h.get_buffer(j, t.shape, dt).data_ptr() for j in range(self.R)],
                dtype=torch.int64,
                device=device,
            )
            for h, t, dt in zip(hdls, tensors, dtypes)
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)  # flags/acks zeroed everywhere before any signal
