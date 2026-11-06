# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NCCL hand-off transfers to different peers must not serialize into a wait cycle."""

import os
import time

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.coordinator_setup import transfer_rank_pairs
from megatron.core.inference.disaggregation.transfer_backends import nccl as nccl_mod
from megatron.core.inference.shards_spec import parse_inference_shards_spec


def test_transfer_rank_pairs_cover_prefill_decode_cross_product():
    specs = parse_inference_shards_spec("tp=1,dp=2,role=prefill+tp=1,dp=2,role=decode", 4)
    assert transfer_rank_pairs(specs) == [(0, 2), (0, 3), (1, 2), (1, 3)]


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 4 or not torch.cuda.is_available(),
    reason="needs 4 CUDA ranks",
)
def test_crossing_transfer_orders_complete_with_peer_groups():
    """Prefill ranks 0, 1 and decode ranks 2, 3 post in crossing orders.

    On one shared communicator stream this is a wait cycle: 0 sends to 2 then 3, 1 sends to 3
    then 2, 2 receives from 1 then 0, 3 receives from 0 then 1. With per-pair groups every
    transfer completes and carries its sender's data.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    nccl_mod.setup_peer_groups([(0, 2), (0, 3), (1, 2), (1, 3)], device)
    try:
        if rank > 3:
            return
        order = {0: [2, 3], 1: [3, 2], 2: [1, 0], 3: [0, 1]}[rank]
        sending = rank < 2
        handles, buffers = [], {}
        for peer in order:
            if sending:
                payload = torch.full((1 << 20,), float(10 * rank + peer), device=device)
                plan = nccl_mod.NcclTransferPlan(
                    [nccl_mod._p2p(dist.isend, payload, peer)], [payload], []
                )
            else:
                buffers[peer] = torch.zeros(1 << 20, device=device)
                plan = nccl_mod.NcclTransferPlan(
                    [nccl_mod._p2p(dist.irecv, buffers[peer], peer)], [buffers[peer]], []
                )
            handles.append(nccl_mod.NcclTransferBackend.start_prepared([plan]))
        deadline = time.monotonic() + 120
        while not all(h.poll() for h in handles):
            assert time.monotonic() < deadline, "crossing transfers did not complete"
            time.sleep(0.01)
        for peer, buf in buffers.items():
            assert torch.all(buf == float(10 * peer + rank))
    finally:
        nccl_mod._PEER_GROUPS.clear()
