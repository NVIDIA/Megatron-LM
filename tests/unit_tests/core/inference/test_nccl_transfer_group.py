# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NCCL hand-off transfers on the shared transfer group: global post order and pre-connection."""

import os
import time

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.coordinator_setup import transfer_rank_pairs
from megatron.core.inference.disaggregation.transfer_backends import nccl as nccl_mod
from megatron.core.inference.shards_spec import parse_inference_shards_spec

PAIRS = [(0, 2), (0, 3), (1, 2), (1, 3)]
needs_four_ranks = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 4 or not torch.cuda.is_available(),
    reason="needs 4 CUDA ranks",
)


def test_transfer_rank_pairs_cover_prefill_decode_cross_product():
    specs = parse_inference_shards_spec("tp=1,dp=2,role=prefill+tp=1,dp=2,role=decode", 4)
    assert transfer_rank_pairs(specs) == PAIRS


@pytest.fixture
def transfer_group():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    nccl_mod.setup_transfer_group(PAIRS, device)
    yield dist.get_rank(), device
    nccl_mod._TRANSFER_GROUP = None


def _post(rank, device, peer, value=None):
    """Post one 4 MiB transfer with ``peer`` as a backend plan; return its handle and buffer."""
    if rank < 2:
        buf = torch.full((1 << 20,), float(value), device=device)
        op = nccl_mod._p2p(dist.isend, buf, peer)
    else:
        buf = torch.zeros(1 << 20, device=device)
        op = nccl_mod._p2p(dist.irecv, buf, peer)
    plan = nccl_mod.NcclTransferPlan([op], [buf], [])
    return nccl_mod.NcclTransferBackend.start_prepared([plan]), buf


@needs_four_ranks
def test_transfers_in_one_global_order_complete_on_the_shared_group(transfer_group):
    """Ranks post in the coordinator's order (t1: 0->2, t2: 1->3, t3: 0->3, t4: 1->2).

    Every rank posts its share in that order, as the READY/SEND_KV relay guarantees, so the
    shared communicator completes all four even though prefill 1 and decode 3 each reach
    their two peers in the opposite order from prefill 0 and decode 2.
    """
    rank, device = transfer_group
    if rank > 3:
        return
    order = {0: [2, 3], 1: [3, 2], 2: [0, 1], 3: [1, 0]}[rank]
    posted = [(peer, *_post(rank, device, peer, 10 * rank + peer)) for peer in order]
    deadline = time.monotonic() + 120
    while not all(handle.poll() for _, handle, _ in posted):
        assert time.monotonic() < deadline, "globally ordered transfers did not complete"
        time.sleep(0.01)
    if rank >= 2:
        for peer, _, buf in posted:
            assert torch.all(buf == float(10 * peer + rank))


@needs_four_ranks
def test_first_transfer_after_setup_does_not_block_the_poster(transfer_group):
    """The pairs are connected at setup, so posting returns before the peer posts."""
    rank, device = transfer_group
    if rank not in (0, 2):
        return
    if rank == 0:
        start = time.monotonic()
        handle, _ = _post(rank, device, 2, 7)
        assert time.monotonic() - start < 1.0, "posting blocked on the peer"
        handle.wait()
    else:
        time.sleep(3)
        handle, buf = _post(rank, device, 0)
        handle.wait()
        assert torch.all(buf == 7.0)
