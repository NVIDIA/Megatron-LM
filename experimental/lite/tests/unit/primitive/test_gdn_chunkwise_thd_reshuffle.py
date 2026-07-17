# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packing-aware THD CP reshuffle for the restored GDN ``chunkwise`` mode.

The GDN ``chunkwise`` CP mode reshuffles the Megatron zigzag CP layout to
contiguous-time chunks (and back) around the FLA recurrence. For packed THD this
reshuffle must route on the *global* ``cu_seqlens`` so per-sequence zigzag chunk
boundaries survive even when a sequence spans the contiguous CP-rank boundary — the
exact case the removed ``sharded`` copy corrupted (it sliced ``cu_seqlens // cp`` and
swapped each sequence independently).

``test_thd_rank_indices_*`` pin the index math on CPU without any distributed setup;
``test_thd_reshuffle_roundtrip_gloo`` runs the real primitive under gloo and asserts a
bitwise round-trip against the ground-truth contiguous span. The per-head recurrence is
unchanged FLA GPU code, so a bitwise-correct reshuffle => the GPU chunkwise path is fed
the same contiguous-time tokens upstream Megatron feeds its kernel (GPU numeric parity
is the separate ``gpu`` gate).
"""
from __future__ import annotations

import multiprocessing
import os

import pytest
import torch

from megatron.lite.primitive.parallel.cp import get_thd_context_parallel_rank_indices

pytestmark = pytest.mark.mlite

# Hand-computed indices for the anchor case (cp=2, cu=[0,8,12]): seq0 len8 splits into
# 4 chunks [0,1|2,3|4,5|6,7] with zigzag owners r0,r1,r1,r0; seq1 len4 -> [8|9|10|11]
# owners r0,r1,r1,r0. Contiguous halves are the trivial global slices.
ANCHOR_CU = [0, 8, 12]
ANCHOR_ZIGZAG = {0: [0, 1, 6, 7, 8, 11], 1: [2, 3, 4, 5, 9, 10]}
ANCHOR_CONTIGUOUS = {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11]}


def test_thd_rank_indices_match_hand_computed_anchor():
    cu = torch.tensor(ANCHOR_CU, dtype=torch.long)
    for rank in range(2):
        zz = get_thd_context_parallel_rank_indices(cu, 2, rank, "zigzag").tolist()
        cont = get_thd_context_parallel_rank_indices(cu, 2, rank, "contiguous").tolist()
        assert zz == ANCHOR_ZIGZAG[rank], f"zigzag rank{rank}: {zz}"
        assert cont == ANCHOR_CONTIGUOUS[rank], f"contiguous rank{rank}: {cont}"


@pytest.mark.parametrize(
    "cp, cu",
    [
        (2, [0, 8, 12]),
        (4, [0, 16, 24, 32]),
        (2, [0, 16, 24]),
    ],
)
def test_thd_rank_indices_partition_invariants(cp, cu):
    """Each layout partitions the global tokens exactly once; contiguous is a plain slice."""
    cu_t = torch.tensor(cu, dtype=torch.long)
    total = cu[-1]
    part = total // cp
    for layout in ("zigzag", "contiguous"):
        owned = [
            get_thd_context_parallel_rank_indices(cu_t, cp, r, layout).tolist() for r in range(cp)
        ]
        flat = sorted(idx for span in owned for idx in span)
        assert flat == list(range(total)), f"{layout} not a clean partition: {flat}"
        assert all(len(span) == part for span in owned), f"{layout} uneven shards"
    for r in range(cp):
        cont = get_thd_context_parallel_rank_indices(cu_t, cp, r, "contiguous").tolist()
        assert cont == list(range(r * part, (r + 1) * part))


def test_thd_rank_indices_rejects_indivisible_length():
    # seq len 6 is not divisible by 2*cp=4 -> zigzag chunking is undefined.
    cu = torch.tensor([0, 6], dtype=torch.long)
    with pytest.raises(ValueError):
        get_thd_context_parallel_rank_indices(cu, 2, 0, "zigzag")


# --------------------------------------------------------------------- gloo round-trip
def _reshuffle_worker(rank, world, cu_list, port, results):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world)
    )
    import torch.distributed as dist

    from megatron.lite.primitive.parallel.cp import (
        contiguous_to_zigzag_chunks,
        zigzag_to_contiguous_chunks,
    )

    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)))
    try:
        cu = torch.tensor(cu_list, dtype=torch.long)
        total = int(cu[-1])
        part = total // world
        torch.manual_seed(20260715)
        full = torch.randn(total, 8)  # identical across ranks (same seed)

        zz_idx = get_thd_context_parallel_rank_indices(cu, world, rank, "zigzag")
        local_zigzag = full.index_select(0, zz_idx).contiguous()

        got_contig = zigzag_to_contiguous_chunks(local_zigzag, group, seq_dim=0, cu_seqlens=cu)
        expect_contig = full[rank * part : (rank + 1) * part].contiguous()
        fwd = (got_contig - expect_contig).abs().max().item()

        back = contiguous_to_zigzag_chunks(got_contig, group, seq_dim=0, cu_seqlens=cu)
        rt = (back - local_zigzag).abs().max().item()
        results.append((rank, fwd, rt))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize(
    "cp, cu, port",
    [
        (2, [0, 8, 12], 29630),
        (4, [0, 16, 24, 32], 29631),
        (2, [0, 16, 24], 29632),
    ],
)
def test_thd_reshuffle_roundtrip_gloo(cp, cu, port):
    import torch.multiprocessing as mp

    mgr = multiprocessing.Manager()
    results = mgr.list()
    mp.spawn(_reshuffle_worker, args=(cp, cu, port, results), nprocs=cp, join=True)
    assert len(results) == cp, f"missing ranks: {list(results)}"
    for rank, fwd, rt in results:
        assert fwd == 0.0, f"rank{rank} zigzag->contiguous not bitwise: max_abs={fwd}"
        assert rt == 0.0, f"rank{rank} round-trip not bitwise: max_abs={rt}"
