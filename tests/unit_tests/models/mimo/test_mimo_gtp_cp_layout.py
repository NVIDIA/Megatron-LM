# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MIMO rank ownership and bridge routing with simulated distributed groups."""

from math import lcm
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from examples.mimo.training.topology import ModuleGridSpec, _build_grid, pg_collection_from_grid
from examples.mimo.utils.hetero import get_data_lane_rank
from megatron.core.gtp_parallel_layout import GTPParallelLayout
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator


@pytest.fixture
def rank_world(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    world = SimpleNamespace(rank=0)
    monkeypatch.setenv("WORLD_SIZE", "4096")
    monkeypatch.setattr(
        dist, "get_rank", lambda group=None: world.rank if group is None else group.rank()
    )
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dist, "barrier", lambda: None)

    def group(ranks, **kwargs):
        if world.rank not in ranks:
            return None
        ranks = sorted(ranks)
        return SimpleNamespace(
            ranks=ranks, rank=lambda: ranks.index(world.rank), size=lambda: len(ranks)
        )

    monkeypatch.setattr(dist, "new_group", group)
    monkeypatch.setattr(
        dist,
        "new_subgroups_by_enumeration",
        lambda rank_lists, **kwargs: (
            next((group(ranks) for ranks in rank_lists if world.rank in ranks), None),
            [],
        ),
    )
    # Keep the test's fake groups out of the bridge's process-wide caches.
    monkeypatch.setattr(BridgeCommunicator, "_broadcast_pg_cache", {})
    monkeypatch.setattr(BridgeCommunicator, "_bridge_pg_cache", {})
    return world


@pytest.mark.parametrize(
    "tp,pp,cp,weight,dp",
    [
        (1, 1, 8, 1, 1),
        (1, 1, 1, 4, 1),
        (1, 1, 1, 1, 4),
        (1, 1, 8, 4, 1),
        (1, 1, 128, 64, 1),
        (2, 2, 2, 4, 2),
        (1, 2, 4, 6, 2),
        (1, 1, 3, 4, 1),
    ],
)
def test_mimo_groups_match_stock_weight_and_token_ownership(
    rank_world: SimpleNamespace, tp: int, pp: int, cp: int, weight: int, dp: int
) -> None:
    size = tp * pp * lcm(cp, weight) * dp
    spec = ModuleGridSpec(
        "language",
        size,
        tp=tp,
        pp=pp,
        cp=cp,
        gtp_remat=weight,
        ep=2,
        expt_gtp_remat=2,
        rank_offset=4,
        gtp_remat_fold_cp=True,
    )
    layout = GTPParallelLayout(size, tp, pp, cp, weight, rank_offset=4)
    expected = {
        "cp": layout.data.get_ranks("cp"),
        "dp_gtp_remat": layout.data.get_ranks("dp-gtp_remat"),
        "dp_cp_gtp_remat": layout.data.get_ranks("dp-cp-gtp_remat"),
        "gtp_remat": layout.weights.get_ranks("gtp_remat"),
        "dp_cp": layout.weights.get_ranks("dp"),
        "mp": layout.weights.get_ranks("tp-gtp_remat-pp"),
    }
    assert spec.dp == dp
    for rank in range(4, 4 + size):
        rank_world.rank = rank
        grid = _build_grid(spec)
        pgc = pg_collection_from_grid(grid)
        for field, groups in expected.items():
            assert getattr(pgc, field).ranks == next(ranks for ranks in groups if rank in ranks)
        assert pgc.intra_dp_cp is pgc.dp_cp
        assert get_data_lane_rank(pgc) == pgc.dp_gtp_remat.ranks.index(rank)
        assert pgc.ep.size() == 2
        assert pgc.expt_gtp_remat.size() == 2
        assert pgc.expt_dp.size() == size // (pp * 4)


@pytest.mark.parametrize("cp,weight", [(8, 1), (1, 4), (8, 4), (2, 4), (4, 6)])
@pytest.mark.parametrize("encoder_lanes_per_sample", [1, 2])
def test_bridge_routes_each_sample_to_all_context_partitions(
    rank_world: SimpleNamespace, cp: int, weight: int, encoder_lanes_per_sample: int
) -> None:
    tp, pp, dp = 2, 2, 2
    size = tp * pp * lcm(cp, weight) * dp
    layout = GTPParallelLayout(size, tp, pp, cp, weight)
    encoder_size = layout.sample_size * encoder_lanes_per_sample
    language_spec = ModuleGridSpec(
        "language",
        size,
        tp=tp,
        pp=pp,
        cp=cp,
        gtp_remat=weight,
        rank_offset=encoder_size,
        gtp_remat_fold_cp=True,
    )
    for rank in range(encoder_size, encoder_size + size // pp):
        rank_world.rank = rank
        # Bridge caches are rank-local in production; simulate a fresh rank each time.
        BridgeCommunicator._broadcast_pg_cache.clear()
        BridgeCommunicator._bridge_pg_cache.clear()
        src = _build_grid(ModuleGridSpec("images", encoder_size))
        dest = _build_grid(language_spec)
        pgc = pg_collection_from_grid(dest)
        sample_rank = get_data_lane_rank(pgc)
        bridge = BridgeCommunicator(src, dest)
        assert bridge.dest_cp_size == cp
        assert len(bridge.dest_tp_leaders) == layout.sample_size
        assert len(bridge.dest_grid_broadcast_ranks) == tp * cp
        assert bridge.dest_local_leader_rank == bridge.dest_tp_leaders[sample_rank]
        leader_info = bridge.comm_map[bridge.dest_local_leader_rank]
        assert leader_info.recv_from_ranks == list(
            range(
                sample_rank * encoder_lanes_per_sample, (sample_rank + 1) * encoder_lanes_per_sample
            )
        )
        if cp > 1 and pgc.tp.rank() == 0:
            assert bridge.dest_cp_reduce_pg is pgc.cp
        else:
            assert bridge.dest_cp_reduce_pg is None


def test_cp_extension_keeps_mimo_weight_and_optimizer_groups(rank_world: SimpleNamespace) -> None:
    memberships = []
    for cp in (2, 4, 8):
        rank_world.rank = 11
        grid = _build_grid(
            ModuleGridSpec("language", 16, cp=cp, gtp_remat=4, gtp_remat_fold_cp=True)
        )
        pgc = pg_collection_from_grid(grid)
        memberships.append((pgc.gtp_remat.ranks, pgc.dp_cp.ranks, pgc.mp.ranks))
    assert memberships[0] == memberships[1] == memberships[2]
