# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-distributed (8-GPU, no mocks) tests for the hetero MIMO grid topology.

Layout under test: encoder grid tp=2,dp=2 at ranks 0-3, language grid tp=2,pp=2 at ranks 4-7.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo.training.topology import (
    ModuleGridSpec,
    _get_pg_options,
    _validate_grid_layout,
    create_topology,
)
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.parallel_state import default_embedding_ranks
from tests.unit_tests.test_utilities import Utils

ENCODER = "images"


@pytest.fixture
def mock_nccl_options(monkeypatch):
    monkeypatch.setattr(
        dist,
        "ProcessGroupNCCL",
        SimpleNamespace(Options=lambda **kwargs: SimpleNamespace(**kwargs)),
    )


def _specs():
    return [
        ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, rank_offset=0),
        ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2, pp=2, rank_offset=4),
    ]


def _gtp_specs():
    return [
        ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, rank_offset=0),
        ModuleGridSpec(
            name=MIMO_LANGUAGE_MODULE_KEY,
            num_ranks=4,
            tp=2,
            gtp_remat=2,
            ep=2,
            expt_gtp_remat=2,
            rank_offset=4,
        ),
    ]


class TestModuleGridSpecResolution:
    def test_derived_dims_resolve_to_concrete_ints(self):
        # num_ranks=4,tp=2 with default expt_tp=1: dp=2, expt_dp=4.
        spec = ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2)
        assert isinstance(spec.dp, int) and spec.dp == 2
        assert spec.expt_tp == 1
        assert isinstance(spec.expt_dp, int) and spec.expt_dp == 4

    def test_explicit_expert_dims_resolve_correctly(self):
        # num_ranks=4,tp=2,ep=2,expt_tp=2: expt_dp = 4//(2*2*1) = 1.
        spec = ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, ep=2, expt_tp=2)
        assert isinstance(spec.expt_tp, int) and spec.expt_tp == 2
        assert isinstance(spec.expt_dp, int) and spec.expt_dp == 1

    def test_indivisible_dense_raises(self):
        with pytest.raises(ValueError):
            ModuleGridSpec(name=ENCODER, num_ranks=4, tp=3)

    def test_indivisible_expert_raises(self):
        with pytest.raises(ValueError):
            ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, ep=3, expt_tp=2)


@pytest.mark.parametrize(
    ("group_name", "requested", "enabled"),
    [
        ("tp", ["tp"], True),
        ("gtp_remat", ["gtp_remat"], True),
        ("ep", ["ep"], True),
        ("expt_gtp_remat", ["expt_gtp_remat"], True),
        ("mp", ["tp"], False),
        ("cp", ["cp"], True),
        ("tp", None, False),
    ],
)
def test_high_priority_pg_options(mock_nccl_options, group_name, requested, enabled):
    options = _get_pg_options(group_name, requested)

    assert (options is not None) is enabled
    if options is not None:
        assert options.is_high_priority_stream


def test_high_priority_options_are_wired_to_each_module_grid(monkeypatch, mock_nccl_options):
    class FakeGrid:
        instances = []

        def __init__(self, *args, **kwargs):
            self.calls = []
            self.instances.append(self)

        def register_view(self, *args, **kwargs):
            pass

        def create_pg(self, dims, view=None, pg_options=None):
            self.calls.append((tuple(dims), view, pg_options))

    monkeypatch.setattr("examples.mimo.training.topology.HyperCommGrid", FakeGrid)
    monkeypatch.setattr("examples.mimo.training.topology._validate_grid_layout", lambda grids: None)
    monkeypatch.setattr(
        "examples.mimo.training.topology.pg_collection_from_grid",
        lambda grid, is_language, high_priority_stream_groups: object(),
    )
    monkeypatch.setattr(
        "examples.mimo.training.topology.build_schedule_pg_collection",
        lambda grids, module_pgs, language_name: object(),
    )

    create_topology(_specs(), ["tp", "gtp_remat_dp_cp", "ep_tp", "tp_ep_gtp_remat_pp"])

    expected = {
        (("tp",), None),
        (("cp", "gtp_remat", "dp"), None),
        (("expt_tp",), "expert"),
        (("expt_tp", "ep", "expt_gtp_remat", "pp"), "expert"),
    }
    assert len(FakeGrid.instances) == 2
    for grid in FakeGrid.instances:
        enabled = {(dims, view) for dims, view, options in grid.calls if options is not None}
        assert enabled == expected
        assert all(options.is_high_priority_stream for _, _, options in grid.calls if options)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestHeteroTopology:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_grids_partition_world(self):
        topo = create_topology(_specs())
        try:
            encoder_ranks = set(range(0, 4))
            llm_ranks = set(range(4, 8))
            assert encoder_ranks & llm_ranks == set()
            assert encoder_ranks | llm_ranks == set(range(dist.get_world_size()))
            assert topo.grids[ENCODER].rank_offset == 0
            assert topo.grids[MIMO_LANGUAGE_MODULE_KEY].rank_offset == 4
        finally:
            topo.destroy()

    def test_pgc_group_sizes(self):
        topo = create_topology(_specs())
        try:
            rank = dist.get_rank()
            if rank < 4:
                pgc = topo.module_pgs[ENCODER]
                assert pgc.tp.size() == 2
                assert pgc.pp.size() == 1
                assert pgc.dp.size() == 2
                assert pgc.dp_cp.size() == 2
            else:
                pgc = topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
                assert pgc.tp.size() == 2
                assert pgc.pp.size() == 2
                assert pgc.dp.size() == 1
                assert pgc.dp_cp.size() == 1
        finally:
            topo.destroy()

    def test_gtp_pgc_group_sizes(self):
        topo = create_topology(_gtp_specs())
        try:
            rank = dist.get_rank()
            pgc = (
                topo.module_pgs[ENCODER] if rank < 4 else topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
            )
            if rank < 4:
                assert pgc.gtp_remat.size() == 1
                assert pgc.expt_gtp_remat.size() == 1
            else:
                assert pgc.tp.size() == 2
                assert pgc.gtp_remat.size() == 2
                assert pgc.dp.size() == 1
                assert pgc.dp_cp_gtp_remat.size() == 2
                assert pgc.mp.size() == 4
                assert pgc.expt_tp.size() == 1
                assert pgc.ep.size() == 2
                assert pgc.expt_gtp_remat.size() == 2
                assert pgc.expt_dp.size() == 1
                assert pgc.expt_dp_gtp_remat.size() == 2
                assert pgc.tp_ep_pp_with_egtp_remat.size() == 4
        finally:
            topo.destroy()

    def test_embedding_groups(self):
        # Language grid is tp=2,pp=2 at ranks 4-7: each PP group is [first,last] (size 2),
        # so first/last-stage ranks get a 2-rank .embd and the first stage gets .pos_embd.
        topo = create_topology(_specs())
        try:
            rank = dist.get_rank()
            if rank < 4:
                pgc = topo.module_pgs[ENCODER]
                assert pgc.embd is None
                assert pgc.pos_embd is None
            else:
                pgc = topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
                pp_ranks = dist.get_process_group_ranks(pgc.pp)
                pp_rank = pgc.pp.rank()
                expected_embd = len(default_embedding_ranks(pp_ranks))
                assert pgc.embd is not None
                assert pgc.embd.size() == expected_embd
                if pp_rank == 0:
                    assert pgc.pos_embd is not None
                    assert pgc.pos_embd.size() == 1
                else:
                    assert pgc.pos_embd is None
        finally:
            topo.destroy()

    def test_validate_rejects_overlapping_not_equal(self):
        # Illegal: encoder spans 0-3, llm spans 2-5 (overlap, not equal, not disjoint).
        a = HyperCommGrid([2, 2], ["tp", "dp"], rank_offset=0, backend="nccl")
        b = HyperCommGrid([2, 2], ["tp", "dp"], rank_offset=2, backend="nccl")
        try:
            with pytest.raises(ValueError, match="disjoint"):
                _validate_grid_layout({ENCODER: a, MIMO_LANGUAGE_MODULE_KEY: b})
        finally:
            a.destroy()
            b.destroy()

    def test_validate_rejects_gap_in_world_coverage(self):
        # Illegal: encoder spans 0-3, llm spans 4-7 leaves nothing uncovered, so instead
        # use disjoint grids that fail to span the full 8-rank world (ranks 6-7 uncovered).
        a = HyperCommGrid([2, 1], ["tp", "dp"], rank_offset=0, backend="nccl")
        b = HyperCommGrid([2, 1], ["tp", "dp"], rank_offset=4, backend="nccl")
        try:
            with pytest.raises(ValueError, match="partition the world"):
                _validate_grid_layout({ENCODER: a, MIMO_LANGUAGE_MODULE_KEY: b})
        finally:
            a.destroy()
            b.destroy()
