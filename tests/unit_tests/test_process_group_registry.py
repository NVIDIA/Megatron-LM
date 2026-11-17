# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""register_process_groups(): the process groups that the parallel_state accessors return."""

import dataclasses

import pytest
import torch

import megatron.core.parallel_state as ps
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

FIELDS = [field.name for field in dataclasses.fields(ProcessGroupCollection)]

# The parallel_state accessor of each collection field. dp_cp_ag and expt_dp_ag have none.
ACCESSORS = {
    "tp": lambda: ps.get_tensor_model_parallel_group(check_initialized=False),
    "pp": lambda: ps.get_pipeline_model_parallel_group(check_initialized=False),
    "mp": lambda: ps.get_model_parallel_group(check_initialized=False),
    "embd": lambda: ps.get_embedding_group(check_initialized=False),
    "pos_embd": lambda: ps.get_position_embedding_group(check_initialized=False),
    "cp": lambda: ps.get_context_parallel_group(check_initialized=False),
    "dynamic_cp": ps.get_dynamic_data_context_parallel_group_map,
    "dynamic_tp_cp": ps.get_dynamic_tensor_data_context_parallel_group_map,
    "tp_cp": lambda: ps.get_tensor_and_context_parallel_group(check_initialized=False),
    "hcp": lambda: ps.get_hierarchical_context_parallel_groups(check_initialized=False),
    "ep": lambda: ps.get_expert_model_parallel_group(check_initialized=False),
    "expt_tp": lambda: ps.get_expert_tensor_parallel_group(check_initialized=False),
    "tp_ep": lambda: ps.get_expert_tensor_and_model_parallel_group(check_initialized=False),
    "tp_ep_pp": lambda: ps.get_expert_tensor_model_pipeline_parallel_group(check_initialized=False),
    "tp_ep_pp_with_egtp_remat": lambda: ps.get_expert_tensor_model_pipeline_parallel_group(
        check_initialized=False, with_egtp_remat=True
    ),
    "tp_dp_cp": lambda: ps.get_tensor_and_data_parallel_group(
        check_initialized=False, with_context_parallel=True
    ),
    "dp": lambda: ps.get_data_parallel_group(with_gtp_remat=False),
    "dp_gtp_remat": lambda: ps.get_data_parallel_group(),
    "dp_cp": lambda: ps.get_data_parallel_group(with_context_parallel=True, with_gtp_remat=False),
    "dp_cp_gtp_remat": lambda: ps.get_data_parallel_group(with_context_parallel=True),
    "gtp_remat": lambda: ps.get_gtp_weight_remat_group(check_initialized=False),
    "expt_gtp_remat": lambda: ps.get_expert_gtp_weight_remat_group(check_initialized=False),
    "expt_dp": lambda: ps.get_expert_data_parallel_group(
        check_initialized=False, with_gtp_remat=False
    ),
    "expt_dp_gtp_remat": lambda: ps.get_expert_data_parallel_group(check_initialized=False),
    "intra_dp_cp": lambda: ps.get_data_parallel_group(
        with_context_parallel=True, with_gtp_remat=False, partial_data_parallel=True
    ),
    "intra_expt_dp": lambda: ps.get_expert_data_parallel_group(
        check_initialized=False, with_gtp_remat=False, partial_expert_data_parallel=True
    ),
    "intra_dp_cp_gloo": lambda: ps.get_data_parallel_group_gloo(
        with_context_parallel=True, partial_data_parallel=True, check_initialized=False
    ),
    "intra_expt_dp_gloo": lambda: ps.get_expert_data_parallel_group_gloo(
        partial_expert_data_parallel=True, check_initialized=False
    ),
    "inter_dist_opt": lambda: ps.get_inter_distributed_optimizer_instance_group(
        check_initialized=False
    ),
    "intra_dist_opt": lambda: ps.get_intra_distributed_optimizer_instance_group(
        check_initialized=False
    ),
}


def _assert_same_groups(value, expected, name):
    """The same group objects; a dict field is compared key by key."""
    if isinstance(expected, dict):
        assert sorted(value) == sorted(expected), name
        assert all(value[key] is group for key, group in expected.items()), name
    else:
        assert value is expected, name


def _assert_parallel_state_returns(pg_collection):
    """Every accessor and use_mpu_process_groups() return the groups of pg_collection."""
    assert ps.get_registered_process_groups() is pg_collection
    for name, accessor in ACCESSORS.items():
        _assert_same_groups(accessor(), getattr(pg_collection, name), name)
    shim = ProcessGroupCollection.use_mpu_process_groups()
    for name in ACCESSORS:
        _assert_same_groups(getattr(shim, name), getattr(pg_collection, name), name)
    assert shim.dp_cp_ag is None and shim.expt_dp_ag is None


def test_accessor_table_covers_every_field():
    assert sorted(ACCESSORS) == sorted(set(FIELDS) - {"dp_cp_ag", "expt_dp_ag"})


# CPU tests: fake groups that know their ranks, and no collectives.


class _Group:
    def __init__(self, ranks, rank):
        self.ranks = list(ranks)
        self.local_rank = self.ranks.index(rank) if rank in self.ranks else -1

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.local_rank


def _empty_like(value):
    """Return the uninitialized form of a parallel_state module global.

    Keyed registries (a dict whose values are lists) keep their keys with empty lists, so code
    that indexes them by a fixed key still works; other dicts become empty and the rest None.
    """
    if isinstance(value, dict):
        if value and all(isinstance(v, list) for v in value.values()):
            return {key: [] for key in value}
        return {}
    return None


@pytest.fixture
def fake_world(monkeypatch):
    """Run parallel_state on fake groups; every change to its module state is undone."""
    for name, value in list(vars(ps).items()):
        if name.startswith("_") and name.isupper():
            monkeypatch.setattr(ps, name, _empty_like(value))
    monkeypatch.setattr(ps, "_global_process_group_list", ps._global_process_group_list)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(ps, "get_nccl_options", lambda *args: None)

    def use_rank(world_size, rank):
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)
        monkeypatch.setattr(
            torch.distributed,
            "get_rank",
            lambda group=None: rank if group is None else group.rank(),
        )
        monkeypatch.setattr(ps, "create_group", lambda ranks, **kwargs: _Group(ranks, rank))

    return use_rank


# (world size, rank, initialize_model_parallel arguments)
FAKE_GRIDS = [
    (8, 5, dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)),
    (
        8,
        6,
        dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, order="tp-cp-ep-pp-dp"),
    ),
    (8, 3, dict(context_parallel_size=4, hierarchical_context_parallel_sizes=[2, 2])),
    (8, 7, dict(expert_model_parallel_size=2, num_distributed_optimizer_instances=2)),
    (8, 2, dict(tensor_model_parallel_size=2, gtp_remat_size=2)),
    (8, 1, dict(context_parallel_size=2, dynamic_context_parallel=True)),
    (
        8,
        6,
        dict(tensor_model_parallel_size=2, context_parallel_size=2, dynamic_context_parallel=True),
    ),
]


@pytest.mark.parametrize("world_size,rank,kwargs", FAKE_GRIDS)
def test_initialize_model_parallel_registers_its_collection(fake_world, world_size, rank, kwargs):
    fake_world(world_size, rank)
    pg_collection = ps.initialize_model_parallel(**kwargs)
    _assert_parallel_state_returns(pg_collection)


@pytest.mark.parametrize("world_size,rank,kwargs", FAKE_GRIDS)
def test_destroy_clears_every_registered_value(fake_world, world_size, rank, kwargs):
    fake_world(world_size, rank)
    before = {name: value for name, value in vars(ps).items() if name.isupper()}
    ps.initialize_model_parallel(**kwargs)
    registered = [
        name
        for name, value in vars(ps).items()
        if name.isupper() and name.startswith("_") and value is not before[name]
    ]
    assert "_REGISTERED_PROCESS_GROUPS" in registered

    ps.destroy_model_parallel()

    for name in registered:
        assert getattr(ps, name) in (None, {}), name
    assert ps.get_registered_process_groups() is None
    assert ProcessGroupCollection.use_mpu_process_groups(["hcp"]).hcp is None
    with pytest.raises(AssertionError, match="intra partial data parallel group"):
        ProcessGroupCollection.use_mpu_process_groups(["intra_dp_cp"])
    for rank_accessor in (
        ps.get_tensor_model_parallel_src_rank,
        ps.get_model_parallel_src_rank,
        ps.get_data_parallel_src_rank,
        ps.get_pipeline_model_parallel_first_rank,
        ps.get_expert_model_parallel_src_rank,
    ):
        with pytest.raises(AssertionError, match="not initialized"):
            rank_accessor()
    assert not ps.is_rank_in_embedding_group()


def _ranks_along(rank, stride, size):
    """The ranks that differ from rank only along the grid dimension with this stride and size."""
    first = rank - (rank // stride % size) * stride
    return [first + i * stride for i in range(size)]


def _fake_collection(rank):
    """This rank's groups on a dense PP2 x TP2 x DP2 grid of 8 ranks, PP varying fastest.

    The default order of initialize_model_parallel() varies TP fastest, so these groups have
    other ranks than the default grid of the same sizes.
    """
    pp, tp, dp = (_ranks_along(rank, stride, 2) for stride in (1, 2, 4))
    tp_pp = sorted({t + p - rank for t in tp for p in pp})
    tp_dp = sorted({t + d - rank for t in tp for d in dp})
    groups = {
        "tp": _Group(tp, rank),
        "pp": _Group(pp, rank),
        "dp": _Group(dp, rank),
        "mp": _Group(tp_pp, rank),
        "cp": _Group([rank], rank),
        "ep": _Group([rank], rank),
        "tp_dp_cp": _Group(tp_dp, rank),
        "intra_dist_opt": _Group(range(8), rank),
        "dp_gloo": _Group(dp, rank),
    }
    # Word embeddings on the first and last pipeline stages, position embeddings on the first.
    embd = _Group(pp, rank)
    pos_embd = _Group(pp[:1], rank) if rank == pp[0] else None
    return ProcessGroupCollection(
        tp=groups["tp"],
        pp=groups["pp"],
        mp=groups["mp"],
        embd=embd,
        pos_embd=pos_embd,
        cp=groups["cp"],
        dynamic_cp={},
        dynamic_tp_cp={},
        tp_cp=groups["tp"],
        hcp=None,
        ep=groups["ep"],
        expt_tp=groups["tp"],
        tp_ep=groups["tp"],
        tp_ep_pp=groups["mp"],
        tp_ep_pp_with_egtp_remat=groups["mp"],
        tp_dp_cp=groups["tp_dp_cp"],
        dp=groups["dp"],
        dp_gtp_remat=groups["dp"],
        dp_cp=groups["dp"],
        dp_cp_gtp_remat=groups["dp"],
        dp_cp_ag=None,
        gtp_remat=None,
        expt_gtp_remat=None,
        expt_dp=groups["dp"],
        expt_dp_gtp_remat=groups["dp"],
        expt_dp_ag=None,
        intra_dp_cp=groups["dp"],
        intra_expt_dp=groups["dp"],
        intra_dp_cp_gloo=groups["dp_gloo"],
        intra_expt_dp_gloo=groups["dp_gloo"],
        inter_dist_opt=None,
        intra_dist_opt=groups["intra_dist_opt"],
    )


@pytest.mark.parametrize("rank", range(8))
def test_registered_collection_sets_groups_ranks_and_sizes(fake_world, rank):
    fake_world(8, rank)
    pg_collection = _fake_collection(rank)
    ps.register_process_groups(pg_collection)
    _assert_parallel_state_returns(pg_collection)

    pp_ranks = pg_collection.pp.ranks
    stage = pg_collection.pp.rank()
    assert ps.get_pipeline_model_parallel_rank() == stage
    assert ps.get_pipeline_model_parallel_world_size() == 2
    assert ps.get_pipeline_model_parallel_first_rank() == pp_ranks[0]
    assert ps.get_pipeline_model_parallel_last_rank() == pp_ranks[1]
    assert ps.get_pipeline_model_parallel_next_rank() == pp_ranks[1 - stage]
    assert ps.get_pipeline_model_parallel_prev_rank() == pp_ranks[1 - stage]
    assert ps.is_pipeline_first_stage() == (stage == 0)
    assert ps.is_pipeline_last_stage() == (stage == 1)
    assert ps.is_rank_in_embedding_group()
    assert ps.is_rank_in_position_embedding_group() == (stage == 0)
    assert ps.get_tensor_model_parallel_rank() == pg_collection.tp.rank()
    assert ps.get_tensor_model_parallel_world_size() == 2
    assert ps.get_tensor_model_parallel_src_rank() == pg_collection.tp.ranks[0]
    assert ps.get_model_parallel_src_rank() == pg_collection.mp.ranks[0]
    assert ps.get_data_parallel_rank() == pg_collection.dp.rank()
    assert ps.get_data_parallel_world_size(with_context_parallel=True) == 2
    assert ps.get_data_parallel_src_rank() == pg_collection.dp.ranks[0]
    assert ps.get_data_parallel_src_rank(with_context_parallel=True) == pg_collection.dp.ranks[0]
    assert ps.get_context_parallel_global_ranks() == [rank]
    assert ps.get_context_parallel_world_size() == 1
    assert ps.get_expert_model_parallel_src_rank() == rank
    assert ps.get_expert_tensor_parallel_world_size() == 2
    # GTP_remat is off, so the partial data parallel group spanning it is intra_dp_cp.
    partial = ps.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True)
    assert partial is pg_collection.intra_dp_cp
    assert ps.get_expert_data_parallel_group(partial_expert_data_parallel=True) is (
        pg_collection.intra_expt_dp
    )
    # A collection has no field for these.
    assert ps.get_tensor_and_data_parallel_group(check_initialized=False) is None
    with pytest.raises(AssertionError, match="not initialized"):
        ps.get_data_parallel_group_gloo()
    with pytest.raises(AssertionError, match="not initialized"):
        ps.get_expert_data_parallel_group_gloo()


def test_size_and_rank_overrides_apply_to_a_registered_collection(fake_world):
    fake_world(8, 6)
    pg_collection = _fake_collection(6)
    pg_collection.expt_tp = None
    ps.register_process_groups(pg_collection)

    ps.set_tensor_model_parallel_world_size(4)
    ps.set_tensor_model_parallel_rank(3)
    ps.set_pipeline_model_parallel_world_size(8)
    ps.set_pipeline_model_parallel_rank(5)
    ps.set_data_parallel_rank(1)
    ps.set_expert_model_parallel_world_size(2)
    ps.set_expert_model_parallel_rank(1)
    assert ps.get_tensor_model_parallel_world_size() == 4
    assert ps.get_tensor_model_parallel_rank() == 3
    assert ps.get_pipeline_model_parallel_world_size() == 8
    assert ps.get_pipeline_model_parallel_rank() == 5
    assert ps.get_data_parallel_rank() == 1
    assert ps.get_expert_model_parallel_world_size() == 2
    assert ps.get_expert_model_parallel_rank() == 1
    # Without an expert tensor parallel group, the expert TP size and rank are the TP overrides.
    assert ps.get_expert_tensor_parallel_world_size() == 4
    assert ps.get_expert_tensor_parallel_rank() == 3
    # The overrides change what the size and rank accessors return, not the groups.
    assert ps.get_tensor_model_parallel_group() is pg_collection.tp
    assert ProcessGroupCollection.use_mpu_process_groups(["tp"]).tp is pg_collection.tp


def test_register_collection_with_unset_fields(fake_world):
    fake_world(8, 0)
    tp = _Group([0, 1], 0)
    dp = _Group([0, 2, 4, 6], 0)
    pg_collection = ProcessGroupCollection(
        tp=tp, dp=dp, dp_gtp_remat=dp, dp_cp=dp, dp_cp_gtp_remat=dp, intra_dp_cp=dp
    )
    ps.register_process_groups(pg_collection)

    assert ps.get_registered_process_groups() is pg_collection
    assert ps.get_tensor_model_parallel_group() is tp
    assert ps.get_data_parallel_group() is dp
    assert ps.get_tensor_model_parallel_src_rank() == 0
    assert ps.get_data_parallel_src_rank() == 0
    assert ps.get_dynamic_data_context_parallel_group_map() == {}
    assert ps.get_dynamic_tensor_data_context_parallel_group_map() == {}
    with pytest.raises(AssertionError, match="context parallel group is not initialized"):
        ps.get_context_parallel_group()
    with pytest.raises(AssertionError, match="Pipeline parallel group is not initialized"):
        ps.get_pipeline_model_parallel_first_rank()
    shim = ProcessGroupCollection.use_mpu_process_groups()
    assert shim.tp is tp and shim.dp is dp
    assert shim.cp is None and shim.pp is None and shim.hcp is None
    assert shim.dynamic_cp == {} and shim.dynamic_tp_cp == {}


def test_register_twice_raises_and_keeps_the_first_registration(fake_world):
    fake_world(8, 6)
    first = _fake_collection(6)
    ps.register_process_groups(first)
    with pytest.raises(RuntimeError, match="data parallel group is already initialized"):
        ps.register_process_groups(_fake_collection(6))
    _assert_parallel_state_returns(first)

    ps.destroy_model_parallel()
    # A rank that is a member of no group registers an empty collection.
    empty = ProcessGroupCollection()
    ps.register_process_groups(empty)
    with pytest.raises(RuntimeError, match="already registered"):
        ps.register_process_groups(ProcessGroupCollection())
    assert ps.get_registered_process_groups() is empty


def test_register_rejects_other_types(fake_world):
    fake_world(8, 6)
    with pytest.raises(TypeError, match="ProcessGroupCollection"):
        ps.register_process_groups({"tp": _Group([6, 7], 6)})
    with pytest.raises(TypeError, match="StandardGridExtras"):
        ps.register_process_groups(_fake_collection(6), extras={})
    assert ps.get_registered_process_groups() is None
    assert ps.get_tensor_model_parallel_group(check_initialized=False) is None


# GPU tests: real process groups on the default process group.


def _require_divisible(num_ranks):
    if Utils.world_size % num_ranks != 0:
        pytest.skip(f"needs a world size divisible by {num_ranks}, got {Utils.world_size}")


def _initialize(**kwargs):
    Utils.destroy_model_parallel()
    Utils.initialize_distributed()
    pg_collection = ps.initialize_model_parallel(**kwargs)
    Utils.inited = True
    return pg_collection


def _global_ranks(group):
    return torch.distributed.get_process_group_ranks(group)


def test_register_hyper_comm_grid_collection():
    _require_divisible(4)
    # A standard grid first, so that values left over from it would show.
    _initialize(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    Utils.destroy_model_parallel()

    # PP varies fastest, unlike in the default order of initialize_model_parallel().
    dp_size = Utils.world_size // 4
    grid = HyperCommGrid([2, 2, 1, 1, dp_size], ["pp", "tp", "cp", "ep", "dp"])
    pgs = ProcessGroupCollection(
        tp=grid.create_pg("tp"),
        pp=grid.create_pg("pp"),
        mp=grid.create_pg(["pp", "tp"]),
        cp=grid.create_pg("cp"),
        dynamic_cp={},
        dynamic_tp_cp={},
        tp_cp=grid.create_pg(["tp", "cp"]),
        hcp=None,
        ep=grid.create_pg("ep"),
        tp_ep=grid.create_pg(["tp", "ep"]),
        tp_ep_pp=grid.create_pg(["pp", "tp", "ep"]),
        tp_dp_cp=grid.create_pg(["tp", "cp", "dp"]),
        dp=grid.create_pg("dp"),
        dp_cp=grid.create_pg(["cp", "dp"]),
        dp_cp_ag=None,
        gtp_remat=None,
        expt_gtp_remat=None,
        expt_dp_ag=None,
        inter_dist_opt=None,
        intra_dist_opt=grid.create_pg(["pp", "tp", "cp", "ep", "dp"]),
    )
    pgs.expt_tp = pgs.tp
    pgs.tp_ep_pp_with_egtp_remat = pgs.tp_ep_pp
    pgs.dp_gtp_remat = pgs.dp
    pgs.dp_cp_gtp_remat = pgs.dp_cp
    pgs.intra_dp_cp = pgs.dp_cp
    pgs.expt_dp = pgs.dp
    pgs.expt_dp_gtp_remat = pgs.dp
    pgs.intra_expt_dp = pgs.dp
    # new_group is collective, so every rank creates every embedding group.
    rank = torch.distributed.get_rank()
    pgs.embd = pgs.pos_embd = None
    embedding_groups = []
    for stage_ranks in grid.get_rank_enum("pp"):
        embd = torch.distributed.new_group(stage_ranks)
        pos_embd = torch.distributed.new_group(stage_ranks[:1])
        if rank in stage_ranks:
            pgs.embd = embd
            embedding_groups.append(embd)
        if rank == stage_ranks[0]:
            pgs.pos_embd = pos_embd
            embedding_groups.append(pos_embd)

    ps.register_process_groups(pgs)
    try:
        _assert_parallel_state_returns(pgs)
        assert _global_ranks(pgs.pp) == [rank - rank % 2, rank - rank % 2 + 1]

        pp_ranks = _global_ranks(pgs.pp)
        stage = pgs.pp.rank()
        assert ps.get_pipeline_model_parallel_rank() == stage
        assert ps.get_pipeline_model_parallel_world_size() == 2
        assert ps.get_pipeline_model_parallel_first_rank() == pp_ranks[0]
        assert ps.get_pipeline_model_parallel_last_rank() == pp_ranks[1]
        assert ps.get_pipeline_model_parallel_next_rank() == pp_ranks[1 - stage]
        assert ps.get_pipeline_model_parallel_prev_rank() == pp_ranks[1 - stage]
        assert ps.is_pipeline_first_stage() == (stage == 0)
        assert ps.is_rank_in_embedding_group()
        assert ps.is_rank_in_position_embedding_group() == (stage == 0)
        assert ps.get_tensor_model_parallel_rank() == pgs.tp.rank()
        assert ps.get_tensor_model_parallel_world_size() == 2
        assert ps.get_tensor_model_parallel_src_rank() == _global_ranks(pgs.tp)[0]
        assert ps.get_model_parallel_src_rank() == _global_ranks(pgs.mp)[0]
        assert ps.get_data_parallel_world_size() == dp_size
        assert ps.get_data_parallel_rank() == pgs.dp.rank()
        assert ps.get_data_parallel_src_rank() == _global_ranks(pgs.dp)[0]
        assert ps.get_context_parallel_global_ranks() == [rank]
        assert ps.get_expert_model_parallel_src_rank() == rank

        # The registered groups communicate over their own ranks.
        value = torch.tensor([float(rank)], device="cuda")
        torch.distributed.all_reduce(value, group=ps.get_pipeline_model_parallel_group())
        assert value.item() == float(sum(pp_ranks))
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        ps.destroy_model_parallel()
        grid.destroy()
        for group in embedding_groups:
            torch.distributed.destroy_process_group(group)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(context_parallel_size=4, hierarchical_context_parallel_sizes=[2, 2]),
        dict(expert_model_parallel_size=2, num_distributed_optimizer_instances=2),
    ],
    ids=["hierarchical_cp", "ep2_dist_opt2"],
)
def test_destroy_clears_the_registration(kwargs):
    _require_divisible(4)
    pg_collection = _initialize(**kwargs)
    assert ps.get_registered_process_groups() is pg_collection
    intra_dp_cp_gloo = ps._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
    # Registered from the collection, which must carry the Gloo group the builder created.
    assert intra_dp_cp_gloo is not None
    assert intra_dp_cp_gloo is pg_collection.intra_dp_cp_gloo

    Utils.destroy_model_parallel()

    assert ps.get_registered_process_groups() is None
    assert ProcessGroupCollection.use_mpu_process_groups(["hcp"]).hcp is None
    with pytest.raises(AssertionError, match="intra partial data parallel group"):
        ProcessGroupCollection.use_mpu_process_groups(["intra_dp_cp"])
    for rank_accessor in (
        ps.get_tensor_model_parallel_src_rank,
        ps.get_model_parallel_src_rank,
        ps.get_data_parallel_src_rank,
        ps.get_pipeline_model_parallel_first_rank,
        ps.get_expert_model_parallel_src_rank,
    ):
        with pytest.raises(AssertionError, match="not initialized"):
            rank_accessor()
    assert not ps.is_rank_in_embedding_group()
    # destroy_model_parallel() destroys the Gloo groups it registered.
    assert ps._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO is None
    assert torch.distributed.distributed_c10d._world.pg_map.get(intra_dp_cp_gloo) is None


def test_register_builder_result_with_extras():
    _require_divisible(4)
    Utils.destroy_model_parallel()
    Utils.initialize_distributed()
    pg_collection, extras = ps.build_process_group_collection(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=2
    )
    ps.register_process_groups(pg_collection, extras=extras)
    Utils.inited = True
    try:
        _assert_parallel_state_returns(pg_collection)
        assert ps.get_tensor_and_data_parallel_group() is extras.tp_dp
        assert ps.get_data_parallel_group_gloo() is extras.dp_gloo
        assert ps.get_data_parallel_group_gloo(with_context_parallel=True) is extras.dp_cp_gloo
        assert ps.get_expert_data_parallel_group_gloo() is extras.expt_dp_gloo
        assert ps.get_pipeline_model_parallel_first_rank() == extras.pp_global_ranks[0]
    finally:
        Utils.destroy_model_parallel()
