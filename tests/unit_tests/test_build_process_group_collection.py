# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""build_process_group_collection() and the collection that initialize_model_parallel() returns."""

import contextlib
import dataclasses
from unittest import mock

import pytest
import torch

import megatron.core.parallel_state as ps
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

FIELDS = [field.name for field in dataclasses.fields(ProcessGroupCollection)]

# (id, arguments shared by both functions, number the world size must be divisible by)
GRIDS = [
    ("tp2", dict(tensor_model_parallel_size=2), 2),
    ("tp2_pp2", dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2), 4),
    (
        "tp2_pp2_tp_pp_dp_order",
        dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, order="tp-cp-ep-pp-dp"),
        4,
    ),
    (
        "cp2_ep2_etp1",
        dict(context_parallel_size=2, expert_model_parallel_size=2, expert_tensor_parallel_size=1),
        2,
    ),
    (
        "hierarchical_cp",
        dict(context_parallel_size=4, hierarchical_context_parallel_sizes=[2, 2]),
        4,
    ),
    ("tp2_gtp_remat2", dict(tensor_model_parallel_size=2, gtp_remat_size=2), 4),
    ("ep2_dist_opt2", dict(expert_model_parallel_size=2, num_distributed_optimizer_instances=2), 4),
    ("dynamic_cp2", dict(context_parallel_size=2, dynamic_context_parallel=True), 2),
]
GRID_IDS = [grid[0] for grid in GRIDS]


def _require_divisible(num_ranks):
    if Utils.world_size % num_ranks != 0:
        pytest.skip(f"needs a world size divisible by {num_ranks}, got {Utils.world_size}")


def _ranks(value):
    """Global ranks of a group, of each group in a list, or None."""
    if value is None:
        return None
    if isinstance(value, list):
        return [_ranks(group) for group in value]
    return torch.distributed.get_process_group_ranks(value)


def _parallel_state_globals():
    """Every module-level parallel_state global, with a copy of the contents of dicts."""
    return {
        name: (value, dict(value) if isinstance(value, dict) else None)
        for name, value in vars(ps).items()
        if name.startswith("_") and name.isupper()
    }


def _changed_globals(before):
    after = _parallel_state_globals()
    assert sorted(after) == sorted(before)
    return [
        name
        for name, (value, contents) in before.items()
        if after[name][0] is not value or after[name][1] != contents
    ]


@contextlib.contextmanager
def _forbid_parallel_state_accessors():
    """Make every parallel_state accessor raise while the grid is built."""
    pure_helpers = {"get_nccl_options", "get_valid_dynamic_context_parallel_group_sizes"}
    names = [
        name
        for name, value in vars(ps).items()
        if callable(value)
        and getattr(value, "__module__", None) == ps.__name__
        and (name.startswith(("get_", "is_")) or name == "model_parallel_is_initialized")
        and name not in pure_helpers
    ]
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(
                mock.patch.object(
                    ps, name, side_effect=AssertionError(f"read parallel_state.{name}")
                )
            )
        yield


def _destroy_groups(pg_collection, extras):
    groups = {}
    values = [getattr(pg_collection, name) for name in FIELDS]
    values += [getattr(extras, field.name) for field in dataclasses.fields(extras)]
    for value in values:
        if isinstance(value, dict):
            value = list(value.values())
        for group in value if isinstance(value, list) else [value]:
            if isinstance(group, torch.distributed.ProcessGroup):
                groups[id(group)] = group
    for group in groups.values():
        torch.distributed.destroy_process_group(group)


def _initialize(**kwargs):
    Utils.destroy_model_parallel()
    Utils.initialize_distributed()
    pg_collection = ps.initialize_model_parallel(**kwargs)
    Utils.inited = True
    return pg_collection


@pytest.mark.parametrize("grid_id,kwargs,num_ranks", GRIDS, ids=GRID_IDS)
def test_initialize_model_parallel_returns_registered_groups(grid_id, kwargs, num_ranks):
    _require_divisible(num_ranks)
    pg_collection = _initialize(**kwargs)
    try:
        assert isinstance(pg_collection, ProcessGroupCollection)
        # Every field is set; None means this rank is not a member or the group is not built.
        assert sorted(vars(pg_collection)) == sorted(FIELDS)
        assert pg_collection.dp_cp_ag is None and pg_collection.expt_dp_ag is None
        shim = ProcessGroupCollection.use_mpu_process_groups()
        for name in FIELDS:
            assert getattr(pg_collection, name) is getattr(shim, name), name
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("grid_id,kwargs,num_ranks", GRIDS, ids=GRID_IDS)
def test_build_matches_registered_groups_without_parallel_state(grid_id, kwargs, num_ranks):
    _require_divisible(num_ranks)
    # Register a grid with a different layout, so that a parallel_state read changes the ranks.
    _initialize()
    before = _parallel_state_globals()
    with _forbid_parallel_state_accessors():
        pg_collection, extras = ps.build_process_group_collection(**kwargs)
    try:
        assert _changed_globals(before) == []
        assert sorted(vars(pg_collection)) == sorted(FIELDS)

        registered = _initialize(**kwargs)
        for name in FIELDS:
            built, expected = getattr(pg_collection, name), getattr(registered, name)
            assert _ranks(built) == _ranks(expected), name
            assert built is None or built is not expected, name
        assert _ranks(extras.tp_dp) == _ranks(ps.get_tensor_and_data_parallel_group())
        assert _ranks(extras.intra_dp_cp_gtp_remat) == _ranks(
            ps.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True)
        )
        assert _ranks(extras.dp_gloo) == _ranks(ps._DATA_PARALLEL_GROUP_GLOO)
        assert _ranks(extras.expt_dp_gloo) == _ranks(ps._EXPERT_DATA_PARALLEL_GROUP_GLOO)
        assert extras.tp_global_ranks == ps._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
        assert extras.pp_global_ranks == ps._PIPELINE_GLOBAL_RANKS
        assert extras.dp_global_ranks == ps._DATA_PARALLEL_GLOBAL_RANKS
        assert extras.dp_cp_global_ranks == ps._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
        assert extras.cp_global_ranks == ps._CONTEXT_PARALLEL_GLOBAL_RANKS
        assert extras.embd_global_ranks == ps._EMBEDDING_GLOBAL_RANKS
        assert extras.ep_global_ranks == ps._EXPERT_MODEL_PARALLEL_RANKS
        assert sorted(extras.dynamic_dp_cp) == sorted(ps._DYNAMIC_DP_CP_GROUPS)
        for size, group in extras.dynamic_dp_cp.items():
            assert _ranks(group) == _ranks(ps._DYNAMIC_DP_CP_GROUPS[size])
    finally:
        Utils.destroy_model_parallel()
        _destroy_groups(pg_collection, extras)


def test_build_rank_window():
    _require_divisible(4)
    window = Utils.world_size // 2
    _initialize()
    before = _parallel_state_globals()
    pg_collection, extras = ps.build_process_group_collection(
        tensor_model_parallel_size=2, rank_offset=window, local_world_size=window
    )
    try:
        assert _changed_globals(before) == []
        rank = torch.distributed.get_rank()
        if rank < window:
            # Outside the window this rank is a member of no group.
            assert [name for name in FIELDS if getattr(pg_collection, name) is not None] == []
            empty = {field.name: None for field in dataclasses.fields(extras)}
            empty["dynamic_dp_cp"] = {}
            assert extras == ps.StandardGridExtras(**empty)
        else:
            tp_ranks = [rank - rank % 2, rank - rank % 2 + 1]
            assert _ranks(pg_collection.tp) == tp_ranks
            assert extras.tp_global_ranks == tp_ranks
            assert _ranks(pg_collection.mp) == tp_ranks
            assert _ranks(pg_collection.pp) == [rank]
            assert _ranks(pg_collection.dp) == list(range(window + rank % 2, 2 * window, 2))
    finally:
        Utils.destroy_model_parallel()
        _destroy_groups(pg_collection, extras)


def test_initialize_only_arguments():
    _require_divisible(4)
    Utils.destroy_model_parallel()
    Utils.initialize_distributed()
    with pytest.raises(ValueError, match="Cannot set both"):
        ps.initialize_model_parallel(
            context_parallel_size=2, hybrid_context_parallel=True, dynamic_context_parallel=True
        )
    ps.destroy_model_parallel()

    with pytest.warns(DeprecationWarning, match="hybrid_context_parallel is deprecated"):
        pg_collection = _initialize(
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            context_parallel_size=2,
            hybrid_context_parallel=True,
        )
    try:
        assert ps.get_virtual_pipeline_model_parallel_world_size() == 2
        assert ps.get_virtual_pipeline_model_parallel_rank() == 0
        dp_cp_size = pg_collection.dp_cp.size()
        sizes = ps.get_valid_dynamic_context_parallel_group_sizes(dp_cp_size)
        assert sorted(ps._DYNAMIC_DP_CP_GROUPS) == [size for size in sizes if size < dp_cp_size]
    finally:
        Utils.destroy_model_parallel()


def test_initialize_twice_raises_already_initialized():
    _require_divisible(2)
    _initialize(tensor_model_parallel_size=2)
    try:
        with pytest.raises(AssertionError, match="data parallel group is already initialized"):
            ps.initialize_model_parallel(tensor_model_parallel_size=2)
    finally:
        Utils.destroy_model_parallel()
