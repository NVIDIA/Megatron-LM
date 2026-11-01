# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training bootstrap takes its process groups from the registry, not from the global grid.

pretrain() and initialize_megatron() build their collection from the one that
initialize_model_parallel() registers, parallel_state.get_registered_process_groups(), instead of
reading it with ProcessGroupCollection.use_mpu_process_groups(), which 0.23 removes with the
parallel_state accessors. The collection must be the one that use_mpu_process_groups() returned,
field by field. These tests compare the two on the standard grid, on collections that a framework
registers, and on grids set up without the registry, where bootstrap still reads the global grid,
now with a deprecation warning. They also cover the all-gather groups of
--create-all-gather-group, which are built from the collection in any rank order, the timeout
update of the run's groups, and the deprecation of the overrides that --lazy-mpu-init sets.
"""

import dataclasses
import re
import warnings
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.parallel_state import ParallelStateDeprecationWarning
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
    enable_strict_process_group_fallbacks,
)
from megatron.training import arguments, global_vars, initialize, training
from megatron.training.argument_utils import inference_cfg_container_from_args
from tests.unit_tests.test_utilities import Utils

FIELDS = {field.name for field in dataclasses.fields(ProcessGroupCollection)}
DATA_PARALLEL_FIELDS = ("dp", "dp_gtp_remat", "dp_cp", "dp_cp_gtp_remat", "intra_dp_cp")

# (id, initialize_model_parallel() arguments, number the world size must be divisible by)
GRIDS = [
    ("dp", dict(), 1),
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
    ("ep4", dict(expert_model_parallel_size=4), 4),
    (
        "hierarchical_cp",
        dict(context_parallel_size=4, hierarchical_context_parallel_sizes=[2, 2]),
        4,
    ),
    ("ep2_dist_opt2", dict(expert_model_parallel_size=2, num_distributed_optimizer_instances=2), 4),
    ("no_gloo", dict(tensor_model_parallel_size=2, create_gloo_process_groups=False), 2),
    ("dynamic_cp2", dict(context_parallel_size=2, dynamic_context_parallel=True), 2),
    (
        "tp2_dynamic_cp2",
        dict(tensor_model_parallel_size=2, context_parallel_size=2, dynamic_context_parallel=True),
        4,
    ),
]

# Grids with expert parallelism for the all-gather groups. Without a collection,
# create_all_gather_groups() lays the expert ranks out in the default order without expert GTP:
# with the tp-cp-ep-pp-dp order and both pipeline and expert data parallelism its expert groups
# differ from expt_dp, and with expert GTP they name ranks outside the world.
EXPERT_GRIDS = [
    ("ep2", dict(expert_model_parallel_size=2), 2),
    ("ep2_egtp2", dict(expert_model_parallel_size=2, expert_gtp_remat_size=2), 4),
    (
        "tp2_pp2_ep2_tp_pp_dp_order",
        dict(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            order="tp-cp-ep-pp-dp",
        ),
        8,
    ),
]


def _require_divisible(num_ranks):
    if Utils.world_size % num_ranks != 0:
        pytest.skip(f"needs a world size divisible by {num_ranks}, got {Utils.world_size}")


def _require_no_distributed():
    if torch.distributed.is_initialized():
        pytest.skip("registering fake groups would replace the groups of the distributed run")


@contextmanager
def _recorded_warnings():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record


def _of_category(record, category):
    return [str(warning.message) for warning in record if issubclass(warning.category, category)]


def _fallbacks(record):
    """The (owner, argument) of each ProcessGroupFallbackWarning in ``record``."""
    return [
        re.match(r"(.+) was called without `(\w+)`", message).groups()
        for message in _of_category(record, ProcessGroupFallbackWarning)
    ]


@pytest.fixture
def fresh_fallback_record(monkeypatch):
    """Forget which fallbacks warned, so that each test sees its own warnings."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _read_with_the_shim():
    """The collection that bootstrap built before: use_mpu_process_groups() on the global grid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParallelStateDeprecationWarning)
        return ProcessGroupCollection.use_mpu_process_groups()


def _assert_same_collection(actual, expected):
    """Both collections set every field, to the same groups over the same ranks; dict and list
    fields are compared group by group."""
    assert type(actual) is type(expected) is ProcessGroupCollection
    assert set(vars(actual)) == set(vars(expected)) == FIELDS
    for name in sorted(FIELDS):
        value, reference = vars(actual)[name], vars(expected)[name]
        if isinstance(reference, dict):
            assert isinstance(value, dict) and sorted(value) == sorted(reference), name
            pairs = [(value[key], group) for key, group in reference.items()]
        elif isinstance(reference, list):
            assert isinstance(value, list) and len(value) == len(reference), name
            pairs = list(zip(value, reference))
        else:
            pairs = [(value, reference)]
        for group, reference_group in pairs:
            assert group is reference_group, name
            if isinstance(reference_group, torch.distributed.ProcessGroup):
                ranks = torch.distributed.get_process_group_ranks(group)
                assert ranks == torch.distributed.get_process_group_ranks(reference_group), name


def _assert_bootstrap_matches_the_shim():
    registered = parallel_state.get_registered_process_groups()
    expected = _read_with_the_shim()

    built = initialize._registered_pg_collection("owner")

    _assert_same_collection(built, expected)
    # A new collection, with new maps: adding a group to it leaves the registered one unchanged.
    assert built is not registered
    assert built.dynamic_cp is not registered.dynamic_cp
    assert built.dynamic_tp_cp is not registered.dynamic_tp_cp
    registered_fields = dict(vars(registered))
    built.dp_cp_ag = built.dp_cp
    assert vars(registered) == registered_fields


class _Group:
    """A stand-in for a process group of a framework's grid; it has a size."""

    def __init__(self, size=2):
        self._size = size

    def size(self):
        return self._size


def _register(**groups):
    """Register a framework's collection of stand-in groups, without the grid's extras."""
    extras = parallel_state.StandardGridExtras(
        **{field.name: None for field in dataclasses.fields(parallel_state.StandardGridExtras)}
    )
    registered = ProcessGroupCollection(**groups)
    parallel_state.register_process_groups(registered, extras=extras)
    return registered


def _data_parallel_groups():
    return {name: _Group() for name in DATA_PARALLEL_FIELDS}


class TestFrameworkCollection:
    """A framework registers a collection that may leave fields unset."""

    def setup_method(self, method):
        _require_no_distributed()

    def teardown_method(self, method):
        parallel_state.destroy_model_parallel()

    @pytest.mark.parametrize(
        "extra_groups",
        [
            pytest.param(lambda: {}, id="data_parallel_only"),
            pytest.param(
                lambda: dict(tp=_Group(), pp=_Group(), ep=_Group(), hcp=[_Group(), _Group()]),
                id="model_parallel_and_hierarchical_cp",
            ),
            # The all-gather groups have no accessor, so the shim set them to None.
            pytest.param(
                lambda: dict(dp_cp_ag=_Group(), expt_dp_ag=_Group()), id="all_gather_groups"
            ),
            # With TP1 the runtime CP groups are the TP x runtime-CP groups.
            pytest.param(
                lambda: dict(tp=_Group(size=1), dynamic_cp={1: _Group(1), 2: _Group()}),
                id="dynamic_cp_tp1",
            ),
            pytest.param(
                lambda: dict(
                    tp=_Group(),
                    dynamic_cp={1: _Group(1), 2: _Group()},
                    dynamic_tp_cp={1: _Group(), 2: _Group(4)},
                ),
                id="dynamic_cp_tp2",
            ),
            pytest.param(
                lambda: dict(tp=_Group(), dynamic_cp={2: _Group()}), id="dynamic_cp_tp2_no_tp_map"
            ),
            pytest.param(lambda: dict(dynamic_tp_cp={2: _Group()}), id="tp_map_without_cp_map"),
            pytest.param(lambda: dict(dynamic_cp={}, dynamic_tp_cp={}), id="empty_dynamic_maps"),
        ],
    )
    def test_matches_the_shim(self, extra_groups):
        _register(**_data_parallel_groups(), **extra_groups())

        _assert_bootstrap_matches_the_shim()

    @pytest.mark.parametrize(
        "groups, missing",
        [
            pytest.param(
                lambda: {**_data_parallel_groups(), "dp_cp": None}, ["dp_cp"], id="no_dp_cp"
            ),
            pytest.param(lambda: dict(tp=_Group()), list(DATA_PARALLEL_FIELDS), id="no_dp"),
            pytest.param(
                lambda: {**_data_parallel_groups(), "dynamic_cp": {2: _Group()}},
                ["tp"],
                id="dynamic_cp_without_tp",
            ),
        ],
    )
    def test_refuses_what_the_shim_refused(self, groups, missing):
        """The shim asserted that these groups are set; bootstrap raises a ValueError."""
        _register(**groups())

        with pytest.raises(AssertionError):
            _read_with_the_shim()
        with pytest.raises(ValueError, match=re.escape(str(missing))):
            initialize._registered_pg_collection("owner")


@pytest.mark.usefixtures("fresh_fallback_record")
class TestUnregisteredGrid:
    """A grid set up without the registry is read from the global grid, with a warning."""

    def setup_method(self, method):
        _require_no_distributed()

    @pytest.fixture
    def unregistered_grid(self, monkeypatch):
        """Set the global groups that the shim reads, as a framework could, without registering."""
        groups = {}
        for name, field in (
            ("_DATA_PARALLEL_GROUP", "dp"),
            ("_DATA_PARALLEL_GROUP_WITH_GTP_REMAT", "dp_gtp_remat"),
            ("_DATA_PARALLEL_GROUP_WITH_CP", "dp_cp"),
            ("_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT", "dp_cp_gtp_remat"),
            ("_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP", "intra_dp_cp"),
            ("_TENSOR_MODEL_PARALLEL_GROUP", "tp"),
            ("_PIPELINE_MODEL_PARALLEL_GROUP", "pp"),
        ):
            groups[field] = _Group()
            monkeypatch.setattr(parallel_state, name, groups[field])
        assert parallel_state.get_registered_process_groups() is None
        return groups

    def test_falls_back_with_a_warning(self, unregistered_grid):
        expected = _read_with_the_shim()

        with _recorded_warnings() as record:
            built = initialize._registered_pg_collection("pretrain")

        _assert_same_collection(built, expected)
        assert built.dp is unregistered_grid["dp"]
        assert _fallbacks(record) == [("pretrain", "pg_collection")]
        assert _of_category(record, ParallelStateDeprecationWarning) == []

    def test_raises_under_strict_fallbacks(self, unregistered_grid):
        with warnings.catch_warnings():
            enable_strict_process_group_fallbacks()
            with pytest.raises(ProcessGroupFallbackWarning, match="pretrain was called without"):
                initialize._registered_pg_collection("pretrain")

    def test_an_uninitialized_grid_fails_after_the_warning_as_before(self):
        if parallel_state.is_initialized():
            pytest.skip("a grid is initialized")
        with pytest.raises(AssertionError, match="data parallel group"):
            _read_with_the_shim()

        with _recorded_warnings() as record, pytest.raises(AssertionError, match="data parallel"):
            initialize._registered_pg_collection("pretrain")
        assert _fallbacks(record) == [("pretrain", "pg_collection")]

    @pytest.mark.parametrize("seed_from", ["grid", "model", "nothing"])
    def test_initialization_reads_it_only_to_seed_from_it(
        self, monkeypatch, unregistered_grid, seed_from
    ):
        """initialize_megatron() reads the grid only to seed from it: a caller that passes
        seed_pg_collection, or skips seeding, gets no fallback warning."""
        parser = ArgumentParser()
        arguments.add_megatron_arguments(parser)
        args = parser.parse_args([])
        cfg = inference_cfg_container_from_args(args, build_model_config=False)
        monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
        monkeypatch.setattr(initialize, "get_args", lambda: args)
        for name in (
            "initialize_runtime_services",
            "setup_logging",
            "initialize_rerun_state_machine",
            "set_default_log_ranks",
            "print_rank_0",
        ):
            monkeypatch.setattr(initialize, name, Mock())
        seed = Mock()
        monkeypatch.setattr(initialize, "_set_random_seed", seed)
        # torch.distributed and the grid are already set up.
        monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(initialize.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(initialize.torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(initialize.torch.distributed, "get_world_size", lambda: 1)
        model = ProcessGroupCollection()

        with _recorded_warnings() as record:
            initialize.initialize_megatron(
                allow_no_cuda=True,
                skip_dependency_compilation=True,
                skip_random_seed=seed_from == "nothing",
                seed_pg_collection=model if seed_from == "model" else None,
            )

        if seed_from == "nothing":
            seed.assert_not_called()
        else:
            seeded_from = seed.call_args.kwargs["pg_collection"]
            assert (seeded_from is model) == (seed_from == "model")
        if seed_from == "grid":
            assert seeded_from.dp is unregistered_grid["dp"]
            assert _fallbacks(record) == [("initialize_megatron", "seed_pg_collection")]
        else:
            assert _fallbacks(record) == []


@pytest.mark.usefixtures("fresh_fallback_record")
class TestBootstrapReads:
    """pretrain() and initialize_megatron() read the registry: no shim, no warning."""

    def setup_method(self, method):
        _require_no_distributed()

    def teardown_method(self, method):
        parallel_state.destroy_model_parallel()

    @staticmethod
    def _forbid_the_shim(monkeypatch):
        def read_global_grid(*args, **kwargs):
            raise AssertionError("read the global grid with use_mpu_process_groups()")

        monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", read_global_grid)

    @pytest.mark.usefixtures("run_config")
    def test_pretrain_builds_its_collection_from_the_registry(self, monkeypatch):
        registered = _register(**_data_parallel_groups(), tp=_Group(), pp=_Group())
        self._forbid_the_shim(monkeypatch)
        args = SimpleNamespace(create_all_gather_group=False, expert_model_parallel_size=1)

        with _recorded_warnings() as record:
            enable_strict_process_group_fallbacks()
            pg_collection = training._build_pg_collection_from_parallel_state(args)

        assert pg_collection is not registered
        assert set(vars(pg_collection)) == FIELDS
        assert all(vars(pg_collection)[name] is group for name, group in vars(registered).items())
        assert record == []

    @pytest.mark.usefixtures("run_config")
    def test_initialization_seeds_from_the_registered_grid(self, monkeypatch):
        registered = _register(**_data_parallel_groups(), tp=_Group(), pp=_Group())
        self._forbid_the_shim(monkeypatch)
        monkeypatch.setattr(initialize, "get_args", lambda: SimpleNamespace())
        monkeypatch.setattr(initialize, "print_rank_0", Mock())
        monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(initialize.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(initialize.torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(initialize.torch.distributed, "get_world_size", lambda: 1)

        with _recorded_warnings() as record:
            enable_strict_process_group_fallbacks()
            pg_collection = initialize._initialize_distributed(None, None, None)

        assert set(vars(pg_collection)) == FIELDS
        assert pg_collection.pp is registered.pp and pg_collection.dp is registered.dp
        assert pg_collection.ep is None
        assert record == []


class TestLazyMpuInit:
    """--lazy-mpu-init sets the deprecated tensor-parallel overrides, with one warning."""

    @pytest.fixture
    def lazy_initialization(self, monkeypatch):
        parser = ArgumentParser()
        arguments.add_megatron_arguments(parser)
        args = parser.parse_args([])
        args.rank, args.world_size, args.tensor_model_parallel_size = 3, 4, 2
        cfg = inference_cfg_container_from_args(args, build_model_config=False)
        monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
        monkeypatch.setattr(initialize, "get_args", lambda: args)
        for name in (
            "initialize_runtime_services",
            "setup_logging",
            "initialize_rerun_state_machine",
            "_initialize_distributed",
            "_set_random_seed",
            "set_default_log_ranks",
            "print_rank_0",
        ):
            monkeypatch.setattr(initialize, name, Mock())
        # Restore the overrides, and let every call site of a deprecated function warn again.
        for name in ("_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", "_MPU_TENSOR_MODEL_PARALLEL_RANK"):
            monkeypatch.setattr(parallel_state, name, getattr(parallel_state, name))
        monkeypatch.setattr(parallel_state, "_warned_accessor_call_sites", set())
        return cfg

    @pytest.mark.parametrize("lazy", [False, True])
    def test_overrides_warn_once(self, lazy_initialization, lazy):
        lazy_initialization.dist.lazy_mpu_init = lazy

        with _recorded_warnings() as record:
            finish = initialize.initialize_megatron(
                allow_no_cuda=True, skip_dependency_compilation=True
            )

        deprecations = [
            warning for warning in record if warning.category is ParallelStateDeprecationWarning
        ]
        if not lazy:
            assert finish is None and deprecations == []
            return
        assert callable(finish)
        (deprecation,) = deprecations
        message = str(deprecation.message)
        assert message.startswith(
            "megatron.core.parallel_state.set_tensor_model_parallel_world_size() and "
            "set_tensor_model_parallel_rank() are deprecated since Megatron Core 0.21 and will be "
            "removed in 0.23. --lazy-mpu-init calls them"
        )
        assert "get_registered_process_groups()" in message
        # Attributed to the caller of initialize_megatron(), which chose --lazy-mpu-init.
        assert deprecation.filename == __file__
        assert parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE == 2
        assert parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK == 3


@pytest.mark.skipif(Utils.world_size > 1, reason="A distributed run initializes the real world")
class TestFakeBackend:
    """With --fake-process-group, each process stands for one rank and communicates nothing."""

    def test_all_gather_groups_mirror_the_collection(self, monkeypatch):
        """The fake all-gather returns this rank's own entry from every rank."""
        _require_no_distributed()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        monkeypatch.setattr(parallel_state, "_global_process_group_list", None)
        torch.distributed.init_process_group("fake", store=FakeStore(), rank=5, world_size=8)
        try:
            grid = HyperCommGrid([2, 2, 2], ["tp", "ep", "dp"])
            pg_collection = ProcessGroupCollection(
                dp_cp=grid.create_pg(["ep", "dp"]),
                ep=grid.create_pg("ep"),
                expt_dp=grid.create_pg("dp"),
            )

            all_gather_groups = parallel_state.create_all_gather_groups(
                for_expert_parallelism=True, pg_collection=pg_collection
            )

            for all_gather_group, group in zip(
                all_gather_groups, (pg_collection.dp_cp, pg_collection.expt_dp)
            ):
                assert all_gather_group is not group
                assert torch.distributed.get_process_group_ranks(
                    all_gather_group
                ) == torch.distributed.get_process_group_ranks(group)
        finally:
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(Utils.world_size < 2, reason="Requires a distributed run")
class TestStandardGrid:
    """On the grid that initialize_model_parallel() builds and registers."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("grid", [grid[1:] for grid in GRIDS], ids=[grid[0] for grid in GRIDS])
    def test_bootstrap_matches_the_shim(self, grid):
        kwargs, num_ranks = grid
        _require_divisible(num_ranks)
        Utils.initialize_model_parallel(**kwargs)

        _assert_bootstrap_matches_the_shim()

    @pytest.mark.parametrize(
        "grid", [grid[1:] for grid in EXPERT_GRIDS], ids=[grid[0] for grid in EXPERT_GRIDS]
    )
    def test_all_gather_groups_mirror_the_collection(self, grid):
        kwargs, num_ranks = grid
        _require_divisible(num_ranks)
        Utils.initialize_model_parallel(**kwargs)
        pg_collection = parallel_state.get_registered_process_groups()

        with _recorded_warnings() as record:
            dp_cp_ag, expt_dp_ag = parallel_state.create_all_gather_groups(
                for_expert_parallelism=True,
                timeout=timedelta(minutes=3),
                pg_collection=pg_collection,
            )

        assert _fallbacks(record) == []
        for all_gather_group, group in (
            (dp_cp_ag, pg_collection.dp_cp),
            (expt_dp_ag, pg_collection.expt_dp),
        ):
            # A separate communicator over the ranks of the group it gathers for.
            assert all_gather_group is not group
            assert torch.distributed.get_process_group_ranks(
                all_gather_group
            ) == torch.distributed.get_process_group_ranks(group)

    @pytest.mark.parametrize(
        "grid", [grid[1:] for grid in EXPERT_GRIDS], ids=[grid[0] for grid in EXPERT_GRIDS]
    )
    def test_all_gather_groups_exchange_one_rank_per_group(self, monkeypatch, grid):
        """Each rank sends the lowest rank of each of its groups, not all their ranks, which
        every rank would receive from every rank: with groups that span the world, that is
        the square of the world size."""
        kwargs, num_ranks = grid
        _require_divisible(num_ranks)
        Utils.initialize_model_parallel(**kwargs)
        pg_collection = parallel_state.get_registered_process_groups()
        sent = []
        all_gather_object = torch.distributed.all_gather_object

        def record_all_gather_object(object_list, obj, *args, **kwargs):
            sent.append(obj)
            return all_gather_object(object_list, obj, *args, **kwargs)

        monkeypatch.setattr(torch.distributed, "all_gather_object", record_all_gather_object)

        parallel_state.create_all_gather_groups(
            for_expert_parallelism=True, pg_collection=pg_collection
        )

        assert sent == [
            tuple(
                min(torch.distributed.get_process_group_ranks(group))
                for group in (pg_collection.dp_cp, pg_collection.expt_dp)
            )
        ]

    @pytest.mark.usefixtures("run_config", "fresh_fallback_record")
    @pytest.mark.parametrize(
        "grid", [grid[1:] for grid in EXPERT_GRIDS], ids=[grid[0] for grid in EXPERT_GRIDS]
    )
    def test_training_builds_all_gather_groups_in_any_rank_order(self, grid):
        """--create-all-gather-group supported experts only in the default order without EGTP."""
        kwargs, num_ranks = grid
        _require_divisible(num_ranks)
        Utils.initialize_model_parallel(**kwargs)
        args = SimpleNamespace(create_all_gather_group=True, expert_model_parallel_size=2)

        with _recorded_warnings() as record:
            pg_collection = training._build_pg_collection_from_parallel_state(args)

        assert _fallbacks(record) == []
        for name in ("dp_cp", "expt_dp"):
            all_gather_group = getattr(pg_collection, f"{name}_ag")
            assert all_gather_group is not getattr(pg_collection, name)
            assert torch.distributed.get_process_group_ranks(
                all_gather_group
            ) == torch.distributed.get_process_group_ranks(getattr(pg_collection, name))

    @pytest.mark.usefixtures("fresh_fallback_record")
    def test_all_gather_groups_without_a_collection_warn(self):
        """The deprecated form keeps its groups: the default rank order on the global grid."""
        _require_divisible(2)
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        pg_collection = parallel_state.get_registered_process_groups()

        with _recorded_warnings() as record:
            dp_cp_ag, expt_dp_ag = parallel_state.create_all_gather_groups(
                for_expert_parallelism=True
            )

        assert _fallbacks(record) == [("create_all_gather_groups", "pg_collection")]
        assert _of_category(record, ParallelStateDeprecationWarning) == []
        for all_gather_group, group in (
            (dp_cp_ag, pg_collection.dp_cp),
            (expt_dp_ag, pg_collection.expt_dp),
        ):
            assert torch.distributed.get_process_group_ranks(
                all_gather_group
            ) == torch.distributed.get_process_group_ranks(group)

    @pytest.mark.parametrize("multi_module", [False, True])
    def test_timeout_update_reaches_the_groups_of_the_collection(self, monkeypatch, multi_module):
        """Groups that parallel_state did not create are updated through the run's collection."""
        Utils.initialize_model_parallel()
        registered = parallel_state.get_registered_process_groups()
        # A framework's group, created without parallel_state.create_group().
        world = list(range(torch.distributed.get_world_size()))
        framework_group = torch.distributed.new_group(ranks=world)
        pg_collection = ProcessGroupCollection(
            tp=registered.tp, dp=framework_group, hcp=[framework_group, registered.cp]
        )
        if multi_module:
            pg_collection = MultiModuleProcessGroupCollection(
                module_pgs={
                    "encoder": ProcessGroupCollection(cp=registered.cp),
                    "llm": pg_collection,
                },
                language_model_module_name="llm",
            )
        updated = []
        monkeypatch.setattr(
            torch.distributed.distributed_c10d,
            "_set_pg_timeout",
            lambda timeout, group: updated.append(group),
        )
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        created = list(parallel_state._global_process_group_list)

        try:
            parallel_state.update_pg_timeout(timedelta(minutes=7))
            assert updated == created

            updated.clear()
            parallel_state.update_pg_timeout(timedelta(minutes=7), pg_collection=pg_collection)
            # Each group once: the created ones, then the framework's.
            assert updated == created + [framework_group]
        finally:
            torch.distributed.destroy_process_group(framework_group)

    @pytest.mark.parametrize("multi_module", [False, True])
    def test_timeout_update_reaches_a_grid_that_a_framework_built(self, monkeypatch, multi_module):
        """A run whose groups a framework built, as a multi-module run's HyperCommGrid does,
        created no group with parallel_state.create_group()."""
        _require_divisible(2)
        Utils.initialize_distributed()
        monkeypatch.setattr(parallel_state, "_global_process_group_list", None)
        grid = HyperCommGrid([2, torch.distributed.get_world_size() // 2], ["tp", "dp"])
        tp, dp = grid.create_pg("tp"), grid.create_pg("dp")
        pg_collection = ProcessGroupCollection(tp=tp, dp=dp)
        if multi_module:
            pg_collection = MultiModuleProcessGroupCollection(
                module_pgs={"llm": pg_collection}, language_model_module_name="llm"
            )
        updated = []
        monkeypatch.setattr(
            torch.distributed.distributed_c10d,
            "_set_pg_timeout",
            lambda timeout, group: updated.append(group),
        )
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

        try:
            parallel_state.update_pg_timeout(timedelta(minutes=7))
            # None is the default process group.
            assert updated == [None]

            updated.clear()
            parallel_state.update_pg_timeout(timedelta(minutes=7), pg_collection=pg_collection)
            assert updated == [None, tp, dp]
        finally:
            grid.destroy()
