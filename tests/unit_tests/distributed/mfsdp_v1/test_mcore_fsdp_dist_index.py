# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process groups behind the Megatron-FSDP v1 adapter's distributed index."""

import contextlib
import re
import sys
import types
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV1
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils
from tools import check_process_group_usage

pytestmark = pytest.mark.skipif(
    not is_torch_min_version("2.4.0"), reason="Megatron-FSDP requires torch >= 2.4.0"
)


def _skip_unless_world_size_divisible_by(size):
    if Utils.world_size % size != 0:
        pytest.skip(f"needs a world size divisible by {size}, got {Utils.world_size}")


@contextlib.contextmanager
def _forbid_global_process_groups():
    """Make every parallel_state accessor that the process-group checker counts raise.

    The accessors are patched on parallel_state and wherever an imported megatron module bound
    them by name, and ProcessGroupCollection.use_mpu_process_groups() raises too.
    """
    accessors = {
        name: value
        for name, value in vars(parallel_state).items()
        if callable(value) and check_process_group_usage._is_deprecated_accessor(name)
    }

    def forbidden(name):
        def read_global_grid(*args, **kwargs):
            raise AssertionError(f"parallel_state.{name}() reads the global parallel grid")

        return read_global_grid

    stubs = {id(value): forbidden(name) for name, value in accessors.items()}
    with contextlib.ExitStack() as patches:
        for module_name, module in list(sys.modules.items()):
            if module_name.split(".")[0] != "megatron" or not isinstance(module, types.ModuleType):
                continue
            for attr, value in list(vars(module).items()):
                if id(value) in stubs:
                    patches.enter_context(mock.patch.object(module, attr, stubs[id(value)]))
        patches.enter_context(
            mock.patch.object(
                ProcessGroupCollection,
                "use_mpu_process_groups",
                side_effect=AssertionError("use_mpu_process_groups() reads the global grid"),
            )
        )
        yield


@contextlib.contextmanager
def _record_group_creation():
    """Record the ranks of every process group created in the block, in creation order."""
    created = []
    new_group = dist.distributed_c10d.new_group

    def recording_new_group(*args, **kwargs):
        ranks = kwargs["ranks"] if "ranks" in kwargs else (args[0] if args else None)
        created.append(None if ranks is None else sorted(ranks))
        return new_group(*args, **kwargs)

    with (
        mock.patch.object(dist, "new_group", recording_new_group),
        mock.patch.object(dist.distributed_c10d, "new_group", recording_new_group),
    ):
        yield created


def _fsdp_inputs(*, moe, hsdp):
    """Constructor arguments of a small Megatron-FSDP v1 model, without its process groups."""
    return dict(
        config=TransformerConfig(
            num_attention_heads=1, num_layers=1, num_moe_experts=4 if moe else None
        ),
        ddp_config=DistributedDataParallelConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=10000,
            use_megatron_fsdp=True,
            num_distributed_optimizer_instances=2 if hsdp else 1,
        ),
        module=torch.nn.Sequential(
            torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)
        ).cuda(),
        fsdp_unit_modules=[torch.nn.Linear],
    )


def _grid_collection(*, moe, hsdp, with_tp=True):
    """Collection of a TP=1 grid; expert layers run EP=2; HSDP uses two optimizer instances.

    The global grid in these tests runs TP=2, EP=1 and one optimizer instance, so each group here
    differs from its global counterpart. Creating groups is collective: call on every rank.
    """
    world_size = dist.get_world_size()
    if hsdp:
        grid = HyperCommGrid([1, world_size // 2, 2], ["tp", "intra_dp", "inter_dp"])
        pg_collection = ProcessGroupCollection(
            intra_dp_cp=grid.create_pg("intra_dp"),
            inter_dist_opt=grid.create_pg("inter_dp"),
            dp_cp=grid.create_pg(["intra_dp", "inter_dp"]),
        )
    else:
        grid = HyperCommGrid([1, world_size], ["tp", "dp"])
        pg_collection = ProcessGroupCollection(dp_cp=grid.create_pg("dp"))
    if with_tp:
        pg_collection.tp = grid.create_pg("tp")
    if moe:
        if hsdp:
            expert_grid = HyperCommGrid(
                [1, 2, world_size // 4, 2], ["tp", "ep", "intra_dp", "inter_dp"]
            )
            pg_collection.intra_expt_dp = expert_grid.create_pg("intra_dp")
            pg_collection.expt_dp = expert_grid.create_pg(["intra_dp", "inter_dp"])
        else:
            expert_grid = HyperCommGrid([1, 2, world_size // 2], ["tp", "ep", "dp"])
            pg_collection.expt_dp = expert_grid.create_pg("dp")
        pg_collection.ep = expert_grid.create_pg("ep")
        if with_tp:
            pg_collection.expt_tp = expert_grid.create_pg("tp")
    return pg_collection


def _groups_in_index(fsdp):
    """Every process group the adapter put into its distributed index, by role."""
    index = fsdp.megatron_fsdp_dist_index
    groups = {
        "tp": fsdp.tp_group,
        "mesh tp": index.device_mesh["tp"].get_group(),
        "fsdp": index.get_fsdp_group(),
        "fsdp all-gather": index.get_fsdp_group(independent_all_gather=True),
        "outer fsdp": index.get_outer_fsdp_group(),
        "dp": index.get_dp_group(),
        "hybrid expert dp": index.hybrid_fsdp_expt_group,
    }
    if index.expt_device_mesh is not None:
        groups.update(
            {
                "expert mesh tp": index.expt_device_mesh["tp"].get_group(),
                "expert fsdp": index.get_fsdp_group(is_expert_parallel=True),
                "expert fsdp all-gather": index.get_fsdp_group(
                    is_expert_parallel=True, independent_all_gather=True
                ),
                "expert outer fsdp": index.get_outer_fsdp_group(is_expert_parallel=True),
                "expert dp": index.get_dp_group(is_expert_parallel=True),
            }
        )
    return groups


def _group_names(groups):
    return {role: None if group is None else group.group_name for role, group in groups.items()}


def _ranks_in_index(fsdp):
    """Ranks of every process group in the adapter's distributed index, and its meshes."""
    ranks = {
        role: None if group is None else dist.get_process_group_ranks(group)
        for role, group in _groups_in_index(fsdp).items()
    }
    index = fsdp.megatron_fsdp_dist_index
    ranks["mesh"] = index.device_mesh.mesh.tolist()
    if index.expt_device_mesh is not None:
        ranks["expert mesh"] = index.expt_device_mesh.mesh.tolist()
    return ranks


@pytest.fixture
def global_grid_tp2():
    """A global grid whose layout differs from the collections that the tests pass."""
    _skip_unless_world_size_divisible_by(4)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("hsdp", [False, True], ids=["fsdp", "hsdp"])
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_explicit_collection_uses_only_its_groups(global_grid_tp2, moe, hsdp):
    """No global read and no new group: every group in the index comes from the collection."""
    pg_collection = _grid_collection(moe=moe, hsdp=hsdp)
    inputs = _fsdp_inputs(moe=moe, hsdp=hsdp)

    with _forbid_global_process_groups(), _record_group_creation() as created:
        fsdp = FullyShardedDataParallelV1(**inputs, pg_collection=pg_collection)
    fsdp.stop_communication()

    assert created == []
    expected = {
        "tp": pg_collection.tp,
        "mesh tp": pg_collection.tp,
        "fsdp": pg_collection.intra_dp_cp if hsdp else pg_collection.dp_cp,
        "fsdp all-gather": None,
        "outer fsdp": pg_collection.inter_dist_opt if hsdp else None,
        "dp": pg_collection.dp_cp,
        "hybrid expert dp": pg_collection.expt_dp if hsdp else None,
    }
    if moe:
        expected.update(
            {
                "expert mesh tp": pg_collection.expt_tp,
                "expert fsdp": pg_collection.intra_expt_dp if hsdp else pg_collection.expt_dp,
                "expert fsdp all-gather": None,
                "expert outer fsdp": pg_collection.inter_dist_opt if hsdp else None,
                "expert dp": pg_collection.expt_dp,
            }
        )
    assert _group_names(_groups_in_index(fsdp)) == _group_names(expected)


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_collection_without_tp_groups_creates_them_on_every_rank(global_grid_tp2, moe):
    """Without tp/expt_tp, every rank creates the same single-rank groups in the same order."""
    pg_collection = _grid_collection(moe=moe, hsdp=False, with_tp=False)
    inputs = _fsdp_inputs(moe=moe, hsdp=False)

    with _forbid_global_process_groups(), _record_group_creation() as created:
        fsdp = FullyShardedDataParallelV1(**inputs, pg_collection=pg_collection)
    fsdp.stop_communication()

    created_on_every_rank = [None] * dist.get_world_size()
    dist.all_gather_object(created_on_every_rank, created)
    assert all(
        created_on_rank == created_on_every_rank[0] for created_on_rank in created_on_every_rank
    ), f"ranks created different process groups: {created_on_every_rank}"
    single_rank = [dist.get_rank()]
    ranks = _ranks_in_index(fsdp)
    assert ranks["tp"] == ranks["mesh tp"] == single_rank
    if moe:
        assert ranks["expert mesh tp"] == single_rank


@pytest.mark.parametrize(
    "moe, hsdp, field",
    [
        (False, False, "dp_cp"),
        (True, False, "ep"),
        (True, False, "expt_dp"),
        (False, True, "intra_dp_cp"),
        (False, True, "inter_dist_opt"),
        (True, True, "intra_expt_dp"),
    ],
)
def test_collection_missing_a_needed_group_raises(global_grid_tp2, moe, hsdp, field):
    pg_collection = _grid_collection(moe=moe, hsdp=hsdp)
    delattr(pg_collection, field)
    inputs = _fsdp_inputs(moe=moe, hsdp=hsdp)

    with (
        _forbid_global_process_groups(),
        pytest.raises(ValueError, match=re.escape(f"['{field}']")),
    ):
        FullyShardedDataParallelV1(**inputs, pg_collection=pg_collection)


@pytest.mark.parametrize(
    "grid, needs",
    [
        pytest.param({}, 1, id="dp"),
        pytest.param({"tensor_model_parallel_size": 2}, 2, id="tp2"),
        pytest.param({"context_parallel_size": 2}, 2, id="cp2"),
        pytest.param({"expert_model_parallel_size": 2}, 2, id="ep2"),
        pytest.param(
            {
                "tensor_model_parallel_size": 2,
                "expert_model_parallel_size": 2,
                "expert_tensor_parallel_size": 1,
            },
            2,
            id="tp2-ep2-etp1",
        ),
        pytest.param({"num_distributed_optimizer_instances": 2}, 2, id="hsdp2"),
        pytest.param(
            {"tensor_model_parallel_size": 2, "num_distributed_optimizer_instances": 2},
            4,
            id="tp2-hsdp2",
        ),
        pytest.param(
            {"expert_model_parallel_size": 2, "num_distributed_optimizer_instances": 2},
            4,
            id="ep2-hsdp2",
        ),
    ],
)
def test_standard_grid_collection_selects_the_fallback_ranks(grid, needs):
    """On the standard grid, the collection path picks the ranks the global fallback picks."""
    _skip_unless_world_size_divisible_by(needs)
    moe = grid.get("expert_model_parallel_size", 1) > 1
    hsdp = grid.get("num_distributed_optimizer_instances", 1) > 1
    Utils.initialize_model_parallel(**grid)
    try:
        fallback = FullyShardedDataParallelV1(**_fsdp_inputs(moe=moe, hsdp=hsdp))
        fallback.stop_communication()
        explicit = FullyShardedDataParallelV1(
            **_fsdp_inputs(moe=moe, hsdp=hsdp),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        explicit.stop_communication()

        assert _ranks_in_index(explicit) == _ranks_in_index(fallback)
    finally:
        Utils.destroy_model_parallel()
