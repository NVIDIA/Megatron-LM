# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent AG communicators must preserve FSDP buffer shard coordinates."""

from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_model
from tests.unit_tests.distributed.mfsdp_v1.test_mfsdp_nonuniform_sharding import destroy_device_mesh
from tests.unit_tests.test_utilities import Utils


@contextmanager
def buffer_mesh(ag_layout, hybrid=True):
    """Build real shard/AG groups; every rank creates them in the same order."""
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2:
        pytest.skip("Requires an even world size of at least four ranks.")
    shape = (2, world_size // 2, 1) if hybrid else (world_size, 1)
    names = ("outer", "shard", "tp") if hybrid else ("shard", "tp")
    mesh = init_device_mesh("cuda", mesh_shape=shape, mesh_dim_names=names)
    if hybrid:
        mesh[("outer", "shard")]._flatten("replica")
    inner_size = shape[-2]
    if ag_layout == "replica":
        rank_sets = [list(range(world_size))]
    elif ag_layout == "inner":
        rank_sets = [list(range(i, i + inner_size)) for i in range(0, world_size, inner_size)]
    elif ag_layout == "different":
        rank_sets = [list(range(i, world_size, 2)) for i in range(2)]
    else:
        rank_sets = []
    ag_group = None
    for ranks in rank_sets:
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            ag_group = group
    try:
        yield mesh, ag_group
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        if ag_group is not None:
            dist.destroy_process_group(ag_group)
        destroy_device_mesh(mesh)


def build_buffer_model(
    mesh, ag_group, expert=False, outer_strategy="no_shard", strategy="optim_grads_params"
):
    """Initialize real MFSDP buffers for a small dense or expert parameter set."""
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    if expert:
        # MFSDP classifies parameters by the '.experts.' infix.
        model.block.experts = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False)])
    else:
        model.block.linear = torch.nn.Linear(8, 8, bias=False)
    hybrid = "outer" in mesh.mesh_dim_names
    replica_group = mesh["replica"].get_group() if hybrid else None
    return fully_shard_model(
        model.cuda(),
        fsdp_unit_modules=[torch.nn.Linear],
        device_mesh=mesh,
        dp_shard_dim="shard",
        tp_dim="tp",
        dp_outer_dim="outer" if hybrid else None,
        hybrid_fsdp_group=replica_group,
        hybrid_fsdp_expt_group=replica_group if expert else None,
        expt_device_mesh=mesh if expert else None,
        fsdp_group_ag=None if expert else ag_group,
        expt_fsdp_group_ag=ag_group if expert else None,
        zero_dp_strategy=strategy,
        outer_dp_sharding_strategy=outer_strategy,
    )


class TestSeparateAllGatherGroups:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("expert", [False, True])
    @pytest.mark.parametrize("ag_layout", ["replica", "inner", "off"])
    def test_basic_hsdp_requires_matching_ag_shards(self, expert, ag_layout):
        with buffer_mesh(ag_layout) as (mesh, ag_group):
            if ag_layout == "replica":
                domain = "expert" if expert else "dense"
                with pytest.raises(
                    ValueError, match=f"all-gather group for {domain}.*same ordered ranks"
                ):
                    build_buffer_model(mesh, ag_group, expert)
            else:
                model = build_buffer_model(mesh, ag_group, expert)
                for group in model.param_and_grad_buffer.parameter_groups:
                    assert group.is_expert_param == expert
                    assert group.model_weight_buffer.data_size == group.main_weight_buffer.data_size

    def test_same_size_different_ag_membership_is_rejected(self):
        with buffer_mesh("different") as (mesh, ag_group):
            assert ag_group.size() == mesh["shard"].get_group().size()
            with pytest.raises(ValueError, match="same ordered ranks"):
                build_buffer_model(mesh, ag_group)

    def test_outer_sharding_does_not_consume_independent_ag(self):
        with buffer_mesh("replica") as (mesh, ag_group):
            model = build_buffer_model(mesh, ag_group, outer_strategy="optim")
            for group in model.param_and_grad_buffer.parameter_groups:
                assert group.model_weight_buffer.data_parallel_group is not ag_group

    def test_ordinary_fsdp_accepts_matching_separate_ag(self):
        with buffer_mesh("replica", hybrid=False) as (mesh, ag_group):
            model = build_buffer_model(mesh, ag_group)
            for group in model.param_and_grad_buffer.parameter_groups:
                assert group.model_weight_buffer.data_parallel_group is ag_group

    def test_unsharded_parameters_do_not_consume_independent_ag(self):
        with buffer_mesh("replica") as (mesh, ag_group):
            model = build_buffer_model(mesh, ag_group, strategy="no_shard")
            for group in model.param_and_grad_buffer.parameter_groups:
                assert group.model_weight_buffer is None
