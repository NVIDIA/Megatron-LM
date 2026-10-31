# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch.distributed as dist
from torch.distributed import DeviceMesh

from megatron.core import parallel_state
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _get_dp_tp_mesh, _get_hsdp_tp_mesh
from tests.unit_tests.test_utilities import Utils


def _check_mesh(mesh, groups):
    """Check every axis against real process groups, including rank ordering."""
    assert tuple(mesh.shape) == tuple(group.size() for group in groups)
    assert mesh.unique().numel() == mesh.numel()
    coordinate = tuple(group.rank() for group in groups)
    assert mesh[coordinate].item() == dist.get_rank()
    for axis, group in enumerate(groups):
        index = list(coordinate)
        index[axis] = slice(None)
        assert mesh[tuple(index)].tolist() == dist.get_process_group_ranks(group)
    device_mesh = DeviceMesh.from_group(
        groups,
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=tuple(f"axis{i}" for i in range(len(groups))),
    )
    assert tuple(device_mesh.get_coordinate()) == coordinate
    for axis, group in enumerate(groups):
        assert dist.get_process_group_ranks(
            device_mesh.get_group(axis)
        ) == dist.get_process_group_ranks(group)


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize(
    "tp,cp,ep,etp",
    [
        (1, 1, 1, 1),
        (2, 1, 1, 2),
        (1, 2, 1, 1),
        (1, 1, 2, 1),
        (2, 1, 2, 1),
        (1, 2, 2, 1),
        (2, 2, 2, 2),
    ],
)
def test_dp_tp_mesh(pp, tp, cp, ep, etp):
    """Dense and expert meshes must exclude PP, including nonzero stages."""
    if Utils.world_size % (pp * tp * cp) or Utils.world_size % (pp * ep * etp):
        pytest.skip("Topology does not divide the available world size")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    try:
        for dp_group, tp_group, ep_size in (
            (
                parallel_state.get_data_parallel_group(with_context_parallel=True),
                parallel_state.get_tensor_model_parallel_group(),
                1,
            ),
            (
                parallel_state.get_expert_data_parallel_group(),
                parallel_state.get_expert_tensor_parallel_group(),
                ep,
            ),
        ):
            mesh = _get_dp_tp_mesh(dp_group, tp_group, ep_size)
            _check_mesh(mesh, [dp_group, tp_group])
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize(
    "tp,cp,ep,etp",
    [(1, 1, 1, 1), (2, 1, 1, 2), (1, 2, 1, 1), (1, 1, 2, 1), (2, 1, 2, 1), (1, 2, 2, 1)],
)
def test_hsdp_tp_mesh(pp, tp, cp, ep, etp):
    """Hybrid sharding must also keep pipeline stages out of DP meshes."""
    outer_size = 2
    if Utils.world_size % (pp * tp * cp * outer_size) or Utils.world_size % (
        pp * ep * etp * outer_size
    ):
        pytest.skip("Topology requires more data-parallel ranks")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        num_distributed_optimizer_instances=outer_size,
    )
    try:
        outer_group = parallel_state.get_inter_distributed_optimizer_instance_group()
        for dp_group, tp_group, ep_size in (
            (
                parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                ),
                parallel_state.get_tensor_model_parallel_group(),
                1,
            ),
            (
                parallel_state.get_expert_data_parallel_group(partial_expert_data_parallel=True),
                parallel_state.get_expert_tensor_parallel_group(),
                ep,
            ),
        ):
            mesh = _get_hsdp_tp_mesh(outer_group, dp_group, tp_group, ep_size)
            _check_mesh(mesh, [outer_group, dp_group, tp_group])
    finally:
        Utils.destroy_model_parallel()
