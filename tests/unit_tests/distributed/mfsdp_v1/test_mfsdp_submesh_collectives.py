# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""
Megatron-FSDP collectives stay on the groups of the model that issues them.

Each test splits the world into two models on disjoint halves, as an actor and a
reference model would share one job. The second model issues one more call than
the first, so any collective on the default (WORLD) group would pair ranks of
different models. The tests make such a collective fail at once instead of hanging.

Run with torchrun (CPU ranks use gloo):
    torchrun --nproc_per_node=4 -m pytest test_mfsdp_submesh_collectives.py -v
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp import (
    param_and_grad_buffer as param_and_grad_buffer_module,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_optimizer
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    ParamAndGradBuffer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import FSDPDistributedIndex

# The megatron_fsdp package exports a fully_shard function that shadows this module.
fully_shard_module = importlib.import_module(
    "megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard"
)

MODEL = "model"
EP = "ep"
DP_SHARD = "dp_shard"


def _group_ranks(group):
    return dist.get_process_group_ranks(group)


def _build_model_dist_index(distributed_setup, use_expert_mesh):
    """Place this rank in one of two models, each data-parallel over half the world.

    Returns this rank's model index, its FSDPDistributedIndex, and the rank lists of
    the groups Megatron-FSDP registers user buffers on: the FSDP group, then the
    expert FSDP group when an expert mesh is given.
    """
    world_size = distributed_setup.world_size
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("Requires an even world size of at least 4.")
    if use_expert_mesh and world_size % 4 != 0:
        pytest.skip("The expert mesh requires a world size divisible by 4.")

    if not dist.is_initialized():
        # Name the backend so that DeviceMesh does not create the default group for
        # every device type it finds.
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device_type = distributed_setup.device.type
    mesh = init_device_mesh(device_type, (2, world_size // 2), mesh_dim_names=(MODEL, DP_SHARD))
    expert_mesh = None
    if use_expert_mesh:
        # EP=2 inside each model, so the expert FSDP group is half the model's FSDP group.
        expert_mesh = init_device_mesh(
            device_type, (2, 2, world_size // 4), mesh_dim_names=(MODEL, EP, DP_SHARD)
        )

    dist_index = FSDPDistributedIndex(
        device_mesh=mesh, dp_shard_dim=DP_SHARD, expt_device_mesh=expert_mesh
    )
    groups = [_group_ranks(mesh.get_group(DP_SHARD))]
    if use_expert_mesh:
        groups.append(_group_ranks(expert_mesh.get_group(DP_SHARD)))
    return mesh.get_local_rank(MODEL), dist_index, groups


def _forbid_world_collective(monkeypatch, name):
    """Make torch.distributed.<name> fail on the default group; record the other groups."""
    real_collective = getattr(dist, name)
    calls = []

    def collective(*args, group=None, **kwargs):
        if group is None or group is dist.group.WORLD:
            raise AssertionError(f"torch.distributed.{name} was issued on the default group.")
        calls.append(_group_ranks(group))
        return real_collective(*args, group=group, **kwargs)

    monkeypatch.setattr(dist, name, collective)
    return calls


@pytest.fixture
def stub_nccl_allocator(monkeypatch):
    """Run user-buffer setup without NCCL memory pools. Returns the registered groups."""
    registered_groups = []
    stub = SimpleNamespace(
        init=lambda: None,
        create_nccl_mem_pool=lambda symmetric=True: object(),
        register_mem_pool=lambda pool, group, symmetric=True: registered_groups.append(
            _group_ranks(group)
        ),
    )
    monkeypatch.setattr(param_and_grad_buffer_module, "nccl_allocator", stub)
    monkeypatch.setattr(param_and_grad_buffer_module, "NCCL_MEMORY_POOL", None)
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return registered_groups


class _StopAfterUserBufferSetup(Exception):
    """Ends ParamAndGradBuffer construction once its user-buffer groups are set up."""


@pytest.mark.parametrize("use_expert_mesh", [False, True])
def test_buffer_construction_barriers_only_on_registered_groups(
    distributed_setup, monkeypatch, stub_nccl_allocator, use_expert_mesh
):
    """Building a buffer with nccl_ub barriers on its own groups only."""
    model_id, dist_index, ubr_groups = _build_model_dist_index(distributed_setup, use_expert_mesh)

    def stop(*args, **kwargs):
        raise _StopAfterUserBufferSetup()

    monkeypatch.setattr(ParamAndGradBuffer, "get_mem_alloc_context", stop)
    barrier_groups = _forbid_world_collective(monkeypatch, "barrier")

    # The second model builds one more buffer, e.g. for a second virtual pipeline chunk.
    num_buffers = model_id + 1
    for _ in range(num_buffers):
        with pytest.raises(_StopAfterUserBufferSetup):
            ParamAndGradBuffer(
                ddp_config=DistributedDataParallelConfig(nccl_ub=True),
                module=torch.nn.Linear(2, 2),
                bucketing_policy=BucketingPolicy(),
                dist_index=dist_index,
                device=distributed_setup.device,
            )

    assert barrier_groups == ubr_groups * num_buffers


@pytest.mark.parametrize("use_expert_mesh", [False, True])
def test_manual_buffer_registration_barriers_only_on_registered_groups(
    distributed_setup, monkeypatch, stub_nccl_allocator, use_expert_mesh
):
    """Manual registration synchronizes and registers on the buffer's own groups only."""
    model_id, dist_index, ubr_groups = _build_model_dist_index(distributed_setup, use_expert_mesh)
    groups = [dist_index.get_fsdp_group()]
    if use_expert_mesh:
        groups.append(dist_index.get_fsdp_group(is_expert_parallel=True))

    num_buffers = model_id + 1
    buffers = []
    for _ in range(num_buffers):
        buffer = ParamAndGradBuffer.__new__(ParamAndGradBuffer)
        buffer.ddp_config = SimpleNamespace(
            nccl_ub=True,
            fsdp_double_buffer=True,
            fsdp_manual_registration=True,
            disable_symmetric_registration=False,
        )
        buffer.already_registered = False
        buffer.ubr_groups = groups
        buffers.append(buffer)
    barrier_groups = _forbid_world_collective(monkeypatch, "barrier")

    for buffer in buffers:
        buffer.manual_buffer_registration()

    assert barrier_groups == ubr_groups * num_buffers
    assert stub_nccl_allocator == ubr_groups * num_buffers


@pytest.mark.parametrize("use_expert_mesh", [False, True])
def test_optimizer_state_dict_gathers_keys_within_the_model(
    distributed_setup, monkeypatch, use_expert_mesh
):
    """The DCP pre-processing hook gathers optimizer state keys from its model's ranks only.

    The two models use different optimizers. The last rank of each model holds an empty
    shard and has no optimizer state, so it must take its keys from its own model.
    """
    model_id, dist_index, key_groups = _build_model_dist_index(distributed_setup, use_expert_mesh)
    model_ranks = set(key_groups[0])
    dp_mesh = dist_index.get_submesh(DP_SHARD)
    dp_size = dp_mesh.size()
    dp_rank = dp_mesh.get_local_rank()

    # Every rank but the last holds one row of the parameter.
    rows = 1 if dp_rank < dp_size - 1 else 0
    param = torch.nn.Parameter(
        DTensor.from_local(
            torch.ones(rows, 2, device=distributed_setup.device),
            dp_mesh,
            [Shard(0)],
            shape=(dp_size - 1, 2),
            stride=(2, 1),
        )
    )
    param._megatron_fsdp_model = SimpleNamespace(dist_index=dist_index)
    if model_id == 0:
        optimizer = torch.optim.Adam([param], lr=0.1)
        expected_keys = {"exp_avg", "exp_avg_sq"}
    else:
        optimizer = torch.optim.SGD([param], lr=0.1, momentum=0.9)
        expected_keys = {"momentum_buffer"}
    optimizer = fully_shard_optimizer(optimizer, preproc_state_dict_for_dcp_ckpt=True)

    gather_calls = _forbid_world_collective(monkeypatch, "all_gather_object")
    key_gathers = []
    extended_keys = []
    real_preprocess = fully_shard_module.preprocess_state_dict_for_uneven_dtensor
    num_seen = [0]

    def preprocess(state_dict):
        # Gathers issued after the previous chunk-metadata sync are this call's key gather.
        key_gathers.append(gather_calls[num_seen[0] :])
        extended_keys.append(
            {
                idx: {key for key, val in state.items() if isinstance(val, DTensor)}
                for idx, state in state_dict["state"].items()
            }
        )
        result = real_preprocess(state_dict)
        num_seen[0] = len(gather_calls)
        return result

    monkeypatch.setattr(fully_shard_module, "preprocess_state_dict_for_uneven_dtensor", preprocess)

    num_calls = model_id + 1
    for _ in range(num_calls):
        state_dict = optimizer.state_dict()

    assert key_gathers == [key_groups] * num_calls
    assert all(set(ranks) <= model_ranks for ranks in gather_calls)
    # The rank without optimizer state mocks exactly its own optimizer's keys.
    assert extended_keys == [{0: expected_keys}] * num_calls
    if rows:
        for key in expected_keys:
            (chunk,) = state_dict["state"][0][key]._local_tensor.__create_chunk_list__()
            assert chunk.offsets == (dp_rank, 0)
    else:
        assert state_dict["state"] == {}
