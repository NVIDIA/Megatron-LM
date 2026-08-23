# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with an already-EP-sharded grouped expert weight.

Covers the representation NeMo AutoModel's EP parallelizer produces: a single contiguous
3D expert weight ``[E, h_in, h_out]`` replaced by
``nn.Parameter(distribute_tensor(param, ep_mesh, [Shard(0)]))``. mFSDP receives the full
``(ep, dp)`` mesh, shards only the parameter's EP-local shard, and re-expresses the result
on the full mesh so the global expert index survives into the optimizer and checkpoints.

See https://github.com/NVIDIA/Megatron-LM/issues/6600.
"""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir

_EP_SIZE = 2
_NUM_EXPERTS = 4
_H_IN = 8
_H_OUT = 4

# dp_axes names the FSDP axis inside the full (ep, dp) mesh; the leftover axis is EP.
_FLAT_DP = Placements(
    dp_axes=["dp"], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
)


class GroupedExperts(nn.Module):
    """One contiguous 3D expert weight, consumed through ``to_local()`` like grouped GEMMs."""

    def __init__(self, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                _NUM_EXPERTS, _H_IN, _H_OUT, generator=generator, device=device, dtype=torch.float32
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this rank's local experts to its local slice of the batch."""
        weight = self.weight
        local_weight = weight.to_local() if isinstance(weight, DTensor) else weight
        return torch.bmm(x, local_weight)


def _skip_unless_composable(world_size: int) -> int:
    if world_size < _EP_SIZE * 2 or world_size % _EP_SIZE != 0:
        pytest.skip(f"This test requires a multiple of {_EP_SIZE * 2} ranks.")
    return world_size // _EP_SIZE


def _build_ep_sharded_model(device: torch.device, mesh) -> GroupedExperts:
    # Identical seed on every rank so the pre-EP global weight is identical, matching a
    # real load-then-shard flow.
    generator = torch.Generator(device=device).manual_seed(1234)
    model = GroupedExperts(generator, device)
    sharded = distribute_tensor(model.weight.data, mesh["ep"], [Shard(0)])
    dist_param = nn.Parameter(sharded)
    dist_param.requires_grad = model.weight.requires_grad
    model.weight = dist_param
    return model


def test_ep_sharded_dtensor_expert_weight_shards_and_keeps_global_spec(distributed_setup):
    """fully_shard accepts an EP-sharded DTensor and keeps the global expert extent."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    dp_size = _skip_unless_composable(world_size)

    mesh = init_device_mesh("cuda", (_EP_SIZE, dp_size), mesh_dim_names=("ep", "dp"))
    model = _build_ep_sharded_model(device, mesh)
    assert isinstance(model.weight.data, DTensor)
    assert tuple(model.weight.to_local().shape) == (_NUM_EXPERTS // _EP_SIZE, _H_IN, _H_OUT)

    with fully_shard_context(device=device, use_symmetric_memory=False):
        fully_shard(model, mesh=mesh, placements=_FLAT_DP)

    sharded = model.weight
    assert isinstance(sharded.data, DTensor), "Expected the sharded parameter to be a DTensor."
    # The global expert extent must survive, or a checkpoint records only local experts.
    assert tuple(sharded.shape) == (
        _NUM_EXPERTS,
        _H_IN,
        _H_OUT,
    ), f"Expected the global expert shape, got {tuple(sharded.shape)}."
    assert sharded.device_mesh.mesh_dim_names == (
        "ep",
        "dp",
    ), f"Expected the full (ep, dp) mesh, got {sharded.device_mesh.mesh_dim_names}."
    # EP shards the expert axis; FSDP shards the EP-local experts underneath it. Mesh-axis
    # order is the nesting order, so both are plain Shard(0) and neither is strided.
    ep_placement, dp_placement = sharded.placements
    assert isinstance(ep_placement, Shard) and ep_placement.dim == 0, ep_placement
    assert isinstance(dp_placement, Shard) and dp_placement.dim == 0, dp_placement


def test_ep_sharded_dtensor_expert_weight_trains(distributed_setup):
    """A step through the sharded model updates the expert weights."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    dp_size = _skip_unless_composable(world_size)

    mesh = init_device_mesh("cuda", (_EP_SIZE, dp_size), mesh_dim_names=("ep", "dp"))
    model = _build_ep_sharded_model(device, mesh)
    with fully_shard_context(device=device, use_symmetric_memory=False):
        fully_shard(model, mesh=mesh, placements=_FLAT_DP)

    before = model.weight.to_local().detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)

    local_experts = _NUM_EXPERTS // _EP_SIZE
    x = torch.randn(local_experts, 3, _H_IN, device=device, dtype=torch.float32)
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()

    after = model.weight.to_local().detach()
    if before.numel() > 0:
        # This is what catches a silently-unwired buffer: losses can look fine while the
        # optimizer updates storage the forward pass never reads.
        assert not torch.equal(before, after), "Expert weights did not change after a step."


def test_ep_sharded_dtensor_requires_an_expert_mesh_axis(distributed_setup):
    """An EP-sharded parameter with no non-DP axis to place it on is rejected clearly."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    _skip_unless_composable(world_size)

    mesh = init_device_mesh("cuda", (_EP_SIZE, world_size // _EP_SIZE), mesh_dim_names=("ep", "dp"))
    model = _build_ep_sharded_model(device, mesh)

    # A DP-only mesh has no axis left for the parameter's existing EP sharding.
    dp_only = mesh["dp"]
    flat_all_axes = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with pytest.raises(ValueError, match="no non-data-parallel axis"):
        with fully_shard_context(device=device, use_symmetric_memory=False):
            fully_shard(model, mesh=dp_only, placements=flat_all_axes)


def test_dp_before_ep_mesh_order_is_rejected(distributed_setup):
    """A (dp, ep) mesh cannot be described without a strided placement, so reject it."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    dp_size = _skip_unless_composable(world_size)

    # Axis order reversed: FSDP axis first. Plain Shard(0) on both axes would describe a
    # different physical layout than EP-outer/FSDP-inner and would corrupt checkpoints.
    mesh = init_device_mesh("cuda", (dp_size, _EP_SIZE), mesh_dim_names=("dp", "ep"))
    model = _build_ep_sharded_model(device, mesh)
    with pytest.raises(NotImplementedError, match="must precede the data-parallel axes"):
        with fully_shard_context(device=device, use_symmetric_memory=False):
            fully_shard(model, mesh=mesh, placements=_FLAT_DP)


def test_ep_sharded_dtensor_checkpoint_records_global_expert_shape(
    distributed_setup, tmp_path_dist_ckpt: Path
):
    """A DCP checkpoint must describe the assembled [E, ...] tensor, not local experts.

    This is the assertion that catches a lost expert axis. Training and loss comparisons do
    not: an FSDP-only spec is perfectly self-consistent within one topology and only shows up
    as corruption when a reader reshards with a different EP degree.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    dp_size = _skip_unless_composable(world_size)

    mesh = init_device_mesh("cuda", (_EP_SIZE, dp_size), mesh_dim_names=("ep", "dp"))
    model = _build_ep_sharded_model(device, mesh)
    with fully_shard_context(device=device, use_symmetric_memory=False):
        fully_shard(model, mesh=mesh, placements=_FLAT_DP)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    fully_shard_optimizer(optimizer)

    # Train one step so the saved state is non-trivial.
    local_experts = _NUM_EXPERTS // _EP_SIZE
    x = torch.randn(local_experts, 3, _H_IN, device=device, dtype=torch.float32)
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()

    saved = model.state_dict()["weight"]
    assert tuple(saved.shape) == (_NUM_EXPERTS, _H_IN, _H_OUT)

    with TempNamedDir(tmp_path_dist_ckpt / "ckpt_ep_dtensor", sync=True) as checkpoint_dir:
        save_checkpoint(model, optimizer, checkpoint_dir)
        dist.barrier(device_ids=[device.index])

        metadata = FileSystemReader(checkpoint_dir).read_metadata()
        entry = metadata.state_dict_metadata["model.weight"]
        assert tuple(entry.size) == (_NUM_EXPERTS, _H_IN, _H_OUT), (
            f"Checkpoint recorded {tuple(entry.size)} for the grouped expert weight; expected the "
            f"global shape ({_NUM_EXPERTS}, {_H_IN}, {_H_OUT}). A local-experts shape means the "
            "global expert index was lost and the checkpoint cannot be resharded."
        )

        # A correct load must overwrite an obviously-different destination.
        before = model.weight.to_local().detach().clone()
        with torch.no_grad():
            model.weight.to_local().zero_()
        load_checkpoint(model, optimizer, checkpoint_dir)
        restored = model.weight.to_local().detach()
        if before.numel() > 0:
            torch.testing.assert_close(restored, before)
