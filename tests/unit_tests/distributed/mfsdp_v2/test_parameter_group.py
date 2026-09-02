# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP parameter-group initialization."""

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing import assert_close

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import Flat
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def test_trainable_group_eagerly_allocates_plain_main_grad(distributed_setup):
    """A trainable group owns its persistent main-grad buffer at construction."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(8, 8, bias=False, device=device)

    with fully_shard_context(device=device) as context:
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups
    assert "main_grad" in vars(group)
    main_grad = vars(group)["main_grad"]
    assert main_grad is not None
    assert main_grad is group.main_grad
    assert main_grad.layout == group.main_weight.layout
    assert main_grad.allocation_stream == context.reduce_scatter_stream


def test_main_weight_uses_preserved_high_precision_initialization(distributed_setup):
    """The FP32 master must use a quantized parameter's preserved initialization value."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(8, 8, bias=False, device=device, dtype=torch.bfloat16)
    parameter = model.weight

    expected = torch.linspace(-0.25, 0.25, parameter.numel(), device=device).reshape_as(parameter)
    with torch.no_grad():
        parameter.zero_()
    preserved = {"value": expected.clone()}
    parameter.get_high_precision_init_val = lambda: preserved["value"]
    parameter.clear_high_precision_init_val = lambda: preserved.__setitem__("value", None)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    assert preserved["value"] is None
    assert len(model._parameter_groups) == 1
    main_weight = uneven_dtensor_to_full_tensor(
        model._parameter_groups[0].main_weight.get_dtensor(0)
    )
    assert_close(main_weight, expected.float(), atol=0, rtol=0)
