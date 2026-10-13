# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient reduction dtype tests for MFSDP v2 (``MixedPrecisionPolicy.grad_comm_dtype``).

The reduce-scatter input (partial_grad) must be allocated in ``grad_comm_dtype``, or in the
main-gradient dtype when it is unset, independently of the compute-gradient dtype.
"""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


@pytest.mark.parametrize(
    ("main_grads_dtype", "grad_comm_dtype", "expected_comm_dtype"),
    [
        # Default: reduce in the main-gradient dtype even though compute grads are bf16.
        (torch.float32, None, torch.float32),
        # Explicit override: reduce in bf16, then accumulate into fp32 main_grad.
        (torch.float32, torch.bfloat16, torch.bfloat16),
        # No main_grads_dtype: both main_grad and the reduction follow the parameter dtype.
        (None, None, torch.bfloat16),
    ],
)
def test_partial_grad_buffer_follows_grad_comm_dtype(
    distributed_setup, monkeypatch, main_grads_dtype, grad_comm_dtype, expected_comm_dtype
):
    """partial_grad is allocated in grad_comm_dtype, defaulting to the main_grad dtype."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(8, 8, device=device, dtype=torch.bfloat16)
    policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=main_grads_dtype,
        grad_comm_dtype=grad_comm_dtype,
    )
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    # Observe the dtype of the partial-grad buffer FSDP allocates during backward.
    allocated_dtypes: list[torch.dtype] = []
    original = FsdpParameterGroup.allocate_partial_grad_buffer

    def spy(self: FsdpParameterGroup):
        buffer = original(self)
        allocated_dtypes.append(buffer.dtype)
        return buffer

    monkeypatch.setattr(FsdpParameterGroup, "allocate_partial_grad_buffer", spy)

    x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    model(x).float().sum().backward()

    assert allocated_dtypes == [expected_comm_dtype], allocated_dtypes
    for name, parameter in model.named_parameters():
        assert parameter.grad.dtype == (main_grads_dtype or torch.bfloat16), name
