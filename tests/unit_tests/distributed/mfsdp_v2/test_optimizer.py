# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP optimizer behavior."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig


class TinyModel(nn.Module):
    """Small model with two separately shardable units."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""
        return self.fc2(self.relu(self.fc1(x)))


class ParamGradCastingMixedPrecisionOptimizer(MixedPrecisionOptimizer):
    """Test optimizer that makes param.grad dtype-compatible with param."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> None:
        super().__init__(optimizer, config, grad_scaler=None, init_state_fn=lambda *_args: None)
        self.is_stub_optimizer = False

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def prepare_grads(self) -> bool:
        for parameter in self.get_parameters():
            grad = parameter.grad.to(dtype=parameter.dtype)
            parameter.grad = None
            parameter.grad_dtype = grad.dtype
            parameter.grad = grad
        # Return False because this test optimizer does not check for inf/nan.
        return False

    def step_with_ready_grads(self) -> bool:
        self.optimizer.step()
        return True

    def reload_model_params(self, state_dict=None) -> None:
        pass

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict) -> None:
        raise NotImplementedError

    def sharded_state_dict(self, *args, **kwargs):
        raise NotImplementedError


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _build_model_with_param_grad_dtype_mismatch(
    mesh: DeviceMesh, device: torch.device
) -> TinyModel:
    torch.manual_seed(2026)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    # These are the defaults, but spell them out so the test clearly exercises
    # mismatched parameter and gradient precision.
    mixed_precision_policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32, main_grads_dtype=torch.bfloat16
    )
    fully_shard(
        model.fc1,
        mesh=mesh,
        placements=_flat_placements(),
        mixed_precision_policy=mixed_precision_policy,
    )
    fully_shard(
        model.fc2,
        mesh=mesh,
        placements=_flat_placements(),
        mixed_precision_policy=mixed_precision_policy,
    )
    return model


def test_adam_without_adapter_raises_precision_error(distributed_setup):
    """Raw Adam should fail on mixed-precision FSDP parameters without the adapter."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (world_size,))
    model = _build_model_with_param_grad_dtype_mismatch(mesh, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()

    with pytest.raises(RuntimeError, match="same device and the same dtype"):
        optimizer.step()


def test_fused_adam_without_adapter_accepts_mismatched_grads(distributed_setup):
    """TE FusedAdam should handle mixed-precision FSDP grads without the adapter."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (world_size,))
    model = _build_model_with_param_grad_dtype_mismatch(mesh, device)
    optimizer = FusedAdam(model.parameters(), lr=0.01)

    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.dtype != parameter.grad.dtype

    params_before_step = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()

    assert any(
        not torch.equal(parameter_before, parameter.detach())
        for parameter_before, parameter in zip(params_before_step, model.parameters())
    )


def test_mixed_precision_optimizer_with_adam_casts_mismatched_grads(distributed_setup):
    """A MixedPrecisionOptimizer subclass can make vanilla Adam dtype-compatible."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (world_size,))
    model = _build_model_with_param_grad_dtype_mismatch(mesh, device)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer_config = OptimizerConfig(optimizer="adam", lr=0.01, bf16=True, clip_grad=0.0)
    optimizer = ParamGradCastingMixedPrecisionOptimizer(base_optimizer, optimizer_config)

    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.dtype != parameter.grad.dtype

    params_before_step = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()

    assert any(
        not torch.equal(parameter_before, parameter.detach())
        for parameter_before, parameter in zip(params_before_step, model.parameters())
    )
