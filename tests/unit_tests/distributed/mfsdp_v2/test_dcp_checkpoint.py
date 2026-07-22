# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DCP save/load roundtrip tests for the experimental Megatron-FSDP path."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_optimizer,
    load_dcp_checkpoint,
    save_dcp_checkpoint,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


class _TinyModel(nn.Module):
    """Two shardable Linear modules; each group packs a weight and a bias unevenly."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _build_sharded(seed, mesh, device, *, param_dtype, mp_policy):
    torch.manual_seed(seed)
    model = _TinyModel().to(device=device, dtype=param_dtype)
    fully_shard(
        model.fc1, mesh=mesh, placements=_flat_placements(), mixed_precision_policy=mp_policy
    )
    fully_shard(
        model.fc2, mesh=mesh, placements=_flat_placements(), mixed_precision_policy=mp_policy
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    # Mixed precision feeds the optimizer main-weight-dtype params with FSDP-dtype grads; the
    # adapter casts them around each step.
    fully_shard_optimizer(optimizer)
    return model, optimizer


def _train_one_step(model, optimizer, device, *, param_dtype):
    x = torch.randn(4, 8, device=device, dtype=param_dtype)
    target = torch.randn(4, 4, device=device, dtype=param_dtype)
    optimizer.zero_grad()
    ((model(x) - target) ** 2).mean().backward()
    optimizer.step()


def _to_local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _snapshot(model, optimizer):
    model_snap = {
        key: _to_local(value).detach().clone()
        for key, value in model.state_dict().items()
        if not key.endswith("_extra_state")
    }
    optim_snap = {}
    for index, state in optimizer.state_dict()["state"].items():
        optim_snap[index] = {
            key: (_to_local(value).detach().clone() if torch.is_tensor(value) else value)
            for key, value in state.items()
        }
    return model_snap, optim_snap


def _assert_matches(snapshot, model, optimizer):
    model_snap, optim_snap = snapshot
    local_nonempty = False

    current_model = {
        key: value for key, value in model.state_dict().items() if not key.endswith("_extra_state")
    }
    assert model_snap.keys() == current_model.keys()
    for key, expected in model_snap.items():
        assert isinstance(current_model[key], DTensor), f"{key} should rest as a DTensor"
        actual = _to_local(current_model[key])
        assert (
            expected.shape == actual.shape
        ), f"model[{key}] shape {expected.shape} != {actual.shape}"
        assert torch.equal(expected, actual), f"model[{key}] value mismatch after roundtrip"
        local_nonempty = local_nonempty or expected.numel() > 0

    current_state = optimizer.state_dict()["state"]
    assert optim_snap.keys() == current_state.keys()
    for index, expected_state in optim_snap.items():
        for key, expected in expected_state.items():
            actual = current_state[index][key]
            if torch.is_tensor(expected):
                actual = _to_local(actual)
                assert expected.shape == actual.shape, f"optim[{index}][{key}] shape mismatch"
                assert torch.equal(expected, actual), f"optim[{index}][{key}] value mismatch"
            else:
                assert expected == actual, f"optim[{index}][{key}] scalar mismatch"

    return local_nonempty


# (param_dtype, mixed_precision_policy): plain fp32, and bf16 compute with fp32 master weights.
_CONFIGS = {
    "fp32": (torch.float32, None),
    "bf16_fp32_master": (torch.bfloat16, MixedPrecisionPolicy(main_params_dtype=torch.float32)),
}


@pytest.mark.parametrize("config", list(_CONFIGS), ids=list(_CONFIGS))
def test_dcp_roundtrip_flat_dp(distributed_setup, shared_checkpoint_dir, config):
    """Saving then loading a flat-DP sharded model+optimizer restores state bit-exactly.

    The fc1 group packs ``weight (16, 8)`` and ``bias (16,)`` into one flat buffer, so per-rank
    shards do not tile like canonical ``Shard(0)`` (e.g. one rank owns no bias rows). This
    exercises the uneven-DTensor metadata path in :func:`save_dcp_checkpoint`.
    """
    if distributed_setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    device = distributed_setup.device
    param_dtype, mp_policy = _CONFIGS[config]
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    # Source: train one step so weights and optimizer state are non-trivial, then save.
    model, optimizer = _build_sharded(
        1234, mesh, device, param_dtype=param_dtype, mp_policy=mp_policy
    )
    _train_one_step(model, optimizer, device, param_dtype=param_dtype)
    snapshot = _snapshot(model, optimizer)
    save_dcp_checkpoint(model, optimizer, shared_checkpoint_dir)

    # Destination: a differently-initialized model+optimizer, so a correct load is non-trivial.
    model2, optimizer2 = _build_sharded(
        4321, mesh, device, param_dtype=param_dtype, mp_policy=mp_policy
    )
    load_dcp_checkpoint(model2, optimizer2, shared_checkpoint_dir)

    local_nonempty = _assert_matches(snapshot, model2, optimizer2)

    # At least one rank must have held real (non-empty) local shards for the check to be meaningful.
    nonempty_flags = [None] * distributed_setup.world_size
    torch.distributed.all_gather_object(nonempty_flags, local_nonempty)
    assert any(nonempty_flags), "All ranks had empty local shards."
