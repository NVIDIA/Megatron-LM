# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DCP save/load roundtrip tests for the experimental Megatron-FSDP path."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir


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


def _build_sharded(
    mesh: DeviceMesh, device: torch.device, *, param_dtype: torch.dtype, zero_init: bool
) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = _TinyModel().to(device=device, dtype=param_dtype)
    if zero_init:
        # Zero the destination weights so they are obviously different from the saved (trained)
        # source; a correct load must overwrite them.
        for parameter in model.parameters():
            nn.init.zeros_(parameter)
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    # main_weight is fp32 by default, so a bf16 model feeds the fp32 optimizer bf16 grads; the
    # adapter casts them around each step.
    fully_shard_optimizer(optimizer)
    return model, optimizer


def _train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    param_dtype: torch.dtype,
) -> None:
    x = torch.randn(4, 8, device=device, dtype=param_dtype)
    target = torch.randn(4, 4, device=device, dtype=param_dtype)
    optimizer.zero_grad()
    ((model(x) - target) ** 2).mean().backward()
    optimizer.step()


def _assert_tensors_identical(expected: torch.Tensor, actual: torch.Tensor, what: str) -> None:
    """Assert two tensors are bit-identical, checking DTensor global metadata when applicable.

    A checkpoint roundtrip must reproduce the values exactly, so tolerances are zero. For DTensors
    it must also reproduce the *global* view: an entry whose global shape or placement changed would
    be silently wrong even if this rank's local shard happens to match.
    """
    assert type(expected) is type(actual), f"{what}: {type(expected)} became {type(actual)}"
    if isinstance(expected, DTensor):
        assert (
            expected.shape == actual.shape
        ), f"{what}: global shape {expected.shape} != {actual.shape}"
        assert expected.placements == actual.placements, f"{what}: placements changed"
        assert expected.device_mesh == actual.device_mesh, f"{what}: device mesh changed"
        expected, actual = expected.to_local(), actual.to_local()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"{what}: value mismatch")


def _snapshot_state(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[dict[str, torch.Tensor], dict[int, dict]]:
    """Clone the model weights and optimizer state, keyed as their state dicts are.

    DTensor entries are cloned as DTensors so the comparison can check the global shape and
    placements, not just this rank's local shard.
    """
    model_snapshot = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer_snapshot: dict[int, dict] = {}
    for index, state in optimizer.state_dict()["state"].items():
        optimizer_snapshot[index] = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in state.items()
        }
    return model_snapshot, optimizer_snapshot


def _assert_model_matches_snapshot(
    model: nn.Module, model_snapshot: dict[str, torch.Tensor]
) -> bool:
    """Assert the model's weights equal the snapshot.

    Returns:
        bool: whether this rank held at least one non-empty local shard. The caller all-gathers
        this flag across ranks and asserts that some rank made a real (non-empty) comparison, so
        the test cannot pass vacuously when a rank happens to own only empty shards.
    """
    current = model.state_dict()
    assert model_snapshot.keys() == current.keys()
    # Tracks whether this rank owned any real (non-empty) shard data; returned for the caller's
    # cross-rank "at least one rank compared something" check.
    local_nonempty = False
    for key, expected in model_snapshot.items():
        assert isinstance(current[key], DTensor), f"{key} should rest as a DTensor"
        _assert_tensors_identical(expected, current[key], f"model[{key}]")
        local_nonempty = local_nonempty or expected.to_local().numel() > 0
    return local_nonempty


def _assert_optimizer_matches_snapshot(
    optimizer: torch.optim.Optimizer, optimizer_snapshot: dict[int, dict]
) -> None:
    """Assert the optimizer's state equals the snapshot."""
    current = optimizer.state_dict()["state"]
    assert optimizer_snapshot.keys() == current.keys()
    for index, expected_state in optimizer_snapshot.items():
        for key, expected in expected_state.items():
            actual = current[index][key]
            if torch.is_tensor(expected):
                _assert_tensors_identical(expected, actual, f"optim[{index}][{key}]")
            else:
                assert expected == actual, f"optim[{index}][{key}] scalar mismatch"


def _assert_checkpoint_records_global_shapes(checkpoint_dir: Path, model: nn.Module) -> None:
    """Assert the saved checkpoint describes every parameter by its full global shape.

    A checkpoint of a sharded model must describe the assembled tensor, not this rank's fragment,
    so a reader that reshards differently sees the right geometry.

    Only the global sizes are checked. The per-chunk offsets deliberately are not: dropping the
    uneven-DTensor metadata does not make them look wrong -- DCP then records exactly the canonical
    even-``Shard(0)`` chunks (two 8-row chunks for a 16-row parameter that the ranks really split
    9/7), which tile the tensor perfectly while the bytes underneath belong to different rows. That
    self-consistency is why the corruption is silent, and why the value comparison after the load is
    what actually guards it.
    """
    metadata = FileSystemReader(checkpoint_dir).read_metadata()
    for key, value in model.state_dict().items():
        entry = metadata.state_dict_metadata[f"model.{key}"]
        assert tuple(entry.size) == tuple(value.shape), f"model.{key}: saved {entry.size}"


@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_checkpoint_roundtrip_flat_dp(
    distributed_setup, tmp_path_dist_ckpt: Path, param_dtype: torch.dtype
) -> None:
    """Saving then loading a flat-DP sharded model+optimizer restores state bit-exactly.

    The fc1 group packs ``weight (16, 8)`` and ``bias (16,)`` into one flat buffer, so with >=2
    ranks the per-rank shards do not tile like canonical ``Shard(0)`` (e.g. one rank owns no bias
    rows), which is what exercises the uneven-DTensor metadata path in :func:`save_checkpoint`. On
    a single rank the sharding degenerates to one full shard and this is a plain roundtrip sanity
    check. With a bf16 model the optimizer's ``main_weight`` stays fp32, covering the
    mixed-precision master-weight path.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    # Source: train one step so weights and optimizer state are non-trivial, then save.
    model, optimizer = _build_sharded(mesh, device, param_dtype=param_dtype, zero_init=False)
    _train_one_step(model, optimizer, device, param_dtype=param_dtype)
    model_snapshot, optimizer_snapshot = _snapshot_state(model, optimizer)

    with TempNamedDir(tmp_path_dist_ckpt / f"ckpt_{param_dtype}", sync=True) as checkpoint_dir:
        save_checkpoint(model, optimizer, checkpoint_dir)
        _assert_checkpoint_records_global_shapes(checkpoint_dir, model)

        # Destination: zero-initialized, so a correct load is non-trivial.
        model, optimizer = _build_sharded(mesh, device, param_dtype=param_dtype, zero_init=True)
        load_checkpoint(model, optimizer, checkpoint_dir)

    local_nonempty = _assert_model_matches_snapshot(model, model_snapshot)
    _assert_optimizer_matches_snapshot(optimizer, optimizer_snapshot)

    # At least one rank must have held non-empty local shards for the check to be meaningful.
    nonempty_flags = [None] * distributed_setup.world_size
    torch.distributed.all_gather_object(nonempty_flags, local_nonempty)
    assert any(nonempty_flags), "All ranks had empty local shards."
