# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DCP save/load roundtrip tests for the experimental Megatron-FSDP path."""

import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict
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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.uneven_dtensor import (
    attach_uneven_dtensor_metadata,
    chunk_metadata_by_fqn,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    gather_and_compute_chunk_metadata,
    preprocess_state_dict_for_uneven_dtensor,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir

# Collectives the chunk metadata must not need. Megatron-FSDP's stable helper recovers each
# shard's offset with one all_gather_object per DTensor; this path derives it from the layout.
_COLLECTIVE_NAMES = ("all_gather", "all_gather_object", "all_gather_into_tensor", "all_reduce")


class _TinyModel(nn.Module):
    """Two shardable Linear modules; each group packs a weight and a bias unevenly."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _PackedBlock(nn.Module):
    """Three parameters whose flat packing leaves a padding gap that a smaller one fills.

    Row sizes 4, 2, and 6 give a chunk size of 12, so ``regular`` (16 elements) pads to 24 and
    ``fragment`` (4 elements) is placed in the 8-element gap it leaves. Over two ranks the buffer
    is 48 elements, so the rank boundary at 24 lands after ``fragment`` ends: rank 1 owns none of
    ``regular`` or ``fragment`` and all of ``tail``, which no canonical ``Shard(0)`` split of any
    of the three parameters describes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.regular = nn.Parameter(torch.randn(4, 4))
        self.fragment = nn.Parameter(torch.randn(2, 2))
        self.tail = nn.Parameter(torch.randn(2, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.linear(x, self.regular)
        y = nn.functional.linear(y[:, :2], self.fragment)
        return nn.functional.linear(y.repeat(1, 3), self.tail)


class _PackedModel(nn.Module):
    """A gap-filling parameter group and a Linear whose weight and bias pack unevenly."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _PackedBlock()
        self.linear = nn.Linear(2, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.block(x))


class _TiedModel(nn.Module):
    """Two Linears sharing one weight, so that weight appears in the state dict under two FQNs."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc2.weight = self.fc1.weight

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


def _build_packed_sharded(
    mesh: DeviceMesh, device: torch.device, *, zero_init: bool = False
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Shard a :class:`_PackedModel`, whose packing no canonical ``Shard(0)`` split describes."""
    model = _PackedModel().to(device=device)
    if zero_init:
        for parameter in model.parameters():
            nn.init.zeros_(parameter)
    with fully_shard_context(device=device):
        fully_shard(model.block, mesh=mesh, placements=_flat_placements())
        fully_shard(model.linear, mesh=mesh, placements=_flat_placements())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    fully_shard_optimizer(optimizer)
    return model, optimizer


def _build_tied_sharded(
    mesh: DeviceMesh, device: torch.device, *, zero_init: bool = False
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Shard a :class:`_TiedModel`, whose weight is reachable under two FQNs."""
    model = _TiedModel().to(device=device)
    if zero_init:
        for parameter in model.parameters():
            nn.init.zeros_(parameter)
    # A tied weight cannot belong to two FsdpModules, so the parent owns the whole group.
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    fully_shard_optimizer(optimizer)
    return model, optimizer


def _train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    param_dtype: torch.dtype,
    in_features: int = 8,
    out_features: int = 4,
) -> None:
    x = torch.randn(4, in_features, device=device, dtype=param_dtype)
    target = torch.randn(4, out_features, device=device, dtype=param_dtype)
    optimizer.zero_grad()
    ((model(x) - target) ** 2).mean().backward()
    optimizer.step()


def _save_through_stable_path(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: Path
) -> None:
    """Save the same state through Megatron-FSDP's stable, gather-based uneven-DTensor helper."""
    model_state_dict = get_model_state_dict(model)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    preprocess_state_dict_for_uneven_dtensor(model_state_dict)
    preprocess_state_dict_for_uneven_dtensor(optimizer_state_dict)
    dcp.save(
        {"model": model_state_dict, "optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir
    )


def _saved_chunks(checkpoint_dir: Path) -> dict[str, list[tuple[tuple[int, ...], ...]]]:
    """Return every saved tensor's chunks, as sorted (offsets, sizes) pairs keyed by entry."""
    metadata = FileSystemReader(checkpoint_dir).read_metadata()
    return {
        key: sorted((tuple(chunk.offsets), tuple(chunk.sizes)) for chunk in entry.chunks)
        for key, entry in metadata.state_dict_metadata.items()
        if getattr(entry, "chunks", None) is not None
    }


def _even_shard_rows(rows: int, world_size: int) -> list[int]:
    """Rows a canonical ``Shard(0)`` split would give each rank, in rank order."""
    shard_rows = math.ceil(rows / world_size)
    return [min(max(rows - rank * shard_rows, 0), shard_rows) for rank in range(world_size)]


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


def test_saved_chunks_match_the_stable_path(distributed_setup, tmp_path_dist_ckpt: Path) -> None:
    """The checkpoint describes the same chunks Megatron-FSDP's stable helper would describe.

    The stable helper recovers each shard's offset with one ``all_gather_object`` per DTensor.
    Deriving the offsets from the parameter layout instead has to describe exactly the same
    chunks, otherwise this path would write checkpoints that differ from the stable path's --
    including the offsets reported for empty shards, which is what makes the two byte-identical.
    """
    device = distributed_setup.device
    world_size = distributed_setup.world_size
    mesh = init_device_mesh(device.type, (world_size,))

    model, optimizer = _build_packed_sharded(mesh, device)
    _train_one_step(
        model, optimizer, device, param_dtype=torch.float32, in_features=4, out_features=16
    )

    with TempNamedDir(tmp_path_dist_ckpt / "analytic", sync=True) as analytic_dir:
        save_checkpoint(model, optimizer, analytic_dir)
        analytic_chunks = _saved_chunks(analytic_dir)
    with TempNamedDir(tmp_path_dist_ckpt / "stable", sync=True) as stable_dir:
        _save_through_stable_path(model, optimizer, stable_dir)
        stable_chunks = _saved_chunks(stable_dir)

    assert analytic_chunks.keys() == stable_chunks.keys()
    for key, chunks in analytic_chunks.items():
        assert chunks == stable_chunks[key], f"{key}: chunks differ from the stable path's"

    # Chunk metadata only matters where a canonical Shard(0) split would misplace the data, so
    # some parameter must really be sharded unevenly for the comparison above to mean anything.
    # On a single rank nothing can be misplaced: the one shard is the whole tensor.
    if world_size > 1:
        uneven = False
        for key, parameter in model.state_dict().items():
            saved_rows = sorted(sizes[0] for _, sizes in analytic_chunks[f"model.{key}"])
            # An empty shard writes nothing, so it has no chunk on either side to compare.
            even_rows = sorted(
                rows for rows in _even_shard_rows(parameter.shape[0], world_size) if rows > 0
            )
            uneven = uneven or saved_rows != even_rows
        assert uneven, "No parameter sharded unevenly, so the metadata was not exercised."


def test_saved_chunks_tile_every_parameter(distributed_setup, tmp_path_dist_ckpt: Path) -> None:
    """Every saved parameter's chunks cover its global shape exactly once.

    Overlapping or missing chunks would still produce a readable checkpoint, but one whose rows
    are duplicated or dropped on load.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    model, optimizer = _build_packed_sharded(mesh, device)
    _train_one_step(
        model, optimizer, device, param_dtype=torch.float32, in_features=4, out_features=16
    )

    with TempNamedDir(tmp_path_dist_ckpt / "tiling", sync=True) as checkpoint_dir:
        save_checkpoint(model, optimizer, checkpoint_dir)
        saved_chunks = _saved_chunks(checkpoint_dir)

    for key, parameter in model.state_dict().items():
        chunks = saved_chunks[f"model.{key}"]
        covered = 0
        for offsets, sizes in chunks:
            if sizes[0] == 0:
                continue
            assert offsets[0] == covered, f"{key} chunks are not contiguous: {chunks}"
            assert tuple(sizes[1:]) == tuple(parameter.shape[1:]), f"{key} chunk shape"
            covered += sizes[0]
        assert covered == parameter.shape[0], f"{key} chunks do not cover the global tensor"


def test_checkpoint_roundtrip_tied_parameter(distributed_setup, tmp_path_dist_ckpt: Path) -> None:
    """A tied weight is described under each of its names and restored bit-exactly.

    One ``nn.Parameter`` reachable under two names appears in the state dict under both, so the
    checkpoint has to describe both; keying the chunk metadata by a deduplicating
    ``named_parameters()`` would leave the second name without a chunk and the save would raise.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    model, optimizer = _build_tied_sharded(mesh, device)
    _train_one_step(model, optimizer, device, param_dtype=torch.float32, out_features=8)
    model_snapshot, optimizer_snapshot = _snapshot_state(model, optimizer)
    assert {"fc1.weight", "fc2.weight"} <= model_snapshot.keys(), "The tie should surface twice"

    with TempNamedDir(tmp_path_dist_ckpt / "tied", sync=True) as checkpoint_dir:
        save_checkpoint(model, optimizer, checkpoint_dir)
        saved_chunks = _saved_chunks(checkpoint_dir)
        assert {"model.fc1.weight", "model.fc2.weight"} <= saved_chunks.keys()
        assert saved_chunks["model.fc1.weight"] == saved_chunks["model.fc2.weight"]

        model, optimizer = _build_tied_sharded(mesh, device, zero_init=True)
        load_checkpoint(model, optimizer, checkpoint_dir)

    _assert_model_matches_snapshot(model, model_snapshot)
    _assert_optimizer_matches_snapshot(optimizer, optimizer_snapshot)


def test_metadata_attach_issues_no_collectives(
    distributed_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attaching the chunk metadata is rank-local.

    Dropping the stable helper's per-DTensor ``all_gather_object`` is the point of this path, and
    a saved checkpoint cannot show its absence -- :func:`dcp.save` issues collectives of its own --
    so this asserts it at the one call the save path makes.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    model, optimizer = _build_packed_sharded(mesh, device)
    _train_one_step(
        model, optimizer, device, param_dtype=torch.float32, in_features=4, out_features=16
    )
    model_state_dict = get_model_state_dict(model)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)

    def _fail(*args, **kwargs):
        raise AssertionError("Attaching chunk metadata must not issue a collective.")

    for name in _COLLECTIVE_NAMES:
        monkeypatch.setattr(dist, name, _fail)
    attach_uneven_dtensor_metadata(model, model_state_dict, optimizer_state_dict)

    assert model_state_dict.keys() == dict(model.named_parameters()).keys()


def test_empty_shard_offsets_match_the_stable_path(distributed_setup) -> None:
    """An empty shard is reported at the same offset the stable helper would report.

    A rank that owns no rows of a parameter writes nothing, so this offset never reaches the
    checkpoint: DCP discards a zero-size chunk, and
    :func:`test_saved_chunks_match_the_stable_path` therefore cannot see it. Reporting it
    anyway keeps the two implementations comparable chunk for chunk, which is what the
    byte-identity claim rests on, so it is asserted directly against the stable helper here.
    """
    device = distributed_setup.device
    world_size = distributed_setup.world_size
    mesh = init_device_mesh(device.type, (world_size,))

    model, _ = _build_packed_sharded(mesh, device)
    metadata_by_fqn = chunk_metadata_by_fqn(model)

    empty_offsets = []
    for fqn, parameter in model.named_parameters():
        chunk = metadata_by_fqn[fqn]
        expected = gather_and_compute_chunk_metadata(parameter)
        assert tuple(chunk.offsets) == tuple(expected.offsets), f"{fqn} chunk offsets"
        assert tuple(chunk.sizes) == tuple(expected.sizes), f"{fqn} chunk sizes"
        if chunk.sizes[0] == 0:
            empty_offsets.append(chunk.offsets[0])

    # An empty chunk at offset 0 says nothing: that is what a canonical split would report too.
    # Only the packing gap makes a rank skip rows it does not own, so require that case.
    if world_size > 1:
        nonzero_flags = [None] * world_size
        dist.all_gather_object(nonzero_flags, any(offset > 0 for offset in empty_offsets))
        assert any(nonzero_flags), "No rank held an empty shard at a non-zero offset."
