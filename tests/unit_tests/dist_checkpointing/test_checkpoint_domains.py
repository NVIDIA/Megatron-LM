# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A torch_dist checkpoint can be saved and loaded by a subgroup of WORLD."""

import shutil
import time

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader

from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from tests.unit_tests.test_utilities import Utils


def _wait_for_marker(marker):
    deadline = time.monotonic() + 60
    while not marker.exists():
        if time.monotonic() > deadline:
            pytest.fail(f"Checkpoint domain could not complete independently: {marker.name}")
        time.sleep(0.1)


@pytest.mark.parametrize("fully_parallel", [False, True])
def test_independent_checkpoint_domains(tmp_path_dist_ckpt, fully_parallel):
    """Two rank groups write complete checkpoints with the same tensor keys."""
    Utils.initialize_distributed()
    if Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip("Requires an even world size of at least four ranks")

    group_size = Utils.world_size // 2
    groups = [
        dist.new_group(ranks=list(range(start, start + group_size))) for start in (0, group_size)
    ]
    domain_id = Utils.rank // group_size
    group = groups[domain_id]
    local_rank = dist.get_rank(group=group)
    # Run the domains one after the other. A hidden WORLD collective in either
    # checkpoint path would block the first domain until this wait times out.
    first_domain_done = tmp_path_dist_ckpt / f"first_domain_done_{fully_parallel}"
    second_domain_done = tmp_path_dist_ckpt / f"second_domain_done_{fully_parallel}"
    if Utils.rank == 0:
        first_domain_done.unlink(missing_ok=True)
        second_domain_done.unlink(missing_ok=True)
    dist.barrier()
    if domain_id == 1:
        _wait_for_marker(first_domain_done)

    checkpoint_dir = tmp_path_dist_ckpt / f"checkpoint_domain_{fully_parallel}_{domain_id}"
    if local_rank == 0:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)
    dist.barrier(group=group)

    device = torch.device("cuda", torch.cuda.current_device())
    tensor_value = float(domain_id + 1)
    state_dict = {
        "tensor": ShardedTensor.from_rank_offsets(
            "tensor",
            torch.full((2,), tensor_value, device=device),
            (0, local_rank, group_size),
            replica_id=Utils.rank,
        ),
        "replicated": ShardedTensor.from_rank_offsets(
            "replicated", torch.full((2,), tensor_value, device=device), replica_id=Utils.rank
        ),
        "rank_state": ShardedObject(
            "rank_state",
            {"domain": domain_id, "rank": local_rank},
            (group_size,),
            (local_rank,),
            replica_id=Utils.rank,
        ),
        "shared_object": ShardedObject(
            "shared_object", {"domain": domain_id}, (1,), (0,), replica_id=Utils.rank
        ),
        "domain": domain_id,
    }
    save_strategy = TorchDistSaveShardedStrategy(process_group=group)
    if fully_parallel:
        save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, group)
    save(
        state_dict,
        checkpoint_dir,
        save_strategy,
        process_group=group,
        preferred_writer_ranks=[domain_id * group_size],
        verify_integrity=True,
    )

    load_state_dict = {
        "tensor": ShardedTensor.from_rank_offsets(
            "tensor",
            torch.zeros(2, device=device),
            (0, local_rank, group_size),
            replica_id=Utils.rank,
        ),
        "replicated": ShardedTensor.from_rank_offsets(
            "replicated", torch.zeros(2, device=device), replica_id=Utils.rank
        ),
        "rank_state": ShardedObject(
            "rank_state", None, (group_size,), (local_rank,), replica_id=Utils.rank
        ),
        "shared_object": ShardedObject("shared_object", None, (1,), (0,), replica_id=Utils.rank),
    }
    load_strategy = TorchDistLoadShardedStrategy()
    if fully_parallel:
        load_strategy = FullyParallelLoadStrategyWrapper(load_strategy, group)
    restored = load(
        load_state_dict,
        checkpoint_dir,
        load_strategy,
        process_group=group,
        preferred_reader_ranks=[domain_id * group_size],
        verify_integrity=True,
    )
    assert restored["domain"] == domain_id
    assert restored["rank_state"] == {"domain": domain_id, "rank": local_rank}
    assert restored["shared_object"] == {"domain": domain_id}
    torch.testing.assert_close(restored["tensor"], torch.full((2,), tensor_value, device=device))
    torch.testing.assert_close(
        restored["replicated"], torch.full((2,), tensor_value, device=device)
    )

    if local_rank == 0:
        metadata = FileSystemReader(checkpoint_dir).read_metadata()
        assert all(
            (checkpoint_dir / storage.relative_path).is_file()
            for storage in metadata.storage_data.values()
        )
    dist.barrier(group=group)
    if Utils.rank == 0:
        first_domain_done.touch()
    if Utils.rank == group_size:
        second_domain_done.touch()
    if domain_id == 0:
        _wait_for_marker(second_domain_done)


def test_async_checkpoint_domain_rejected(tmp_path_dist_ckpt):
    """NVRx finalization uses WORLD collectives, so fail before any subgroup I/O."""
    Utils.initialize_distributed()
    if Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip("Requires an even world size of at least four ranks")

    group_size = Utils.world_size // 2
    groups = [
        dist.new_group(ranks=list(range(start, start + group_size))) for start in (0, group_size)
    ]
    group = groups[Utils.rank // group_size]
    with pytest.raises(NotImplementedError, match="NVRx AsyncRequest finalization"):
        save(
            {},
            tmp_path_dist_ckpt / "unsupported_async_domain",
            TorchDistSaveShardedStrategy(process_group=group),
            process_group=group,
            async_sharded_save=True,
        )
