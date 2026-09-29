# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU/Gloo DCP regression for non-GTP checkpoints loaded into padded TP x GTP shards."""

import pytest
import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor
from megatron.core.utils import _make_gtp_logical_sharded_tensor
from tests.unit_tests.dist_checkpointing.cpu_test_utils import TorchDistCPUSaveShardedStrategy

pytestmark = pytest.mark.internal


@pytest.mark.parametrize("tp_axis", [0, 1])
@pytest.mark.parametrize("pad_length", [2, 4])
def test_non_gtp_checkpoint_loads_into_padded_model_and_optimizer(
    cpu_default_process_group, tmp_path_dist_ckpt, tp_axis, pad_length
):
    """Real DCP reads the unpadded global tensor, including an entirely padded trailing rank."""
    world_size = dist.get_world_size()
    assert world_size >= 4 and world_size % 4 == 0, "Run with four or eight torchrun ranks"
    rank = dist.get_rank()
    tp_rank, gtp_rank, dp_rank = rank % 4 // 2, rank % 2, rank // 4
    logical_rows, columns = 8 - pad_length, 3
    shape = (logical_rows * 2, columns) if tp_axis == 0 else (logical_rows, columns * 2)
    reference = torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.float32).reshape(shape)
    source_data = reference.chunk(2, dim=tp_axis)[tp_rank].contiguous()
    source = ShardedTensor.from_rank_offsets(
        "weight", source_data, (tp_axis, tp_rank, 2), replica_id=(0, 0, dp_rank * 2 + gtp_rank)
    )
    source_state = make_sharded_optimizer_tensor(
        source, source_data + 1000, "optimizer.state.exp_avg"
    )
    checkpoint_dir = tmp_path_dist_ckpt / f"padding-axis{tp_axis}-pad{pad_length}"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(
        {"weight": source, "state": source_state},
        checkpoint_dir,
        sharded_strategy=TorchDistCPUSaveShardedStrategy("torch_dist", 1),
    )

    target_data = torch.zeros(4, columns)
    target = _make_gtp_logical_sharded_tensor(
        target_data,
        "weight",
        tp_axis=tp_axis,
        tp_rank=tp_rank,
        tp_size=2,
        gtp_rank=gtp_rank,
        gtp_remat_size=2,
        pad_length=pad_length,
        prepend_offsets=(),
        prepend_axis_num=0,
        other_offsets=((tp_axis, tp_rank, 2),) if tp_axis else (),
        replica_id=(0, 0, dp_rank),
    )
    target_state_data = torch.zeros_like(target_data)
    target_state = make_sharded_optimizer_tensor(
        target, target_state_data, "optimizer.state.exp_avg"
    )
    loaded = dist_checkpointing.load({"weight": target, "state": target_state}, checkpoint_dir)
    keep = target.local_shape[0]
    row = (
        tp_rank * logical_rows + min(gtp_rank * 4, logical_rows)
        if tp_axis == 0
        else min(gtp_rank * 4, logical_rows)
    )
    col = tp_rank * columns if tp_axis else 0
    expected = reference[row : row + keep, col : col + columns]
    for name, delta in (("weight", 0), ("state", 1000)):
        assert loaded[name].shape == target_data.shape
        torch.testing.assert_close(loaded[name][:keep], expected + delta, rtol=0, atol=0)
        assert torch.count_nonzero(loaded[name][keep:]) == 0
