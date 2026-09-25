# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for Muon's unsplit grouped-expert optimizer checkpoint schema."""

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.optimizer.optimizer import _backfill_gtp_sharded_param_map
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.utils import _make_gtp_logical_sharded_tensor


@pytest.mark.parametrize("gtp_rank", [0, 1])
@pytest.mark.parametrize("expert_rank", [0, 1])
@pytest.mark.parametrize("pad_length", [0, 2])
@pytest.mark.parametrize("singleton", [False, True])
def test_grouped_muon_reuses_physical_metadata_after_prefix_replacement(
    gtp_rank, expert_rank, pad_length, singleton
):
    """A physical optimizer shard retains full global shape, expert offset, and unsplit key."""
    logical_rows = 8 - pad_length
    live = torch.nn.Parameter(torch.zeros(4, 3))
    live.is_gtp_weight_remat = True
    live.allreduce = False
    # This name deliberately differs from the checkpoint key, as TEGroupedLinear names do.
    live._debug_name = f"module.layers.0.experts.linear_fc1.weight{expert_rank}"
    offsets = () if singleton else ((0, expert_rank, 2),)
    source = _make_gtp_logical_sharded_tensor(
        live,
        "linear_fc1.weight",
        tp_axis=0,
        tp_rank=0,
        tp_size=1,
        gtp_rank=gtp_rank,
        gtp_remat_size=2,
        pad_length=pad_length,
        prepend_offsets=offsets,
        prepend_axis_num=len(offsets),
        other_offsets=(),
        replica_id=(0, 0, 0),
    )
    gathered = ShardedTensor.from_rank_offsets(
        source.key,
        torch.zeros(logical_rows, 3),
        *offsets,
        prepend_axis_num=len(offsets),
        replica_id=(0, gtp_rank, 0),
    )
    factory = apply_swiglu_sharded_factory(gathered, offsets, singleton)
    factory.gtp_source_param = live
    factory.gtp_source_sharded_tensor = source
    model_state = {"linear_fc1.weight0": factory}
    replace_prefix_for_sharding(model_state, "linear_fc1.", "layers.0.experts.linear_fc1.")

    param_map = {}
    _backfill_gtp_sharded_param_map(param_map, [[live]], model_state)
    physical = param_map[0]
    assert isinstance(physical, ShardedTensor)
    assert physical.key == "layers.0.experts.linear_fc1.weight"
    assert physical.global_shape == ((2,) if offsets else ()) + (logical_rows, 3)
    assert physical.global_offset == ((expert_rank,) if offsets else ()) + (gtp_rank * 4, 0)
    assert physical.replica_id == (0, 0, 0)
    assert source.key == "linear_fc1.weight"

    state = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    optim_entry = make_sharded_optimizer_tensor(physical, state, "optimizer.state.momentum_buffer")
    assert isinstance(optim_entry, ShardedTensor)
    assert optim_entry.key == "optimizer.state.momentum_buffer.layers.0.experts.linear_fc1.weight"
    assert optim_entry.global_shape == physical.global_shape
    assert optim_entry.global_offset == physical.global_offset
    assert optim_entry.local_shape == source.local_shape
    if source.local_shape[0] < live.shape[0]:
        assert physical.gtp_pad_src is live
        assert optim_entry.gtp_pad_src is state
    optim_entry.validate_metadata_integrity()


def test_raw_gathered_factory_cannot_be_reused_for_an_expert_muon_state():
    """A source-parameter backlink alone cannot make gathered model metadata shard-compatible."""
    live = torch.nn.Parameter(torch.zeros(4, 3))
    live.is_gtp_weight_remat = True
    live.allreduce = False
    live._debug_name = "module.experts.linear_fc1.weight0"
    gathered = ShardedTensor.from_rank_offsets("experts.linear_fc1.weight", torch.zeros(8, 3))
    factory = apply_swiglu_sharded_factory(gathered, ())
    factory.gtp_source_param = live
    with pytest.raises(ValueError, match="EP-unaware rebuild"):
        _backfill_gtp_sharded_param_map({}, [[live]], {"weight": factory})
