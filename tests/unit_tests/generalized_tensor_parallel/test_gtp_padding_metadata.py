# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for the logical layout of alignment-padded GTP checkpoint shards."""

import pytest
import torch

from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.dist_checkpointing.strategies.torch import _gtp_restore_padded
from megatron.core.tensor_parallel.gtp_utils import gtp_entry_backlink, untrimmed_gtp_shard
from megatron.core.utils import _make_gtp_logical_sharded_tensor


def _entry(tensor, *, tp_axis=0, tp_rank=0, gtp_rank=0, pad_length=2, expert=False):
    prepend_offsets = ((0, 1, 2),) if expert else ()
    prepend_axis_num = len(prepend_offsets)
    other_offsets = ((prepend_axis_num + tp_axis, tp_rank, 2),) if tp_axis else ()
    return _make_gtp_logical_sharded_tensor(
        tensor,
        "experts.weight" if expert else "weight",
        tp_axis=tp_axis,
        tp_rank=tp_rank,
        tp_size=2,
        gtp_rank=gtp_rank,
        gtp_remat_size=2,
        pad_length=pad_length,
        prepend_offsets=prepend_offsets,
        prepend_axis_num=prepend_axis_num,
        other_offsets=other_offsets,
        replica_id=(0, 0, 0),
    )


@pytest.mark.parametrize("tp_axis", [0, 1])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("gtp_rank", [0, 1])
@pytest.mark.parametrize("expert", [False, True])
def test_padded_shards_cover_the_unpadded_reference(tp_axis, tp_rank, gtp_rank, expert):
    """Every shard indexes the independently constructed, unpadded global reference."""
    logical_rows, shard_rows, columns = 6, 4, 3
    global_shape = (logical_rows * 2, columns) if tp_axis == 0 else (logical_rows, columns * 2)
    reference = torch.arange(global_shape[0] * global_shape[1]).reshape(global_shape)
    tensor = torch.full((shard_rows, columns), -1, dtype=reference.dtype)
    sharded = _entry(tensor, tp_axis=tp_axis, tp_rank=tp_rank, gtp_rank=gtp_rank, expert=expert)
    start = gtp_rank * shard_rows
    keep = min(shard_rows, logical_rows - start)
    row_offset = tp_rank * logical_rows + start if tp_axis == 0 else start
    col_offset = tp_rank * columns if tp_axis == 1 else 0
    expected = reference[row_offset : row_offset + keep, col_offset : col_offset + columns]
    prefix = (2,) if expert else ()
    assert sharded.global_shape == prefix + global_shape
    assert sharded.global_offset == ((1,) if expert else ()) + (row_offset, col_offset)
    assert sharded.local_shape == expected.shape
    sharded.data.copy_(expected)
    restored = _gtp_restore_padded(sharded)
    assert restored is tensor
    torch.testing.assert_close(restored[:keep], expected)
    assert torch.all(restored[keep:] == -1)
    assert sharded.axis_fragmentations is None
    assert not sharded.allow_shape_mismatch
    sharded.validate_metadata_integrity()


@pytest.mark.parametrize("pad_length,keep", [(2, 2), (4, 0), (6, 0)])
def test_native_fp8_identity_and_untrimmed_data_remain_separate(pad_length, keep):
    """A dequantized buffer supplies collectives and loads; optimizer matching uses the live param."""
    live = torch.nn.Parameter(torch.zeros(4, 3))
    dequantized = torch.ones(4, 3, dtype=torch.bfloat16)
    dequantized._gtp_dequant_src = live
    sharded = _entry(dequantized, gtp_rank=1, pad_length=pad_length)
    assert sharded.local_shape == (keep, 3)
    assert gtp_entry_backlink(sharded) is live
    assert untrimmed_gtp_shard(sharded) is dequantized
    sharded.data = sharded.data.detach()
    assert gtp_entry_backlink(sharded) is live
    assert _gtp_restore_padded(sharded) is dequantized
    assert get_param_id_to_sharded_param_map({"w": sharded}, [live])[0] is sharded
    from megatron.core.optimizer.distrib_optimizer import _resolve_gtp_sharded_metadata

    assert _resolve_gtp_sharded_metadata(live, {"w": sharded}) is sharded


@pytest.mark.parametrize("pad_length,keep", [(2, 2), (4, 0), (6, 0)])
def test_optimizer_padding_is_excluded_without_changing_the_unsplit_key(pad_length, keep):
    """The generic optimizer mapper preserves its physical schema and restores the full buffer."""
    live = torch.nn.Parameter(torch.zeros(4, 3))
    sharded = _entry(live, gtp_rank=1, pad_length=pad_length, expert=True)
    state = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    optim_sharded = make_sharded_optimizer_tensor(sharded, state, "optimizer.state.exp_avg")
    assert optim_sharded.key == "optimizer.state.exp_avg.experts.weight"
    assert optim_sharded.local_shape == (keep, 3)
    assert optim_sharded.global_shape == sharded.global_shape
    assert optim_sharded.global_offset == sharded.global_offset
    assert _gtp_restore_padded(optim_sharded) is state
    optim_sharded.validate_metadata_integrity()


@pytest.mark.parametrize(
    "gtp_rank,pad_length,logical_numel", [(0, 2, 12), (1, 2, 6), (1, 4, 0), (1, 6, 0)]
)
def test_optimizer_load_restores_only_the_known_padding(gtp_rank, pad_length, logical_numel):
    """A short optimizer state is legal only when its length is exactly this shard's logical size."""
    from types import SimpleNamespace

    from megatron.core.optimizer.distrib_optimizer import _restore_gtp_optimizer_padding

    live = torch.nn.Parameter(torch.zeros(4, 3))
    live.is_gtp_weight_remat = True
    live.gtp_remat_size = 2
    live.pad_length = pad_length
    live.group = SimpleNamespace(rank=lambda: gtp_rank)
    state = torch.arange(logical_numel, dtype=torch.float32)
    restored = _restore_gtp_optimizer_padding(state, live)
    torch.testing.assert_close(restored[:logical_numel], state)
    assert restored.numel() == live.numel()
    assert torch.count_nonzero(restored[logical_numel:]) == 0
    with pytest.raises(ValueError, match="Optimizer state has"):
        _restore_gtp_optimizer_padding(torch.zeros(logical_numel + 1), live)


def test_optimizer_load_rejects_unrelated_truncated_states():
    """Padding support does not turn incomplete ordinary optimizer tensors into valid states."""
    from megatron.core.optimizer.distrib_optimizer import _restore_gtp_optimizer_padding

    live = torch.nn.Parameter(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="no GTP alignment padding"):
        _restore_gtp_optimizer_padding(torch.zeros(6), live)


@pytest.mark.parametrize("pad_length", [2, 4])
@pytest.mark.parametrize("local_range", [(0, 12), (1, 11), (8, 12), (12, 12)])
def test_fs_model_space_padded_fragments(pad_length, local_range):
    """The real optimizer save method trims only logical rows and restores each flat DP slice."""
    from types import SimpleNamespace

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from tests.unit_tests.tensor_parallel.test_gtp_checkpoint_optimizer import (
        _loaded_tree,
        _optimizer_stub,
    )

    live = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    live.is_gtp_weight_remat = True
    live.gtp_remat_size = 2
    live.pad_length = pad_length
    live.group = SimpleNamespace(rank=lambda: 1)
    source = _entry(live, gtp_rank=1, pad_length=pad_length)
    result = DistributedOptimizer.sharded_param_state_fs_model_space(
        _optimizer_stub(live, local_range), {"weight": source}, metadata={}
    )
    logical_numel = source.data.numel()
    for state_key, factory in result[0].items():
        merged = factory.merge_fn(_loaded_tree(factory.build()))
        delta = {"fp32_param": 0, "exp_avg": 5000, "exp_avg_sq": 9000}[state_key]
        expected = live.detach().flatten().clone() + delta
        expected[logical_numel:] = 0
        torch.testing.assert_close(merged, expected[slice(*local_range)], rtol=0, atol=0)


def test_fs_model_space_rejects_inconsistent_padding_metadata():
    """Physical size alone cannot authorize an arbitrary short checkpoint tensor."""
    from types import SimpleNamespace

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from tests.unit_tests.tensor_parallel.test_gtp_checkpoint_optimizer import _optimizer_stub

    live = torch.nn.Parameter(torch.zeros(4, 3))
    live.is_gtp_weight_remat = True
    live.gtp_remat_size = 2
    live.pad_length = 1
    live.group = SimpleNamespace(rank=lambda: 1)
    source = _entry(live, gtp_rank=1, pad_length=2)
    with pytest.raises(ValueError, match="expected 9 logical elements"):
        DistributedOptimizer.sharded_param_state_fs_model_space(
            _optimizer_stub(live, (0, 12)), {"weight": source}, metadata={}
        )
