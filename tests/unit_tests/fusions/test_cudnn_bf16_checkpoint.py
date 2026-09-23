# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import replace

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory


@pytest.mark.parametrize("interleave", [None, 1, 16, 32])
@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("shape", [(256,), (256, 64)])
@pytest.mark.parametrize("optimizer_state", [False, True])
def test_cudnn_glu_checkpoint_layout(interleave, singleton, shape, optimizer_state):
    """Canonical checkpoint values/offsets and inverse local layout, including FP32 states."""
    canonical = torch.arange(torch.tensor(shape).prod().item(), device="cuda").float()
    canonical = (canonical / 1003).reshape(shape)
    index = torch.arange(shape[0], device="cuda")
    if interleave is not None:
        half = shape[0] // 2
        index = torch.cat(
            [
                torch.cat(
                    (
                        index[start : start + interleave],
                        index[half + start : half + start + interleave],
                    )
                )
                for start in range(0, half, interleave)
            ]
        )
    local = canonical.bfloat16().index_select(0, index)
    offsets = () if singleton else ((0, 3, 8),)
    axis = len(offsets)
    sharded = ShardedTensor.from_rank_offsets(
        "expert.weight", local, *offsets, (axis, 1, 2), prepend_axis_num=axis
    )
    factory = apply_swiglu_sharded_factory(
        sharded, offsets, singleton, glu_interleave_size=interleave
    )
    if optimizer_state:
        # Real optimizer factories replace data/key; never capture the BF16 parameter's data.
        local = canonical.index_select(0, index)
        factory = replace(factory, data=local, key="optimizer.state.exp_avg.expert.weight")
    else:
        canonical = canonical.bfloat16()
    before = local.clone()
    pieces = factory.build()
    assert len(pieces) == 2
    for piece, expected in zip(pieces, canonical.chunk(2)):
        assert torch.equal(piece.data, expected)
        assert piece.dtype == canonical.dtype
        assert piece.local_shape == expected.shape
    assert pieces[0].global_offset[axis] == shape[0] // 2
    assert pieces[1].global_offset[axis] == (shape[0] // 2 if singleton else 3 * shape[0] // 2)
    assert pieces[0].global_shape[axis] == (shape[0] if singleton else 2 * shape[0])
    assert pieces[0].key == factory.key + ("_w" if singleton else "")
    assert pieces[1].key == factory.key + ("_v" if singleton else "")
    assert torch.equal(factory.merge_fn([piece.data.clone() for piece in pieces]), local)
    assert torch.equal(local, before)


@pytest.mark.parametrize("interleave", [0, -1, 3, True, 1.5])
def test_cudnn_glu_checkpoint_invalid_block(interleave):
    sharded = ShardedTensor.from_rank_offsets("weight", torch.empty(256, 64, device="cuda"))
    with pytest.raises(ValueError, match="interleave size"):
        apply_swiglu_sharded_factory(sharded, (), glu_interleave_size=interleave)
