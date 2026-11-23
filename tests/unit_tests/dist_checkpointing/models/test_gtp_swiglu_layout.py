# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""SwiGLU checkpoint tiles must describe contiguous GTP slices of TP-local pairs."""

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory


# GTP3 shards straddle the w/v boundary of each TP-local [w|v] pair; GTP2 and GTP4 shards align
# with it. ``dequantized`` reaches the GTP size through the ``_gtp_dequant_src`` backlink.
@pytest.mark.parametrize(
    ("tp_size", "gtp_size", "singleton", "dequantized"),
    [
        pytest.param(2, 3, False, False, id="tp2-gtp3-straddle"),
        pytest.param(2, 3, True, True, id="tp2-gtp3-straddle-singleton-dequantized"),
        pytest.param(2, 2, True, False, id="tp2-gtp2-singleton"),
        pytest.param(1, 4, False, True, id="tp1-gtp4-dequantized"),
    ],
)
def test_gtp_swiglu_checkpoint_layout(tp_size, gtp_size, singleton, dequantized):
    # Distinct values in every row expose permutations that constant weights hide.
    half_rows = 3 * gtp_size
    full = torch.arange(2 * tp_size * half_rows * 4, dtype=torch.float32).reshape(-1, 4)
    w, v = full.chunk(2)
    targets = {"fc1_w": w, "fc1_v": v} if singleton else {"fc1": full}
    restored = {key: torch.empty_like(value) for key, value in targets.items()}
    coverage = {key: torch.zeros_like(value, dtype=torch.bool) for key, value in targets.items()}
    for tp_rank in range(tp_size):
        tp_pair = torch.cat([w.chunk(tp_size)[tp_rank], v.chunk(tp_size)[tp_rank]])
        for gtp_rank, shard in enumerate(tp_pair.chunk(gtp_size)):
            shard = shard.clone()
            shard.gtp_remat_size = gtp_size
            if dequantized:
                data = shard.clone()
                data._gtp_dequant_src = shard
            else:
                data = shard
            original = ShardedTensor.from_rank_offsets(
                "fc1", data, (0, tp_rank * gtp_size + gtp_rank, tp_size * gtp_size)
            )
            factory = apply_swiglu_sharded_factory(original, (), singleton)
            pieces = factory.build()
            loaded = []
            for piece in pieces:
                begin = piece.global_offset[0]
                end = begin + piece.local_shape[0]
                assert not coverage[piece.key][begin:end].any()
                restored[piece.key][begin:end] = piece.data
                coverage[piece.key][begin:end] = True
                # Loading a canonical checkpoint must recover this physical shard.
                loaded.append(targets[piece.key][begin:end])
            assert torch.equal(factory.merge_fn(loaded), shard)
    assert all(mask.all() for mask in coverage.values())
    assert all(torch.equal(restored[key], target) for key, target in targets.items())


@pytest.mark.parametrize(("local_rows", "pad_length"), [(3, 0), (4, 2)])
def test_gtp_swiglu_rejects_unsupported_shards(local_rows, pad_length):
    data = torch.arange(local_rows * 4, dtype=torch.float32).reshape(local_rows, 4)
    data.gtp_remat_size = 2
    original = ShardedTensor.from_rank_offsets("fc1", data, (0, 0, 2))
    original.gtp_pad_length = pad_length
    factory = apply_swiglu_sharded_factory(original, ())
    with pytest.raises(AssertionError):
        factory.build()
