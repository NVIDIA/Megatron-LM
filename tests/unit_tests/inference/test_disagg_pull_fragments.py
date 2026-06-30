# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unified one-sided pull: the fragment-descriptor generator must coalesce a
full-head fragment (the same-TP case) to one descriptor per (k, layer) -- the
same whole-block stride math the old index-based begin_pull produced -- and
split per token only for a partial head range (a TP remap)."""

from megatron.core.inference.disaggregation.kv_transfer_pull import _kv_fragment_triples


def _kv_dims(L, TB, BS, H, HD, elem=2):
    return {"num_layers": L, "total_blocks": TB, "block_size": BS,
            "heads": H, "hidden": HD, "elem": elem}


def test_full_head_fragment_coalesces_to_whole_block_descriptors():
    """A full-head fragment must coalesce to one descriptor per (k, layer),
    byte-identical to the whole-entry stride math: addr = (o*total_blocks +
    block)*inner with inner = block_size*heads*hidden*elem."""
    L, TB, BS, H, HD, elem = 3, 16, 4, 5, 8, 2
    dims = _kv_dims(L, TB, BS, H, HD, elem)
    src_block, dst_block = 7, 2
    triples = _kv_fragment_triples(
        dims, dims, src_block, dst_block,
        range(0, L), range(0, L), range(0, H), range(0, H),
    )
    assert len(triples) == 2 * L  # 2 (k) * L, NOT 2*L*BS
    inner = BS * H * HD * elem
    expected = [
        ((o * TB + dst_block) * inner, (o * TB + src_block) * inner, inner)
        for o in range(2 * L)  # o == k*L + layer, flattened outer index
    ]
    assert triples == expected


def test_partial_head_fragment_splits_per_token():
    """A partial head range (a TP remap) is contiguous only per token, so it
    must emit one descriptor per (k, layer, token)."""
    L, TB, BS, H, HD, elem = 2, 16, 4, 8, 8, 2
    dims = _kv_dims(L, TB, BS, H, HD, elem)
    # Pull heads [0:4) of an 8-head buffer -> partial -> per-token split.
    triples = _kv_fragment_triples(
        dims, dims, 7, 2, range(0, L), range(0, L), range(0, 4), range(0, 4),
    )
    assert len(triples) == 2 * L * BS
    assert all(nb == 4 * HD * elem for _, _, nb in triples)
