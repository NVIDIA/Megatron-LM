# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.transformer.attention import fa4_num_splits

# A GB200 rank of a GQA model with 8 query heads per KV head (e.g. 32 heads, 2 KV groups, TP4).
SMS = 152
QH, KVH = 8, 1
MAX_SEQ = 262144


def splits(batch, max_seqlen_q, max_seqlen_k, min_splits, num_heads_q=QH, num_heads_kv=KVH):
    return fa4_num_splits(
        batch, num_heads_q, num_heads_kv, max_seqlen_q, max_seqlen_k, min_splits, num_sms=SMS
    )


def test_zero_floor_defers_to_fa4():
    # 0 = pass num_splits=0 through so FA4 applies its own heuristic.
    assert splits(128, 1, MAX_SEQ, 0) == 0
    assert splits(4, 1, MAX_SEQ, 0) == 0


def test_large_decode_batch_gets_the_floor():
    # Heuristic: 152 // (128 requests * 1 KV head * 1 m-block) = 1 split; the floor raises it.
    assert splits(128, 1, MAX_SEQ, 1) == 1
    assert splits(128, 1, MAX_SEQ, 8) == 8


def test_small_decode_batch_keeps_the_heuristic():
    # 152 // 4 = 38 splits already exceeds the floor.
    assert splits(4, 1, MAX_SEQ, 8) == 38
    # The heuristic is capped at 128 splits.
    assert splits(1, 1, MAX_SEQ, 8) == 128


def test_short_kv_is_not_split():
    # At most 4 KV tiles of 128: FA4 does not split, and neither does the floor.
    assert splits(128, 1, 4 * 128, 8) == 1


def test_floor_is_capped_at_kv_tiles():
    # 6 KV tiles: more than 6 splits would leave empty CTAs.
    assert splits(128, 1, 6 * 128, 8) == 6


def test_mixed_batch_gets_the_floor():
    # A 4096-token prefill chunk packs 4096 * 8 query rows into 128 m-blocks of 256 (two
    # 128-row stages), so the heuristic is 0 and the floor applies.
    assert splits(64, 4096, MAX_SEQ, 8) == 8


def test_speculative_decode_rows_stay_in_one_m_block():
    # 3 query tokens per request x 8 heads = 24 rows, still one m-block.
    assert splits(32, 3, MAX_SEQ, 2) == 152 // 32


@pytest.mark.parametrize("min_splits", [1, 2, 8, 16])
def test_never_below_heuristic(min_splits):
    for batch in (1, 2, 8, 19, 76, 128, 512):
        heuristic = min(SMS // batch, 128, MAX_SEQ // 128)
        assert splits(batch, 1, MAX_SEQ, min_splits) == max(heuristic, min_splits)
