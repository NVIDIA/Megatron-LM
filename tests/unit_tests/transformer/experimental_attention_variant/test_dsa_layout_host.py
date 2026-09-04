# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent CPU goldens for packed-CP host layouts and their request cache."""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions_from_host,
    build_packed_allgather_cp_query_positions_and_key_reorder_from_host,
)

_CPU = torch.device("cpu")


def _assert_i64(actual: torch.Tensor, expected: list[int]) -> None:
    assert actual.dtype == torch.int64
    assert actual.device == _CPU
    assert actual.tolist() == expected


@pytest.mark.parametrize(
    ("host_cu_seqlens", "cp_size", "cp_rank", "output_size", "cover_output", "expected"),
    [
        pytest.param(
            [0, 3, 7], 1, 0, None, False, [0, 1, 2, 3, 4, 5, 6], id="cp1-multisequence-identity"
        ),
        pytest.param([0, 3, 7], 1, 0, 7, True, [0, 1, 2, 3, 4, 5, 6], id="cp1-covered-output"),
        pytest.param(
            [0, 8, 8, 16],
            2,
            0,
            8,
            True,
            [0, 1, 6, 7, 8, 9, 14, 15],
            id="cp2-rank0-multisequence-zero-entry-covered",
        ),
        pytest.param(
            [0, 8, 8, 16],
            2,
            1,
            8,
            False,
            [2, 3, 4, 5, 10, 11, 12, 13],
            id="cp2-rank1-multisequence-zero-entry",
        ),
        pytest.param([0, 16], 4, 0, 4, True, [0, 1, 14, 15], id="cp4-rank0"),
        pytest.param([0, 16], 4, 1, 4, True, [2, 3, 12, 13], id="cp4-rank1"),
        pytest.param([0, 16], 4, 2, 4, True, [4, 5, 10, 11], id="cp4-rank2"),
        pytest.param([0, 16], 4, 3, 4, True, [6, 7, 8, 9], id="cp4-rank3"),
        pytest.param([0, 0, 0], 2, 0, None, True, [], id="all-empty"),
        pytest.param([0, 0, 0], 4, 3, 2, False, [6, 7], id="all-empty-with-padding"),
        pytest.param([0, 8], 2, 0, 6, False, [0, 1, 6, 7, 8, 9], id="cp2-rank0-padding"),
        pytest.param([0, 8], 2, 1, 6, False, [2, 3, 4, 5, 14, 15], id="cp2-rank1-padding"),
    ],
)
def test_host_local_positions_match_hand_written_golden(
    host_cu_seqlens, cp_size, cp_rank, output_size, cover_output, expected
):
    actual = build_packed_allgather_cp_local_positions_from_host(
        host_cu_seqlens,
        cp_size,
        cp_rank,
        _CPU,
        output_size=output_size,
        cu_seqlens_cover_output=cover_output,
    )

    _assert_i64(actual, expected)


@pytest.mark.parametrize(
    (
        "host_cu_seqlens_q",
        "host_cu_seqlens_kv",
        "cp_size",
        "cp_rank",
        "local_output_size",
        "key_local_output_size",
        "query_cover_output",
        "key_cover_output",
        "expected_query_positions",
        "expected_key_reorder",
    ),
    [
        pytest.param(
            [0, 3, 7],
            [0, 2, 7],
            1,
            0,
            7,
            7,
            True,
            False,
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            id="cp1-different-q-k-boundaries",
        ),
        pytest.param(
            [0, 8, 16],
            [0, 16],
            2,
            1,
            8,
            8,
            False,
            True,
            [2, 3, 4, 5, 10, 11, 12, 13],
            [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
            id="cp2-different-q-k-boundaries",
        ),
        pytest.param(
            [0, 8],
            [0, 8],
            2,
            0,
            6,
            6,
            False,
            False,
            [0, 1, 6, 7, 8, 9],
            [0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 10, 11],
            id="cp2-padded-query-and-key",
        ),
        pytest.param(
            [0, 16],
            [0, 16],
            4,
            2,
            4,
            4,
            True,
            True,
            [4, 5, 10, 11],
            [0, 1, 4, 5, 8, 9, 12, 13, 14, 15, 10, 11, 6, 7, 2, 3],
            id="cp4-covered-query-and-key",
        ),
    ],
)
def test_host_query_positions_and_key_reorder_match_hand_written_golden(
    host_cu_seqlens_q,
    host_cu_seqlens_kv,
    cp_size,
    cp_rank,
    local_output_size,
    key_local_output_size,
    query_cover_output,
    key_cover_output,
    expected_query_positions,
    expected_key_reorder,
):
    query_positions, key_reorder = (
        build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
            host_cu_seqlens_q,
            host_cu_seqlens_kv,
            cp_size,
            cp_rank,
            _CPU,
            local_output_size=local_output_size,
            key_local_output_size=key_local_output_size,
            global_output_size=cp_size * key_local_output_size,
            query_cu_seqlens_cover_output=query_cover_output,
            key_cu_seqlens_cover_output=key_cover_output,
        )
    )

    _assert_i64(query_positions, expected_query_positions)
    _assert_i64(key_reorder, expected_key_reorder)


class _LayerLikeOwner:
    """Owner exposing the production cache helpers."""

    _LAYOUT_HOLDER_ATTR = DSAttention._LAYOUT_HOLDER_ATTR
    _get_packed_cp_layout_cache = DSAttention._get_packed_cp_layout_cache
    _memoized = staticmethod(DSAttention._memoized)


def test_layout_cache_hits_across_two_layer_like_owners():
    packed_seq_params = PackedSeqParams(qkv_format="thd")
    first_layer = _LayerLikeOwner()
    second_layer = _LayerLikeOwner()

    first_cache = first_layer._get_packed_cp_layout_cache(packed_seq_params)
    second_cache = second_layer._get_packed_cp_layout_cache(packed_seq_params)
    assert first_cache is second_cache
    assert first_cache is getattr(packed_seq_params, DSAttention._LAYOUT_HOLDER_ATTR)

    builds = []
    expected = object()

    def build_once():
        builds.append("built")
        return expected

    first = first_layer._memoized(first_cache, ("positions_and_reorder", 2, 0), build_once)
    second = second_layer._memoized(
        second_cache,
        ("positions_and_reorder", 2, 0),
        lambda: pytest.fail("the second layer should reuse the microbatch cache"),
    )

    assert first is expected
    assert second is expected
    assert builds == ["built"]


def test_layout_cache_isolated_between_packed_seq_params():
    owner = _LayerLikeOwner()
    first_params = PackedSeqParams(qkv_format="thd")
    second_params = PackedSeqParams(qkv_format="thd")
    first_cache = owner._get_packed_cp_layout_cache(first_params)
    second_cache = owner._get_packed_cp_layout_cache(second_params)

    assert first_cache is not second_cache
    builds = []

    def build_for_microbatch(name):
        def build():
            value = object()
            builds.append((name, value))
            return value

        return build

    first_value = owner._memoized(first_cache, ("local_positions", 2, 1), build_for_microbatch("a"))
    second_value = owner._memoized(
        second_cache, ("local_positions", 2, 1), build_for_microbatch("b")
    )

    assert first_value is builds[0][1]
    assert second_value is builds[1][1]
    assert first_value is not second_value
    assert [name for name, _ in builds] == ["a", "b"]
