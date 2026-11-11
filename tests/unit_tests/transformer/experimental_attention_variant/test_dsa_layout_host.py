# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent CPU goldens for packed-CP host layouts and their request cache."""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions_from_host,
    build_packed_allgather_cp_query_positions_and_key_reorder,
    build_packed_allgather_cp_query_positions_and_key_reorder_from_host,
    invert_permutation,
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


def test_invert_permutation_matches_hand_written_golden():
    _assert_i64(invert_permutation(torch.tensor([2, 0, 3, 1])), [1, 3, 0, 2])
    _assert_i64(invert_permutation(torch.arange(5)), [0, 1, 2, 3, 4])
    _assert_i64(invert_permutation(torch.empty(0, dtype=torch.int64)), [])


@pytest.mark.parametrize("num_rows", [1, 2, 17, 4096])
def test_invert_permutation_round_trips_random_permutations(num_rows):
    perm = torch.randperm(num_rows, generator=torch.Generator().manual_seed(num_rows))
    inv = invert_permutation(perm)

    _assert_i64(inv, torch.argsort(perm).tolist())
    assert torch.equal(perm[inv], torch.arange(num_rows))
    assert torch.equal(inv[perm], torch.arange(num_rows))


@pytest.mark.parametrize(
    ("perm", "match"),
    [
        pytest.param([0, 0, 2], "repeated entries", id="repeated"),
        pytest.param([1, 1, 1], "repeated entries", id="constant"),
        pytest.param([0, 3, 1], "values in", id="above-range"),
        pytest.param([-1, 0, 1], "values in", id="negative"),
    ],
)
def test_invert_permutation_rejects_non_bijections_on_host(perm, match):
    with pytest.raises(ValueError, match=match):
        invert_permutation(torch.tensor(perm, dtype=torch.int64))


@pytest.mark.parametrize(
    "perm",
    [
        pytest.param(torch.tensor([1, 0], dtype=torch.int32), id="int32"),
        pytest.param(torch.tensor([[1, 0]]), id="two-dimensional"),
    ],
)
def test_invert_permutation_rejects_non_index_tensors(perm):
    with pytest.raises(ValueError, match="1-D int64"):
        invert_permutation(perm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_invert_permutation_device_scatter_matches_host():
    perm = torch.randperm(4099, generator=torch.Generator().manual_seed(0))
    inv = invert_permutation(perm.cuda())

    assert inv.is_cuda
    _assert_i64(inv.cpu(), invert_permutation(perm).tolist())


@pytest.mark.parametrize(
    (
        "host_cu_seqlens_q",
        "host_cu_seqlens_kv",
        "cp_size",
        "cp_rank",
        "key_local_output_size",
        "query_cover_output",
        "key_cover_output",
    ),
    [
        pytest.param([0, 16], [0, 16], 2, 1, 8, True, True, id="cp2-one-sequence-covered"),
        pytest.param([0, 8, 8, 16], [0, 8, 8, 16], 2, 1, 8, False, False, id="cp2-zero-entry"),
        pytest.param([0, 8], [0, 8], 2, 0, 6, False, False, id="cp2-padded"),
        pytest.param([0, 8, 16], [0, 16], 2, 0, 8, False, True, id="cp2-different-q-k"),
        pytest.param(
            [0, 16, 48, 64], [0, 16, 48, 64], 4, 3, 20, False, False, id="cp4-multisequence-padded"
        ),
        pytest.param(
            [0, 64, 128, 256], [0, 64, 128, 256], 8, 5, 32, False, False, id="cp8-multisequence"
        ),
    ],
)
def test_key_reorder_inverse_matches_between_host_and_device_builders(
    host_cu_seqlens_q,
    host_cu_seqlens_kv,
    cp_size,
    cp_rank,
    key_local_output_size,
    query_cover_output,
    key_cover_output,
):
    kwargs = dict(
        local_output_size=key_local_output_size,
        key_local_output_size=key_local_output_size,
        global_output_size=cp_size * key_local_output_size,
        query_cu_seqlens_cover_output=query_cover_output,
        key_cu_seqlens_cover_output=key_cover_output,
    )
    query_positions, key_reorder = (
        build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
            host_cu_seqlens_q, host_cu_seqlens_kv, cp_size, cp_rank, _CPU, **kwargs
        )
    )
    host = build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
        host_cu_seqlens_q,
        host_cu_seqlens_kv,
        cp_size,
        cp_rank,
        _CPU,
        return_inverse_key_reorder=True,
        **kwargs,
    )
    device = build_packed_allgather_cp_query_positions_and_key_reorder(
        torch.tensor(host_cu_seqlens_q, dtype=torch.int32),
        torch.tensor(host_cu_seqlens_kv, dtype=torch.int32),
        cp_size,
        cp_rank,
        _CPU,
        return_inverse_key_reorder=True,
        **kwargs,
    )

    assert len(host) == 3 and len(device) == 3
    # Requesting the inverse leaves the existing outputs untouched.
    _assert_i64(host[0], query_positions.tolist())
    _assert_i64(host[1], key_reorder.tolist())
    _assert_i64(host[2], torch.argsort(key_reorder).tolist())
    for host_tensor, device_tensor in zip(host, device):
        _assert_i64(device_tensor, host_tensor.tolist())


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
        ),
    ],
)
def test_host_key_reorder_inverse_is_validated_on_host_before_copy(monkeypatch, device):
    events = []
    move_host_tensor_to_device = dsa_layout.move_host_tensor_to_device

    def _spy_invert_permutation(perm):
        events.append(("invert", perm.device, perm.tolist()))
        return invert_permutation(perm)

    def _spy_move_host_tensor_to_device(values, *args, **kwargs):
        events.append(("copy", values.device, values.tolist()))
        return move_host_tensor_to_device(values, *args, **kwargs)

    monkeypatch.setattr(dsa_layout, "invert_permutation", _spy_invert_permutation)
    monkeypatch.setattr(dsa_layout, "move_host_tensor_to_device", _spy_move_host_tensor_to_device)
    _, key_reorder, key_reorder_inv = (
        build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
            [0, 16],
            [0, 16],
            4,
            2,
            torch.device(device),
            local_output_size=4,
            key_local_output_size=4,
            global_output_size=16,
            query_cu_seqlens_cover_output=True,
            key_cu_seqlens_cover_output=True,
            return_inverse_key_reorder=True,
        )
    )

    expected_key_reorder = [0, 1, 4, 5, 8, 9, 12, 13, 14, 15, 10, 11, 6, 7, 2, 3]
    # The inverse of the sort is the gathered order itself: each rank's local positions.
    expected_key_reorder_inv = [0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9]
    # One host-side inversion, which is also the bijection check, whatever the destination,
    # issued before either table of the key-reorder pair is copied.
    assert [event[:2] for event in events if event[0] == "invert"] == [("invert", _CPU)]
    assert ("copy", _CPU, expected_key_reorder) in events
    assert ("copy", _CPU, expected_key_reorder_inv) in events
    invert_at = events.index(("invert", _CPU, expected_key_reorder))
    assert invert_at < events.index(("copy", _CPU, expected_key_reorder))
    assert invert_at < events.index(("copy", _CPU, expected_key_reorder_inv))
    assert key_reorder.device.type == device
    assert key_reorder_inv.device.type == device
    _assert_i64(key_reorder.cpu(), expected_key_reorder)
    _assert_i64(key_reorder_inv.cpu(), expected_key_reorder_inv)


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
