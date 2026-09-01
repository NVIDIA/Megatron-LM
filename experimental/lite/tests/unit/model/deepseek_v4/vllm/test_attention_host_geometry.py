from __future__ import annotations

import pytest

from megatron.lite.model.deepseek_v4.vllm.primitive.attention.host_geometry import (
    compressed_sequence_boundaries,
    padded_sequence_boundaries,
)


@pytest.mark.parametrize(
    ("seq_lens", "cp_size", "expected"),
    [
        ((3, 5, 9), 1, (0, 3, 8, 17)),
        ((3, 5, 9), 2, (0, 4, 12, 24)),
        ((3, 5, 9), 4, (0, 8, 16, 32)),
    ],
)
def test_padded_geometry_is_ragged_and_cp_aware(
    seq_lens: tuple[int, ...], cp_size: int, expected: tuple[int, ...]
) -> None:
    assert padded_sequence_boundaries(seq_lens, cp_size=cp_size) == expected


@pytest.mark.parametrize(
    ("boundaries", "ratio", "expected"),
    [
        ((0, 8, 24, 32), 4, (0, 2, 6, 8)),
        ((0, 128, 384), 128, (0, 1, 3)),
        ((0, 7, 16), 4, (0, 1, 3)),
    ],
)
def test_compressed_geometry_floors_each_request(
    boundaries: tuple[int, ...], ratio: int, expected: tuple[int, ...]
) -> None:
    assert compressed_sequence_boundaries(boundaries, ratio=ratio) == expected


def test_invalid_host_geometry_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        padded_sequence_boundaries((4, 0), cp_size=2)
    with pytest.raises(ValueError, match="strictly increasing"):
        compressed_sequence_boundaries((0, 8, 8), ratio=4)
