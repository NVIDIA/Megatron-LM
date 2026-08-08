# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.datasets.utils import get_blend_from_list


def test_get_blend_from_list_none():
    assert get_blend_from_list(None) is None


def test_get_blend_from_list_prefixes_only_odd_length():
    prefixes, weights = get_blend_from_list(["a", "b", "c"])
    assert prefixes == ["a", "b", "c"]
    assert weights is None


def test_get_blend_from_list_weighted():
    prefixes, weights = get_blend_from_list(["30", "a", "70", "b"])
    assert prefixes == ["a", "b"]
    assert weights == [30.0, 70.0]


def test_get_blend_from_list_prefixes_only_even_length():
    prefixes, weights = get_blend_from_list(["a", "b", "c", "d"])
    assert prefixes == ["a", "b", "c", "d"]
    assert weights is None


@pytest.mark.parametrize(
    "blend",
    [
        ["30", "a", "b", "c"],
        ["a", "b", "70", "c"],
    ],
)
def test_get_blend_from_list_mixed_weights_rejected(blend):
    with pytest.raises(AssertionError):
        get_blend_from_list(blend)
