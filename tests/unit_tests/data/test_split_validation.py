# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.datasets.blended_megatron_dataset_config import parse_and_normalize_split


@pytest.mark.parametrize(
    "split, expected",
    [
        ("90,5,5", [0.9, 0.05, 0.05]),
        (" 90, 5, 5 ", [0.9, 0.05, 0.05]),
        ("1", [1.0, 0.0, 0.0]),
        ("3,1", [0.75, 0.25, 0.0]),
        ("0,1,0", [0.0, 1.0, 0.0]),
        ("0.5,0.25,0.25", [0.5, 0.25, 0.25]),
    ],
)
def test_valid_split_proportions(split, expected):
    """Preserve normalization, whitespace handling, and omitted trailing splits."""
    assert parse_and_normalize_split(split) == pytest.approx(expected)


@pytest.mark.parametrize(
    "split",
    [
        "-1,1,0",
        "90,abc5,5",
        "",
        "1,,1",
        "1,1,",
        "1,1,1,1",
        "0,0,0",
        "nan,1,0",
        "inf,1,0",
        "-inf,1,0",
        "1e309,1,0",
        "1e308,1e308,0",
    ],
)
def test_invalid_split_proportions(split):
    """Reject invalid ratios instead of silently changing the data split."""
    with pytest.raises(ValueError):
        parse_and_normalize_split(split)
