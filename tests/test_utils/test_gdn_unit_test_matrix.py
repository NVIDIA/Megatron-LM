# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Protect the interaction coverage of the bounded GDN integration matrix."""

import importlib.util
from itertools import combinations, product
from pathlib import Path

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gdn_test_matrix", ROOT / "tests/unit_tests/ssm/gdn_test_matrix.py"
)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)
POST_FUSION_DISTRIBUTED_CASES = matrix.POST_FUSION_DISTRIBUTED_CASES


def test_post_fusion_matrix_preserves_pairs_and_layout_fusion_recompute_triples():
    axes = (
        ((1, 2, False), (1, 4, False), (2, 2, True), (4, 1, True)),
        ((1, False), (2, False), (1, True)),
        (False, True),
        (False, True),
        (16, 64),
    )
    rows = [(case[:3], case[3:5], *case[5:]) for case in POST_FUSION_DISTRIBUTED_CASES]
    assert len(rows) == len(set(rows)) <= 16
    assert set(rows) <= set(product(*axes))
    for left, right in combinations(range(len(axes)), 2):
        assert {(row[left], row[right]) for row in rows} == set(product(axes[left], axes[right]))
    assert {(row[1], row[2], row[3]) for row in rows} == set(product(*axes[1:4]))
    assert (1, 2, False, 1, True, True, False, 16) in POST_FUSION_DISTRIBUTED_CASES
