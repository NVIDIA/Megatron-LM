# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed GDN output-fusion cases with a separate bounded CI allocation."""

from itertools import combinations, product

import pytest

from tests.unit_tests.ssm.gdn_test_matrix import POST_FUSION_DISTRIBUTED_CASES
from tests.unit_tests.ssm.test_gdn_gated_output_norm_fusion import (
    _check_post_fusion_distributed_layout,
)
from tests.unit_tests.ssm.test_gdn_gated_output_norm_fusion import (
    synchronous_inductor_compilation as synchronous_inductor_compilation,
)


def test_bounded_output_fusion_matrix_preserves_pair_and_layout_triple_coverage():
    """Cover the declared combinations without claiming to execute the full 96-case product."""
    topologies = [(1, 2, False), (1, 4, False), (2, 2, True), (4, 1, True)]
    full = [
        (*topology, batch, packed, pre, recompute, heads)
        for topology, (batch, packed), pre, recompute, heads in product(
            topologies, [(1, False), (2, False), (1, True)], [False, True], [False, True], [16, 64]
        )
    ]

    def pairs(rows):
        factors = [(row[:3], *row[3:]) for row in rows]
        return {(i, j, row[i], row[j]) for row in factors for i, j in combinations(range(6), 2)}

    selected = POST_FUSION_DISTRIBUTED_CASES
    assert len(selected) == len(set(selected)) == 15
    assert pairs(selected) == pairs(full)
    assert {row[4:7] for row in selected} == {row[4:7] for row in full}
    assert (1, 2, False, 1, True, True, False, 16) in selected
    assert (1, 4, False, 1, True, True, True, 16) in selected


@pytest.mark.parametrize(
    "tp,cp,sequence_parallel,batch,packed,pre_fusion,recompute,value_heads",
    POST_FUSION_DISTRIBUTED_CASES,
)
def test_post_fusion_distributed_layout(
    tp, cp, sequence_parallel, batch, packed, pre_fusion, recompute, value_heads
):
    """Enable the post fusion through CP redistribution and the complete backward."""
    _check_post_fusion_distributed_layout(
        tp=tp,
        cp=cp,
        sequence_parallel=sequence_parallel,
        batch=batch,
        packed=packed,
        pre_fusion=pre_fusion,
        recompute=recompute,
        value_heads=value_heads,
    )
