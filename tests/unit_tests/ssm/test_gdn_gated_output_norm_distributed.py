# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed GDN output-fusion cases with a separate bounded CI allocation."""

import pytest

from tests.unit_tests.ssm.gdn_test_matrix import POST_FUSION_DISTRIBUTED_CASES
from tests.unit_tests.ssm.test_gdn_gated_output_norm_fusion import (
    _check_post_fusion_distributed_layout,
)
from tests.unit_tests.ssm.test_gdn_gated_output_norm_fusion import (
    synchronous_inductor_compilation as synchronous_inductor_compilation,
)


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
