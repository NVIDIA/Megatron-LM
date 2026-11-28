# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Three-way serial/static/runtime MLA parity without duplicate static runs."""

import pytest

from megatron.core.utils import is_te_min_version
from tests.unit_tests.transformer.mla_test_utils import check_mla_parallel_correctness

pytestmark = [pytest.mark.launch_on_gb200, pytest.mark.experimental]


@pytest.mark.parametrize(
    ("rope_type", "apply_rope_fusion"),
    [
        pytest.param("rope", False, id="rope"),
        pytest.param("yarn", False, id="yarn"),
        pytest.param("yarn", True, id="yarn_fused"),
    ],
)
@pytest.mark.skipif(not is_te_min_version("1.10.0"), reason="Requires TransformerEngine >= 1.10.0")
def test_mla_runtime_parallel_correctness(
    tmp_path_dist_ckpt, rope_type, apply_rope_fusion, run_config
):
    check_mla_parallel_correctness(
        tmp_path_dist_ckpt,
        rope_type,
        apply_rope_fusion,
        tp=1,
        sp=False,
        cp=2,
        dynamic_cp=True,
        check_parameter_gradients=True,
    )
