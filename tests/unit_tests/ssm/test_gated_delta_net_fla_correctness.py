# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN fla parallel correctness in its own CI bucket."""

import pytest

from tests.unit_tests.ssm.gated_delta_net_test_cases import (
    _test_parallel_gated_delta_net_correctness,
)


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 2),  # CP
        (2, False, 2),  # TP w/o SP + CP
        (2, True, 2),  # TP w/ SP + CP
    ],
)
@pytest.mark.parametrize("gdn_kernel_backend", ["fla"])
@pytest.mark.parametrize("head_dim", [32, 128])
def test_parallel_gated_delta_net_correctness(
    tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, gdn_kernel_backend, head_dim, run_config
):
    _test_parallel_gated_delta_net_correctness(
        tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, gdn_kernel_backend, head_dim, run_config
    )
