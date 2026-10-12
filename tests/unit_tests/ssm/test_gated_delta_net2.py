# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN2 module correctness in a separate CI bucket from GDN."""

import pytest

from megatron.core.ssm.gated_delta_net import HAVE_FLA
from tests.unit_tests.ssm.gated_delta_net_test_cases import GatedDeltaNetModuleTestCases


@pytest.mark.parametrize("use_gdn2", [True], ids=["gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetModuleTestCases):
    """GDN2 module correctness across the original parallel layouts."""
