# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run the cheap selection/matrix contracts in the normal unit-test bucket too."""

from tests.test_utils.test_dev_unit_test_selection import (
    test_every_unit_file_has_exactly_one_platform_bucket,
)
from tests.test_utils.test_gdn_unit_test_matrix import (
    test_post_fusion_matrix_preserves_pairs_and_layout_fusion_recompute_triples,
)

__all__ = [
    "test_every_unit_file_has_exactly_one_platform_bucket",
    "test_post_fusion_matrix_preserves_pairs_and_layout_fusion_recompute_triples",
]
