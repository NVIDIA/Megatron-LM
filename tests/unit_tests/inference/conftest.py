# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="package", autouse=True)
def no_gloo_process_groups():
    """Inference tests never use Gloo groups, so their model-parallel setup skips them."""
    default = Utils.create_gloo_process_groups
    Utils.create_gloo_process_groups = False
    yield
    Utils.create_gloo_process_groups = default
