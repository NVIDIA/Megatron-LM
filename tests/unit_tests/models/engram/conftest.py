# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.unit_tests.test_utilities import release_process_groups


@pytest.fixture(autouse=True)
def _release_process_groups():
    """The cases here look rows up and reduce gradients over process groups of their own
    (tensor, context and expert parallel groups, the row shards of the tables); see
    release_process_groups."""
    with release_process_groups():
        yield
