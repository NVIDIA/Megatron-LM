# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Allow only this suite's local dummy classes in serialization tests."""

import pytest

from megatron.training.config.instantiate_utils import target_allowlist


@pytest.fixture(autouse=True)
def allow_test_config_targets():
    prefix = "tests.unit_tests.bridge."
    added = prefix not in target_allowlist.allowed_prefixes
    if added:
        target_allowlist.add_prefix(prefix)
    yield
    if added:
        target_allowlist.remove_prefix(prefix)
