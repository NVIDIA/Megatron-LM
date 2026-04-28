# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Override the parent fixture in tests/unit_tests/conftest.py.

    Tests under tests/unit_tests/ssm/ build inputs synthetically (torch.ones,
    torch.randn) and never read /opt/data, so we skip the download attempt
    that would otherwise spam the log with FileNotFoundError noise on
    machines where /opt/data is not writable.
    """
    yield
