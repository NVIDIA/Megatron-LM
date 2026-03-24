# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest

from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="class")
def tmp_dir_per_class(tmp_path_factory):
    return tmp_path_factory.mktemp("data")
