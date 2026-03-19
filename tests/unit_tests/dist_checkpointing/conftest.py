# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest

from megatron.core.dist_checkpointing.strategies.base import StrategyAction, get_default_strategy
from megatron.core.msc_utils import MultiStorageClientFeature


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="class")
def tmp_dir_per_class(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope='session', autouse=True)
def set_default_dist_ckpt_strategy():
    # Disable MSC for tests
    MultiStorageClientFeature.disable()

    # Import the real class before patching so the inner function doesn't
    # call back through the mock (which would cause infinite recursion since
    # get_default_strategy now directly instantiates TorchDistSaveShardedStrategy).
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistSaveShardedStrategy as _RealTorchDistSaveShardedStrategy,
    )

    def get_pyt_dist_save_sharded_strategy():
        return _RealTorchDistSaveShardedStrategy()

    with mock.patch(
        'megatron.core.dist_checkpointing.strategies.torch.TorchDistSaveShardedStrategy',
        new=get_pyt_dist_save_sharded_strategy,
    ) as _fixture:
        yield _fixture
