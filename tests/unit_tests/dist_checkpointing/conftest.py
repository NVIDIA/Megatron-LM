from unittest import mock

import pytest

from megatron.core.dist_checkpointing.strategies.base import StrategyAction, get_default_strategy


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="class")
def tmp_dir_per_class(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope='session', autouse=True)
def set_default_dist_ckpt_strategy():
    def get_pyt_dist_save_sharded_strategy():
        return get_default_strategy(StrategyAction.SAVE_SHARDED, 'torch_dist', 1)

    with mock.patch(
        'megatron.core.dist_checkpointing.serialization.get_default_save_sharded_strategy',
        new=get_pyt_dist_save_sharded_strategy,
    ) as _fixture:
        yield _fixture
