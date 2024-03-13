from pathlib import Path
from unittest import mock

import pytest

from megatron.core.dist_checkpointing.strategies.base import StrategyAction, get_default_strategy
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """ Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes. """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir


@pytest.fixture(scope='session', autouse=True)
def set_default_dist_ckpt_strategy():
    def get_pyt_dist_strategy(action: StrategyAction, backend: str, version: int):
        if action == StrategyAction.SAVE_SHARDED and backend != 'torch_dist':
            backend = 'torch_dist'
        return get_default_strategy(action, backend, version)

    with mock.patch(
        'megatron.core.dist_checkpointing.serialization.get_default_strategy',
        new=get_pyt_dist_strategy,
    ) as _fixture:
        yield _fixture
