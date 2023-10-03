from pathlib import Path

import pytest

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
