import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
import torch.distributed
from _pytest.fixtures import FixtureRequest, fixture
from _pytest.tmpdir import TempPathFactory

from tests.unit_tests.dist_checkpointing import empty_dir, TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _mk_tmp_nonnumbered(request: FixtureRequest, factory: TempPathFactory) -> Path:
    name = request.node.name
    print('name', name, flush=True)
    name = re.sub(r"[\W]", "_", name)
    MAXVAL = 30
    name = name[:MAXVAL]
    return factory.mktemp(name)


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
