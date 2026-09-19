# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if torch.distributed.is_initialized():
        print("Waiting for destroy_process_group")
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="function", autouse=True)
def set_env():
    """Configure TE env vars for MoE unit tests.

    ``NVTE_CUTEDSL_FUSED_GROUPED_MLP`` enables TE's cuDSL fused grouped MLP path.
    The kernel additionally requires SM100 (Blackwell), so on H100/A100 CI this
    is a no-op; setting it here means the kernel is picked up automatically when
    Blackwell hardware joins the unit-test matrix.
    """
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'
    os.environ['NVTE_CUTEDSL_FUSED_GROUPED_MLP'] = '1'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir
# Appended to tests/unit_tests/transformer/moe/conftest.py

import os as _os

import pytest as _pytest
import torch as _torch
import torch.distributed as _dist


@_pytest.fixture(scope="session")
def nccl_session():
    """A session-scoped real NCCL process group for the A3 regression tests.

    Skipped when torchrun did not provide RANK/WORLD_SIZE. Lifecycle is deliberately NOT torn
    down here: the session-scoped ``cleanup`` fixture above is the single owner of
    ``destroy_process_group``.

    Why this exists: both A3 test modules need a real distributed group. Giving each module its
    own init/destroy pair breaks when they run in one torchrun session, because a module-scoped
    destroy races the next module's init and the session dies with
    ``ncclRemoteError: ... remote process exited prematurely`` while rank 0 is SIGTERM'd --
    which looks like a code failure but is a test-harness lifecycle bug.
    """
    if not _torch.cuda.is_available():
        _pytest.skip("CUDA not available")
    if "RANK" not in _os.environ or "WORLD_SIZE" not in _os.environ:
        _pytest.skip("requires torchrun with RANK/WORLD_SIZE")

    world = int(_os.environ["WORLD_SIZE"])
    _torch.cuda.set_device(int(_os.environ.get("LOCAL_RANK", 0)))
    if not _dist.is_initialized():
        _dist.init_process_group(
            "nccl", device_id=_torch.device("cuda", _torch.cuda.current_device())
        )
    return world
