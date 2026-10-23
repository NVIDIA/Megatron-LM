# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import weakref

import pytest

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("lifecycle", ["normal_exit", "exception_exit", "garbage_collection"])
def test_temp_named_dir_cleanup_owner(tmp_path, monkeypatch, rank, lifecycle):
    """Only the owning rank can remove a shared directory, even after context exit."""
    monkeypatch.setattr(Utils, "rank", rank)
    directory = tmp_path / "checkpoint"
    temp_dir = TempNamedDir(directory, sync=False)
    checkpoint = directory / "common.pt"
    checkpoint.write_bytes(b"checkpoint contents")

    if lifecycle == "normal_exit":
        with temp_dir:
            pass
    elif lifecycle == "exception_exit":
        with pytest.raises(RuntimeError, match="simulated failure"):
            with temp_dir:
                raise RuntimeError("simulated failure")

    reference = weakref.ref(temp_dir)
    del temp_dir
    gc.collect()

    assert reference() is None
    if rank == 0:
        assert not directory.exists()
    else:
        assert checkpoint.read_bytes() == b"checkpoint contents"
