# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

import pytest

from tools.checkpoint.dist_checkpoint_io import detect_checkpoint_format, resolve_checkpoint_subdir


@pytest.mark.parametrize("marker", ["release", "release\n", " release \n"])
def test_resolve_release_checkpoint_root(tmp_path, marker: str) -> None:
    checkpoint = tmp_path / "release"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"sharded_backend": "torch_dist"}))
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(marker)

    assert resolve_checkpoint_subdir(str(tmp_path)) == (str(checkpoint), None)
    assert detect_checkpoint_format(str(tmp_path)) == "torch_dist"


def test_resolve_numbered_checkpoint_root(tmp_path) -> None:
    checkpoint = tmp_path / "iter_0000042"
    checkpoint.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("42\n")

    assert resolve_checkpoint_subdir(str(tmp_path)) == (str(checkpoint), 42)


def test_resolve_flat_checkpoint_takes_priority(tmp_path) -> None:
    (tmp_path / "metadata.json").write_text(json.dumps({"sharded_backend": "torch_dist"}))
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")

    assert resolve_checkpoint_subdir(str(tmp_path)) == (str(tmp_path), None)


def test_missing_release_directory_preserves_unresolved_root(tmp_path) -> None:
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")

    assert resolve_checkpoint_subdir(str(tmp_path)) == (str(tmp_path), None)
