# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path
from typing import Optional

from megatron.core.msc_utils import MultiStorageClientFeature


TRAIN_STATE_FILE = "train_state.pt"


def join_paths(*paths: str) -> str:
    """Join paths, using MultiStorageClient when needed"""
    if not paths:
        raise ValueError("Empty paths")

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        path_cls = msc.Path
    else:
        path_cls = Path

    path = path_cls(paths[0])
    for part in paths[1:]:
        path = path / part

    return str(path)


def get_checkpoint_train_state_filename(checkpoints_path: str, prefix: Optional[str] = None) -> str:
    """Get the filename for the train state tracker file.

    This file typically stores metadata about the latest checkpoint, like the iteration number.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.
        prefix: Optional prefix (e.g., 'latest') to prepend to the filename.

    Returns:
        The full path to the train state tracker file.
    """
    if prefix is None:
        return join_paths(checkpoints_path, TRAIN_STATE_FILE)
    else:
        return join_paths(checkpoints_path, f"{prefix}_{TRAIN_STATE_FILE}")
