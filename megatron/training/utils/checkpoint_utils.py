# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import sys
from typing import Optional

import torch

from megatron.core._rank_utils import safe_get_rank, safe_get_world_size
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.training.state import TrainState
from megatron.training.utils import print_rank_0


TRAIN_STATE_FILE = "train_state.pt"


def join_paths(*paths: str) -> str:
    """Join paths, using MultiStorageClient when needed"""
    if not paths:
        raise ValueError("Empty paths")

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.os.path.join(*paths)

    return os.path.join(*paths)


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


def read_train_state(train_state_filename: str) -> TrainState:
    """Read the train state metadata from a .pt file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks if
    torch.distributed is initialized. Otherwise, loads the file locally.

    Args:
        train_state_filename: Path to the train state .pt file.

    Returns:
        An initialized TrainState object.
    """
    if torch.distributed.is_initialized():
        state_obj = [None]
        if safe_get_rank() == 0:
            try:
                if MultiStorageClientFeature.is_enabled():
                    msc = MultiStorageClientFeature.import_package()
                    state_dict = msc.torch.load(train_state_filename, map_location="cpu", weights_only=True)
                else:
                    state_dict = torch.load(train_state_filename, map_location="cpu", weights_only=True)
                ts = TrainState()
                ts.load_state_dict(state_dict)
                state_obj[0] = ts
            except Exception as e:
                error_msg = f"ERROR: Unable to load train state file {train_state_filename}: {e}"
                sys.stderr.write(error_msg + "\n")
                state_obj[0] = {"error": True, "msg": error_msg}

        print_rank_0(f"Broadcasting TrainState from rank 0 to all {safe_get_world_size()} ranks")
        torch.distributed.broadcast_object_list(state_obj, src=0)

        if isinstance(state_obj[0], dict) and state_obj[0].get("error", False):
            raise RuntimeError(state_obj[0]["msg"])

        return state_obj[0]

    try:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            state_dict = msc.torch.load(train_state_filename, map_location="cpu", weights_only=True)
        else:
            state_dict = torch.load(train_state_filename, map_location="cpu", weights_only=True)
        ts = TrainState()
        ts.load_state_dict(state_dict)
        return ts
    except Exception as e:
        raise RuntimeError(f"Unable to load train state file {train_state_filename}: {e}") from e
