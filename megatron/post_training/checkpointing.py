# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from pathlib import Path
from typing import Optional

import modelopt.torch.opt as mto
import torch.nn as nn
from modelopt.torch.opt.plugins import restore_sharded_modelopt_state

from megatron.training import get_args
from megatron.training.checkpointing import (
    _load_base_checkpoint,
    checkpoint_exists,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    read_metadata,
)
from megatron.training.utils import print_rank_0

logger = logging.getLogger(__name__)


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint.
    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state exists, False otherwise
    """
    args = get_args()

    try:
        if args.ckpt_format == "torch":
            # Non-sharded
            state_dict, _, _ = _load_base_checkpoint(checkpoint_path, rank0=False)
            if state_dict is None:
                return False
            if "modelopt_state" not in state_dict:
                return False
            return True
        else:
            # Sharded
            load_dir = get_sharded_load_dir(checkpoint_path)
            if load_dir is None:
                return False
            if not (load_dir / "modelopt_state").is_dir():
                return False
            return True
    except Exception as e:
        print_rank_0(f"Failed to inspect checkpoint in {checkpoint_path}: {e}")
        return False


def get_sharded_load_dir(load_dir: str) -> Optional[Path]:
    """Helper to retrieve the sharded load directory from a MLM checkpoint tracker file."""
    if not checkpoint_exists(load_dir):
        return None

    tracker_filename = get_checkpoint_tracker_filename(load_dir)
    iteration, release = read_metadata(tracker_filename)
    sharded_load_dir = Path(get_checkpoint_name(load_dir, iteration, release, return_base_dir=True))

    if not sharded_load_dir.exists():
        return None

    return sharded_load_dir


def load_modelopt_state(model: nn.Module, load_dir: Optional[str] = None) -> None:
    """Loading modelopt_state without loading the model.

    If distributed checkpointing in use, we try to load from the sharded modelopt_state. This will not
    load the model state_dict. Otherwise, if the checkpoint is not sharded, we load the base checkpoint
    (which contains the model state as well) and extract the modelopt_state.

    Args:
        model: the model to load the modelopt_state into
        load_dir: optionally provide a different loading path
    """
    args = get_args()
    load_dir = load_dir or args.load

    if args.ckpt_format == "torch":
        # Non-sharded
        print_rank_0(f"Loading ModelOpt state from base checkpoint ({load_dir})")
        try:
            state_dict, _, _ = _load_base_checkpoint(args.load, rank0=False)
        except Exception:
            print_rank_0("Failed to load base checkpoint via megatron _load_base_checkpoint!")
            return
        if state_dict is None:
            print_rank_0("No checkpoint state_dict found. Skipping loading ModelOpt state.")
            return
        modelopt_state = state_dict.get("modelopt_state", None)
        if modelopt_state is not None:
            mto.restore_from_modelopt_state(model, modelopt_state)
    else:
        # Sharded
        sharded_load_dir = get_sharded_load_dir(load_dir)
        if sharded_load_dir is None:
            print_rank_0("No sharded checkpoint found. Skipping loading modelopt_state.")
            return
        restore_sharded_modelopt_state([model], sharded_load_dir)
