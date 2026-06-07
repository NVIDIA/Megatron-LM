# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

""" Common strategies. """

import logging
import os
from pathlib import Path

import torch

from megatron.core.dist_checkpointing.mapping import StateDict
from megatron.core.msc_utils import MultiStorageClientFeature

from ..mapping import CheckpointingException

COMMON_STATE_FNAME = 'common.pt'

logger = logging.getLogger(__name__)


def save_common(common_state_dict: StateDict, checkpoint_dir: str):
    """Save common part of the state dict."""
    if torch.distributed.get_rank() == 0:
        path = os.path.join(checkpoint_dir, COMMON_STATE_FNAME)
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            msc.torch.save(common_state_dict, path)
        else:
            torch.save(common_state_dict, path)


def load_common(checkpoint_dir: str):
    """Load common (non-sharded) objects state dict from the checkpoint.

    Args:
        checkpoint_dir (Path): checkpoint directory

    Returns:
        StateDict: state dict with non-sharded objects from the checkpoint
    """
    load_path = os.path.join(checkpoint_dir, COMMON_STATE_FNAME)
    try:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            return msc.torch.load(load_path, map_location='cpu')
        else:
            return torch.load(load_path, map_location='cpu')
    except FileNotFoundError as e:
        err_msg = f'Common file {load_path} does not exist'
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            ckpt_files = [f.name for f in msc.Path(checkpoint_dir).iterdir()]
        else:
            ckpt_files = [f.name for f in Path(checkpoint_dir).iterdir()]
        logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
        raise CheckpointingException(err_msg) from e
