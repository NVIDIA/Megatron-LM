# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

""" Common strategies. """

import logging
from pathlib import Path

import torch

from megatron.core.dist_checkpointing.mapping import StateDict

from ..mapping import CheckpointingException

COMMON_STATE_FNAME = 'common.pt'

logger = logging.getLogger(__name__)


def save_common(common_state_dict: StateDict, checkpoint_dir: Path):
    """Save common part of the state dict."""
    if torch.distributed.get_rank() == 0:
        torch.save(common_state_dict, checkpoint_dir / COMMON_STATE_FNAME)


def load_common(checkpoint_dir: Path):
    """Load common (non-sharded) objects state dict from the checkpoint.

    Args:
        checkpoint_dir (Path): checkpoint directory

    Returns:
        StateDict: state dict with non-sharded objects from the checkpoint
    """
    load_path = Path(checkpoint_dir) / COMMON_STATE_FNAME
    try:
        return torch.load(load_path, map_location='cpu', weights_only=False)
    except FileNotFoundError as e:
        err_msg = f'Common file {load_path} does not exist'
        ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
        logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
        raise CheckpointingException(err_msg) from e
