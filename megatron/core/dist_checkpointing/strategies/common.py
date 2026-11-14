# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Common strategies."""

import logging

import torch

from megatron.core.dist_checkpointing.mapping import StateDict

from ..filesystem import create_write_stream, resolve_filesystem
from ..mapping import CheckpointingException

COMMON_STATE_FNAME = 'common.pt'

logger = logging.getLogger(__name__)


def save_common(common_state_dict: StateDict, checkpoint_dir: str, *, filesystem=None):
    """Save common part of the state dict."""
    logger.warning(
        "save_common is deprecated and will be removed in a future release. "
        "`torch_dist` now handles all non-tensor data as part of default PyTorch DCP behavior."
    )

    filesystem = resolve_filesystem(filesystem)
    if torch.distributed.get_rank() == 0:
        path = filesystem.concat_path(checkpoint_dir, COMMON_STATE_FNAME)
        with create_write_stream(filesystem, path) as stream:
            torch.save(common_state_dict, stream)


def load_common(checkpoint_dir: str, *, filesystem=None):
    """Load common (non-sharded) objects state dict from the checkpoint.

    Args:
        checkpoint_dir (Path): checkpoint directory

    Returns:
        StateDict: state dict with non-sharded objects from the checkpoint
    """
    logger.warning(
        "load_common is deprecated and will be removed in a future release. "
        "`torch_dist` now handles all non-tensor data as part of default PyTorch DCP behavior."
    )

    filesystem = resolve_filesystem(filesystem)
    load_path = filesystem.concat_path(checkpoint_dir, COMMON_STATE_FNAME)
    try:
        with filesystem.create_stream(load_path, "rb") as stream:
            return torch.load(stream, map_location='cpu')
    except FileNotFoundError as e:
        err_msg = f'Common file {load_path} does not exist'
        ckpt_files = list(filesystem.listdir(checkpoint_dir))
        logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
        raise CheckpointingException(err_msg) from e
