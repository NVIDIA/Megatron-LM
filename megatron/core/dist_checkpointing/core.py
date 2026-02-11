# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Module for managing distributed checkpoints metadata. """

import os

from megatron.core.msc_utils import MultiStorageClientFeature


CONFIG_FNAME = "metadata.json"
COMMON_FNAME = "common.pt"


class CheckpointingException(Exception):
    """Base checkpointing related exception"""

    pass


def check_is_distributed_checkpoint(checkpoint_dir: str) -> bool:
    """Checks if the checkpoint directory contains .metadata file.

    Args:
        checkpoint_dir: Path to the checkpoint directory to check

    Returns:
        bool: True if .metadata file exists, indicating a distributed checkpoint.
    """
    metadata_file = os.path.join(checkpoint_dir, CONFIG_FNAME)
    common_file = os.path.join(checkpoint_dir, COMMON_FNAME)
    if checkpoint_dir:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            return msc.os.path.exists(metadata_file) and msc.os.path.exists(common_file)
        else:
            return os.path.exists(metadata_file) and os.path.exists(common_file)
    return False
