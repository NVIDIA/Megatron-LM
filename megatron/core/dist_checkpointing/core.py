# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Module for managing distributed checkpoints metadata. """

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

from megatron.core.msc_utils import MultiStorageClientFeature

CONFIG_FNAME = 'metadata.json'


class CheckpointingException(Exception):
    """Base checkpointing related exception"""

    pass


@dataclass
class CheckpointingConfig:
    """Documents backends used in the checkpoint.

    Checkpoint config keeps track of formats used for storing the sharded tensors
    (sharded_backend) and other objects (common_backend).

    Note that versioning is not for the checkpoint content (which is application specific),
    but for the checkpoint format itself.
    """

    sharded_backend: str
    sharded_backend_version: int = 1
    common_backend: str = 'torch'
    common_backend_version: int = 1


def check_is_distributed_checkpoint(checkpoint_dir):
    """Checks if `metadata.json` exists in the checkpoint and is a valid config.

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        bool: True if `metadata.json` exists in the checkpoint and is a valid config.
    """
    return maybe_load_config(checkpoint_dir) is not None


def maybe_load_config(checkpoint_dir: str) -> Optional[CheckpointingConfig]:
    """Returns checkpoint config if `checkpoint_dir` is a distributed checkpoint and None otherwise

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        CheckpointingConfig (optional): None if checkpoint is not a valid distributed checkpoint
    """
    config_path = os.path.join(checkpoint_dir, CONFIG_FNAME)
    if checkpoint_dir:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            if not msc.os.path.exists(config_path):
                return None
            with msc.open(config_path) as f:
                config_dict = json.load(f)
        else:
            if not os.path.exists(config_path):
                return None
            with open(config_path) as f:
                config_dict = json.load(f)
        return CheckpointingConfig(**config_dict)
    return None


def save_config(config: CheckpointingConfig, checkpoint_dir: str):
    """Save given config to checkpoint directory.

    Args:
        config: checkpoint config
        checkpoint_dir: checkpoint directory

    Returns:
        None
    """
    config_path = os.path.join(checkpoint_dir, CONFIG_FNAME)
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        with msc.open(config_path, 'w') as f:
            json.dump(asdict(config), f)
    else:
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f)
