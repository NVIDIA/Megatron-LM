# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Module for managing distributed checkpoints metadata. """

import json
import os
from cloudpathlib import AnyPath
from dataclasses import asdict, dataclass
from typing import Optional

CONFIG_FNAME = 'metadata.json'


class CheckpointingException(Exception):
    """ Base checkpointing related exception  """

    pass


@dataclass
class CheckpointingConfig:
    """ Documents backends used in the checkpoint.

    Checkpoint config keeps track of formats used for storing the sharded tensors
    (sharded_backend) and other objects (common_backend).

    Note that versioning is not for the checkpoint content (which is application specific),
    but for the checkpoint format itself.
    """

    sharded_backend: str
    sharded_backend_version: int = 1
    common_backend: str = 'torch'
    common_backend_version: int = 1


def check_is_distributed_checkpoint(checkpoint_dir: str):
    """ Checks if `metadata.json` exists in the checkpoint and is a valid config.

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        bool: True if `metadata.json` exists in the checkpoint and is a valid config.
    """
    return maybe_load_config(checkpoint_dir) is not None


def maybe_load_config(checkpoint_dir: str) -> Optional[CheckpointingConfig]:
    """ Returns checkpoint config if `checkpoint_dir` is a distributed checkpoint and None otherwise

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        CheckpointingConfig (optional): None if checkpoint is not a valid distributed checkpoint
    """
    config_path = AnyPath(os.path.join(checkpoint_dir, CONFIG_FNAME))
    if not config_path.exists():
        return None
    with config_path.open() as f:
        config_dict = json.load(f)
    return CheckpointingConfig(**config_dict)


def save_config(config: CheckpointingConfig, checkpoint_dir: str):
    """ Save given config to checkpoint directory.

    Args:
        config: checkpoint config
        checkpoint_dir: checkpoint directory

    Returns:
        None
    """
    config_path = AnyPath(os.path.join(checkpoint_dir, CONFIG_FNAME))
    with config_path.open('w') as f:
        json.dump(asdict(config), f)
