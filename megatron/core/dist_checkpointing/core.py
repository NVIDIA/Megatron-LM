# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Module for managing distributed checkpoints metadata."""

import dataclasses
import json
from dataclasses import asdict, dataclass
from typing import Optional

from .filesystem import create_write_stream, resolve_filesystem

CONFIG_FNAME = 'metadata.json'


class CheckpointingException(Exception):
    """Base checkpointing related exception"""

    pass


@dataclass
class CheckpointingConfig:
    """Documents backends used in the checkpoint.

    Checkpoint config keeps track of formats used for storing the sharded tensors
    (sharded_backend).

    Note that versioning is not for the checkpoint content (which is application specific),
    but for the checkpoint format itself.
    """

    sharded_backend: str
    sharded_backend_version: int = 1


def check_is_distributed_checkpoint(checkpoint_dir, *, filesystem=None):
    """Checks if `metadata.json` exists in the checkpoint and is a valid config.

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        bool: True if `metadata.json` exists in the checkpoint and is a valid config.
    """
    return maybe_load_config(checkpoint_dir, filesystem=filesystem) is not None


def maybe_load_config(checkpoint_dir: str, *, filesystem=None) -> Optional[CheckpointingConfig]:
    """Returns checkpoint config if `checkpoint_dir` is a distributed checkpoint and None otherwise

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        CheckpointingConfig (optional): None if checkpoint is not a valid distributed checkpoint
    """
    filesystem = resolve_filesystem(filesystem)
    if checkpoint_dir:
        config_path = filesystem.concat_path(checkpoint_dir, CONFIG_FNAME)
        if not filesystem.exists(config_path):
            return None
        with filesystem.create_stream(config_path, "rb") as f:
            config_dict = json.loads(f.read().decode("utf-8"))
        known_fields = {f.name for f in dataclasses.fields(CheckpointingConfig)}
        return CheckpointingConfig(**{k: v for k, v in config_dict.items() if k in known_fields})
    return None


def save_config(config: CheckpointingConfig, checkpoint_dir: str, *, filesystem=None):
    """Save given config to checkpoint directory.

    Args:
        config: checkpoint config
        checkpoint_dir: checkpoint directory

    Returns:
        None
    """
    filesystem = resolve_filesystem(filesystem)
    config_path = filesystem.concat_path(checkpoint_dir, CONFIG_FNAME)
    with create_write_stream(filesystem, config_path) as f:
        f.write(json.dumps(asdict(config)).encode("utf-8"))
