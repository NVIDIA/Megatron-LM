# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Module for managing distributed checkpoints metadata. """

from pathlib import Path


class CheckpointingException(Exception):
    """Base checkpointing related exception"""

    pass


def check_is_distributed_checkpoint(checkpoint_dir):
    """Checks if the checkpoint directory contains distributed checkpoint files.

    Args:
        checkpoint_dir: Path to the checkpoint directory to check

    Returns:
        bool: True if the directory contains .distcp files indicating it is a distributed checkpoint
    """

    checkpoint_path = Path(checkpoint_dir)
    distcp_files = list(checkpoint_path.glob('*.distcp'))
    return len(distcp_files) > 0
