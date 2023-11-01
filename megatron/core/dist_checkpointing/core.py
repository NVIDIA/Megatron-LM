# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

CONFIG_FNAME = 'metadata.json'


class CheckpointingException(Exception):
    pass


@dataclass
class CheckpointingConfig:
    """ Documents backends used in the checkpoint. """

    sharded_backend: str
    sharded_backend_version: int = 1
    common_backend: str = 'torch'
    common_backend_version: int = 1


def check_is_distributed_checkpoint(checkpoint_dir):
    return maybe_load_config(checkpoint_dir) is not None


def maybe_load_config(checkpoint_dir: str) -> Optional[CheckpointingConfig]:
    config_path = Path(checkpoint_dir, CONFIG_FNAME)
    if not config_path.exists():
        return None
    with config_path.open() as f:
        config_dict = json.load(f)
    return CheckpointingConfig(**config_dict)


def save_config(config: CheckpointingConfig, checkpoint_dir: str):
    config_path = Path(checkpoint_dir, CONFIG_FNAME)
    with config_path.open('w') as f:
        json.dump(asdict(config), f)
