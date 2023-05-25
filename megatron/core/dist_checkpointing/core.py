# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from dataclasses import dataclass, asdict
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
