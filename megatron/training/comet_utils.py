# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path
from typing import Any, Optional

from megatron.training.utils.common_utils import print_rank_last


def on_save_checkpoint_success(
    checkpoint_path: str,
    save_dir: str,
    iteration: int,
    comet_logger: Optional[Any],
) -> None:
    """Callback executed after a checkpoint is successfully saved.

    If a Comet ML experiment is provided, records the checkpoint path and iteration
    as experiment metadata.

    Args:
        checkpoint_path: The path to the specific checkpoint file/directory saved.
        save_dir: The base directory where checkpoints are being saved.
        iteration: The training iteration at which the checkpoint was saved.
        comet_logger: The Comet ML Experiment instance.
                      If None, this function is a no-op.
    """
    if comet_logger is None:
        return

    try:
        resolved_ckpt = str(Path(checkpoint_path).resolve())
        comet_logger.log_other("last_saved_checkpoint", resolved_ckpt)
        comet_logger.log_other("last_saved_iteration", iteration)
    except Exception as exc:
        print_rank_last(f"Failed to log checkpoint information to Comet ML: {exc}")


def on_load_checkpoint_success(
    checkpoint_path: str,
    load_dir: str,
    comet_logger: Optional[Any],
) -> None:
    """Callback executed after a checkpoint is successfully loaded.

    For Comet ML, records the loaded checkpoint path and base directory
    as experiment metadata.

    Args:
        checkpoint_path: The path to the specific checkpoint file/directory loaded.
        load_dir: The base directory from which the checkpoint was loaded.
        comet_logger: The Comet ML Experiment instance.
                      If None, this function is a no-op.
    """
    if comet_logger is None:
        return

    try:
        resolved_ckpt = str(Path(checkpoint_path).resolve())
        resolved_load_dir = str(Path(load_dir).resolve())
        comet_logger.log_other("last_loaded_checkpoint", resolved_ckpt)
        comet_logger.log_other("checkpoint_base_dir", resolved_load_dir)
    except Exception as exc:
        print_rank_last(f"Failed to record loaded checkpoint information to Comet ML: {exc}")
