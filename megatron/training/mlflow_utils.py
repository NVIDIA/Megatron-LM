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

import re
import sys
from pathlib import Path
from typing import Any, Optional

from megatron.training.utils.common_utils import print_rank_last


def on_save_checkpoint_success(
    checkpoint_path: str,
    save_dir: str,
    iteration: int,
    mlflow_logger: Optional[Any],
) -> None:
    """Callback executed after a checkpoint is successfully saved.

    If an MLFlow logger is provided, logs the checkpoint directory as an MLFlow
    artifact under a structured artifact path that includes the iteration number.

    Args:
        checkpoint_path: The path to the specific checkpoint file/directory saved.
        save_dir: The base directory where checkpoints are being saved.
        iteration: The training iteration at which the checkpoint was saved.
        mlflow_logger: The MLFlow module (e.g., ``mlflow``) with an active run.
                       If None, this function is a no-op.
    """
    from megatron.training.checkpoint import get_checkpoint_name

    if mlflow_logger is None:
        return

    try:
        checkpoint_path = str(Path(checkpoint_path).resolve())
        base_name = Path(save_dir).name or "checkpoints"
        expected_ckpt_path = get_checkpoint_name(save_dir, iteration)
        artifact_subdir = f"{base_name}/{Path(expected_ckpt_path).name}"
        mlflow_logger.log_artifacts(checkpoint_path, artifact_path=artifact_subdir)
    except Exception as exc:
        # continue training
        print_rank_last(f"Failed to log checkpoint artifacts to MLFlow: {exc}")


def on_load_checkpoint_success(
    checkpoint_path: str,
    load_dir: str,
    mlflow_logger: Optional[Any],
) -> None:
    """Callback executed after a checkpoint is successfully loaded.

    For MLFlow, this emits a simple metric and tag to document which checkpoint
    was loaded during the run. It does not perform artifact lookups.

    Args:
        checkpoint_path: The path to the specific checkpoint file/directory loaded.
        load_dir: The base directory from which the checkpoint was loaded.
        mlflow_logger: The MLFlow module (e.g., ``mlflow``) with an active run.
                       If None, this function is a no-op.
    """
    if mlflow_logger is None:
        return

    try:
        resolved_ckpt = str(Path(checkpoint_path).resolve())
        resolved_load_dir = str(Path(load_dir).resolve())
        mlflow_logger.set_tags(
            {
                "last_loaded_checkpoint": resolved_ckpt,
                "checkpoint_base_dir": resolved_load_dir,
            }
        )
    except Exception as exc:
        print_rank_last(f"Failed to record loaded checkpoint information to MLFlow: {exc}")


def _sanitize_mlflow_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Sanitize all metric names in a dictionary for MLFlow logging."""

    def _sanitize_key(key):
        sanitized = key.replace("@", "_at_")
        sanitized = re.sub(r"/+", "/", sanitized)
        if "/" in key:
            first, rest = sanitized.split("/", 1)
            sanitized = first + "/" + rest.replace("/", "_")
        sanitized = re.sub(r"[^/\w.\- :]", "_", sanitized)
        return sanitized

    return {_sanitize_key(key): value for key, value in metrics.items()}


def end_active_mlflow_run(status: str) -> None:
    """End the active MLFlow run with the given status.

    Used by the SIGTERM exit path (``status="KILLED"``) and the failure
    excepthook (``status="FAILED"``) to override MLFlow's default
    ``FINISHED`` status so the UI distinguishes interrupted and crashed
    runs from successful ones. Clean exits rely on MLFlow's own atexit
    handler, which already ends the run as ``FINISHED``.

    No-op if MLFlow is not installed or no run is active. Exceptions
    raised inside ``mlflow.end_run`` are caught and logged.

    Args:
        status: An MLFlow ``RunStatus`` string, typically ``"KILLED"`` or
                ``"FAILED"``.
    """
    try:
        import mlflow
    except ImportError:
        return

    if mlflow.active_run() is None:
        return

    try:
        mlflow.end_run(status=status)
    except Exception as exc:
        print_rank_last(f"Failed to end MLFlow run with status={status}: {exc}")


def install_mlflow_failure_hook() -> None:
    """Mark the active MLFlow run as ``FAILED`` on uncaught Python exceptions.

    MLFlow's own atexit handler ends the run with the default status
    ``FINISHED`` on process exit, making a crashed run indistinguishable
    from a clean one in the UI. We chain a ``sys.excepthook`` that fires
    before atexit and explicitly sets ``FAILED`` first; the previous
    excepthook is preserved so default traceback printing still happens.

    Idempotent: a second call after a previous install is a no-op.
    """
    prev_excepthook = sys.excepthook

    # Idempotent: avoid wrapping our own hook in chains across repeated installs.
    if getattr(prev_excepthook, "_mlflow_failure_hook", False):
        return

    def hook(exc_type: type[BaseException], exc_val: BaseException, exc_tb: Any) -> None:
        end_active_mlflow_run("FAILED")
        prev_excepthook(exc_type, exc_val, exc_tb)

    hook._mlflow_failure_hook = True  # type: ignore[attr-defined]
    sys.excepthook = hook
