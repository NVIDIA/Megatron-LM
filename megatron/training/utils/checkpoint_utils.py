# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import sys
from typing import Any, Optional

import torch

try:
    import yaml

    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from megatron.core._rank_utils import safe_get_rank, safe_get_world_size
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.training.config.utils import sanitize_dataclass_config
from megatron.training.state import TrainState
from megatron.training.utils.common_utils import print_rank_0

CONFIG_FILE = "run_config.yaml"
TRAIN_STATE_FILE = "train_state.pt"
_RUNTIME_ONLY_TARGETS = frozenset({"megatron.core.timers.Timers"})
_RECIPE_CONFIG_TARGET = "megatron.core.quantization.quant_config.RecipeConfig"

logger = logging.getLogger(__name__)


def join_paths(*paths: str) -> str:
    """Join paths, using MultiStorageClient when needed"""
    if not paths:
        raise ValueError("Empty paths")

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.os.path.join(*paths)

    return os.path.join(*paths)


def get_checkpoint_train_state_filename(checkpoints_path: str, prefix: Optional[str] = None) -> str:
    """Get the filename for the train state tracker file.

    This file typically stores metadata about the latest checkpoint, like the iteration number.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.
        prefix: Optional prefix (e.g., 'latest') to prepend to the filename.

    Returns:
        The full path to the train state tracker file.
    """
    if prefix is None:
        return join_paths(checkpoints_path, TRAIN_STATE_FILE)
    else:
        return join_paths(checkpoints_path, f"{prefix}_{TRAIN_STATE_FILE}")


def read_train_state(train_state_filename: str) -> TrainState:
    """Read the train state metadata from a .pt file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks if
    torch.distributed is initialized. Otherwise, loads the file locally.

    Args:
        train_state_filename: Path to the train state .pt file.

    Returns:
        An initialized TrainState object.
    """
    if torch.distributed.is_initialized():
        state_obj = [None]
        if safe_get_rank() == 0:
            try:
                if MultiStorageClientFeature.is_enabled():
                    msc = MultiStorageClientFeature.import_package()
                    state_dict = msc.torch.load(train_state_filename, map_location="cpu", weights_only=True)
                else:
                    state_dict = torch.load(train_state_filename, map_location="cpu", weights_only=True)
                ts = TrainState()
                ts.load_state_dict(state_dict)
                state_obj[0] = ts
            except Exception as e:
                error_msg = f"ERROR: Unable to load train state file {train_state_filename}: {e}"
                sys.stderr.write(error_msg + "\n")
                state_obj[0] = {"error": True, "msg": error_msg}

        print_rank_0(f"Broadcasting TrainState from rank 0 to all {safe_get_world_size()} ranks")
        torch.distributed.broadcast_object_list(state_obj, src=0)

        if isinstance(state_obj[0], dict) and state_obj[0].get("error", False):
            raise RuntimeError(state_obj[0]["msg"])

        return state_obj[0]

    try:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            state_dict = msc.torch.load(train_state_filename, map_location="cpu", weights_only=True)
        else:
            state_dict = torch.load(train_state_filename, map_location="cpu", weights_only=True)
        ts = TrainState()
        ts.load_state_dict(state_dict)
        return ts
    except Exception as e:
        raise RuntimeError(f"Unable to load train state file {train_state_filename}: {e}") from e


def get_checkpoint_run_config_filename(checkpoints_path: str) -> str:
    """Get the filename for the run configuration file within a checkpoint directory.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.

    Returns:
        The full path to the run configuration file (e.g., run_config.yaml).
    """
    return join_paths(checkpoints_path, CONFIG_FILE)


def read_run_config(run_config_filename: str) -> dict[str, Any]:
    """Read the run configuration from a YAML file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks.

    Args:
        run_config_filename: Path to the run config YAML file.

    Returns:
        A dictionary containing the run configuration.

    Raises:
        ImportError: If PyYAML is not installed.
        RuntimeError: If reading the config file fails on rank 0.
    """
    if not HAVE_YAML:
        raise ImportError(
            "PyYAML is required to read YAML configuration files from the checkpoint. "
            "Install via `pip install pyyaml`."
        )

    if torch.distributed.is_initialized():
        config_obj = [None]

        if get_rank_safe() == 0:
            try:
                if MultiStorageClientFeature.is_enabled():
                    msc = MultiStorageClientFeature.import_package()
                    with msc.open(run_config_filename, "r") as f:
                        config_dict = yaml.safe_load(f)
                else:
                    with open(run_config_filename, "r") as f:
                        config_dict = yaml.safe_load(f)
                config_dict = _sanitize_run_config_object(config_dict)
                config_dict = apply_run_config_backward_compat(config_dict)
                config_obj[0] = config_dict
            except Exception as e:
                error_msg = f"ERROR: Unable to load config file {run_config_filename}: {e}"
                sys.stderr.write(error_msg + "\n")
                config_obj[0] = {"error": True, "msg": error_msg}

        print_rank_0(
            f"Broadcasting config from rank 0 to all {safe_get_world_size()} ranks",
            rank=safe_get_rank(),
        )
        torch.distributed.broadcast_object_list(config_obj, src=0)

        if isinstance(config_obj[0], dict) and config_obj[0].get("error", False):
            raise RuntimeError(config_obj[0]["msg"])

        return config_obj[0]
    else:
        try:
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                with msc.open(run_config_filename, "r") as f:
                    config_dict = yaml.safe_load(f)
            else:
                with open(run_config_filename, "r") as f:
                    config_dict = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Unable to load config file {run_config_filename}: {e}") from e

        config_dict = _sanitize_run_config_object(config_dict)
        config_dict = apply_run_config_backward_compat(config_dict)
        return config_dict


def _sanitize_run_config_object(obj: Any) -> Any:
    """Remove runtime-only objects from run config dictionaries.

    Timers and other runtime constructs are serialized with `_target_` entries
    that cannot be recreated without additional context (e.g., constructor
    arguments provided at runtime). These objects are not required when loading
    a checkpoint configuration, so we replace them with ``None`` to avoid
    instantiation errors when the config is processed later.
    """

    if isinstance(obj, dict):
        target = obj.get("_target_")
        if isinstance(target, str) and target in _RUNTIME_ONLY_TARGETS:
            return None
        if (
            target == _RECIPE_CONFIG_TARGET
            and obj.get("_call_", True) is True
            and set(obj).issubset({"_target_", "_call_"})
        ):
            logger.warning(
                "Ignoring a legacy quantization recipe whose state was not preserved in run_config.yaml. "
                "The checkpoint can still be loaded, but the original per-module quantization settings "
                "must be supplied separately if they are needed."
            )
            return None
        return {key: _sanitize_run_config_object(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_run_config_object(item) for item in obj]
    return obj


def apply_run_config_backward_compat(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply backward compatibility transformations to run config.

    This function handles dataclass config fields that should not be passed to
    the constructor when loading older checkpoints. It automatically detects
    init=False fields by inspecting the target class.

    The entire config is sanitized recursively to handle init=False fields in any part of the configuration hierarchy.

    Args:
        config_dict: The full run configuration dictionary.

    Returns:
        The config dictionary with backward compatibility fixes applied.
    """
    return sanitize_dataclass_config(config_dict)
