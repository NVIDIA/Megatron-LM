# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Dist checkpointing modules needed for ModelOpt."""

import copy
import logging
import os
from pathlib import Path
from typing import Any

import torch

from megatron.core import mpu
from megatron.core.dist_checkpointing.serialization import load, load_common_state_dict, save
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.safe_globals import safe_load_from_bytes

logger = logging.getLogger(__name__)


def remove_per_module_state(modelopt_state: dict[str, Any]) -> None:
    """Remove metadata from the modelopt_state.

    The metadata of the modelopt_state contains keys which may change with different pipeline
    and expert parallelism. As a result, the metadata must be stored as several ShardedObject with
    global and local layer offset mapping.

    Args:
        modelopt_state: the state_dict that contains all algorithms that have been applied
            to the given model.
    """
    if "modelopt_state_dict" not in modelopt_state:
        return

    for mode, config in modelopt_state["modelopt_state_dict"]:
        metadata = config.get("metadata", None)
        if metadata is not None:
            _ = metadata.pop("quantizer_state", None)
            _ = metadata.pop("subnet_config", None)
            _ = metadata.pop("real_quantizer_state", None)
            _ = metadata.pop("q_tensor_state", None)
        else:
            config["metadata"] = {}


def save_modelopt_state(model: list[torch.nn.Module], state_dict: dict[str, Any]) -> None:
    """Save modelopt_state as a part of the per rank state_dict.

    NOTE: Only used for Megatron-LM.

    Args:
        model: the modelopt optimized model
        state_dict: the current modelopt optimized model state_dict to store
    """
    import modelopt.torch.opt as mto

    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) == 1:
        state_dict["modelopt_state"] = mto.modelopt_state(model[0])
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict[f"modelopt_state_{i}"] = mto.modelopt_state(model[i])


def save_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    sharded_strategy: tuple[str, int] | None = None,
    prefix: str = "",
) -> None:
    """Save modelopt_state in the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        sharded_strategy: configures sharded tensors saving behavior and backend
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
    """
    import modelopt.torch.opt as mto
    import modelopt.torch.utils.distributed as dist

    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")
    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    if dist.is_master():
        os.makedirs(modelopt_checkpoint_name, exist_ok=True)
    modelopt_state = copy.deepcopy(mto.modelopt_state(model[0]))
    remove_per_module_state(modelopt_state)
    save(modelopt_state, modelopt_checkpoint_name, sharded_strategy)


def _load_extra_state_from_sharded_checkpoint(
    model: torch.nn.Module,
    checkpoint_name: str | Path,
    prefix: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Load extra state from sharded checkpoint.

    Note: since extra_state is a subset of full the sharded_state_dict, we use
        strict=StrictHandling.LOG_UNEXPECTED instead of LOG_ALL.

    Args:
        model: the model to load extra state into
        checkpoint_name: the checkpoint folder path
        prefix: the prefix to add to the modelopt_state keys
        metadata: the metadata for distributed checkpointing

    Note:
        The metadata includes several breaking changes. For example, `singleton_local_shards`
        is set to `True` (was not set before) in megatron-core-0.15.0. This flag affects the
        sharded state_dict format and must be consistent between saving and loading.
    """
    sharded_state_dict = model.sharded_state_dict(prefix=prefix)
    extra_sharded_state_dict = {k: v for k, v in sharded_state_dict.items() if "_extra_state" in k}
    extra_state_dict = load(
        extra_sharded_state_dict,
        checkpoint_name,
        TorchDistLoadShardedStrategy(),
        strict=StrictHandling.LOG_UNEXPECTED,
    )
    extra_state_dict_no_prefix = {}

    for k, v in extra_state_dict.items():
        if k.startswith(prefix):
            extra_state_dict_no_prefix[k[len(prefix) :]] = v
    model.load_state_dict(extra_state_dict_no_prefix, strict=False)


def restore_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    prefix: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Restore modelopt_state from the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
        metadata: the metadata for distributed checkpointing

    Note:
        The metadata includes several breaking changes. For example, `singleton_local_shards`
        is set to `True` (was not set before) in megatron-core-0.15.0. This flag affects the
        sharded state_dict format and must be consistent between saving and loading.
    """
    import modelopt
    import modelopt.torch.opt as mto

    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")

    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"

    # Early return if the model already has a modelopt_state or the checkpoint does not exist.
    if not os.path.exists(modelopt_checkpoint_name) or mto.ModeloptStateManager.is_converted(
        model[0]
    ):
        return

    # Loading the common modelopt_state (replicated on all ranks).
    # Detect format: legacy checkpoints store common state in a standalone common.pt file;
    # newer sharded checkpoints store it as a ShardedObject inside the torch_dist checkpoint.
    legacy_common_path = os.path.join(modelopt_checkpoint_name, "common.py")
    if os.path.exists(legacy_common_path):
        common_modelopt_state = safe_load_from_bytes(legacy_common_path)
    else:
        common_modelopt_state = load_common_state_dict(modelopt_checkpoint_name)

    modelopt_load_version = common_modelopt_state["modelopt_version"]

    logger.info(
        f"nvidia-modelopt ckpt/inst version: {modelopt_load_version}/{modelopt.__version__}"
    )

    model[0] = mto.restore_from_modelopt_state(model[0], common_modelopt_state)

    _load_extra_state_from_sharded_checkpoint(model[0], checkpoint_name, prefix, metadata=metadata)
