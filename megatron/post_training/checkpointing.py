# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import os
from pathlib import Path
from typing import Optional

import modelopt
import modelopt.torch.opt as mto
import torch.nn as nn
from modelopt.torch.opt.plugins import (
    restore_sharded_modelopt_state as restore_sharded_modelopt_state_legacy,
)
from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
    _load_extra_state_from_sharded_checkpoint,
)

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import _legacy_common_state_exists
from megatron.core.post_training.modelopt.checkpointing import restore_sharded_modelopt_state
from megatron.core.utils import unwrap_model
from megatron.training import get_args
from megatron.training.checkpointing import (
    _load_base_checkpoint,
    checkpoint_exists,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    load_checkpoint,
    read_metadata,
)
from megatron.training.utils import print_rank_0

from .utils import print_distributed_quant_summary

logger = logging.getLogger(__name__)


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint and contains nontrivial state.

    NOTE: Ignores distillation (KD) state, which is deprecated and unused.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state exists and contains nontrivial state, False otherwise
    """
    args = get_args()

    def _has_nontrivial_modelopt_state(modelopt_state: dict) -> bool:
        """Check whether modelopt_state contains state beyond the (deprecated, unused) KD mode."""
        modes = modelopt_state.get("modelopt_state_dict", [])
        if len(modes) == 1 and modes[0][0] == "kd_loss":
            # Ignore KD state.
            modes = modes[:-1]
        return len(modes) > 0

    try:
        if args.ckpt_format == "torch":
            # Non-sharded
            state_dict, _, _ = _load_base_checkpoint(checkpoint_path, rank0=False)
            if state_dict is None:
                return False
            modelopt_state = state_dict.get("modelopt_state")
            if modelopt_state is None:
                return False
            return _has_nontrivial_modelopt_state(modelopt_state)
        else:
            # Sharded
            load_dir = get_sharded_load_dir(checkpoint_path)
            if load_dir is None:
                return False
            modelopt_checkpoint_name = load_dir / "modelopt_state"
            if not modelopt_checkpoint_name.is_dir():
                return False
            modelopt_state = dist_checkpointing.load_common_state_dict(str(modelopt_checkpoint_name))
            return _has_nontrivial_modelopt_state(modelopt_state)
    except Exception as e:
        print_rank_0(f"Failed to inspect checkpoint in {checkpoint_path}: {e}")
        return False


def get_sharded_load_dir(load_dir: str) -> Optional[Path]:
    """Helper to retrieve the sharded load directory from a MLM checkpoint tracker file."""
    if not checkpoint_exists(load_dir):
        return None

    tracker_filename = get_checkpoint_tracker_filename(load_dir)
    iteration, release = read_metadata(tracker_filename)
    sharded_load_dir = Path(get_checkpoint_name(load_dir, iteration, release, return_base_dir=True))

    if not sharded_load_dir.exists():
        return None

    return sharded_load_dir


def load_modelopt_state(model: nn.Module, load_dir: Optional[str] = None) -> None:
    """Loading modelopt_state without loading the model.

    If distributed checkpointing is in use, we try to load the sharded modelopt_state
    without loading the model state_dict. Otherwise, if the checkpoint is not
    sharded, we load the base checkpoint
    (which contains the model state as well) and extract the modelopt_state.

    Args:
        model: the model to load the modelopt_state into
        load_dir: optionally provide a different loading path
    """
    args = get_args()
    load_dir = load_dir or args.load

    if args.ckpt_format == "torch":
        # Non-sharded
        print_rank_0(f"Loading ModelOpt state from base checkpoint ({load_dir})")
        try:
            state_dict, _, _ = _load_base_checkpoint(args.load, rank0=False)
        except Exception:
            print_rank_0("Failed to load base checkpoint via megatron _load_base_checkpoint!")
            return
        if state_dict is None:
            print_rank_0("No checkpoint state_dict found. Skipping loading ModelOpt state.")
            return
        modelopt_state = state_dict.get("modelopt_state", None)
        if modelopt_state is not None:
            mto.restore_from_modelopt_state(model, modelopt_state)
    else:
        # Sharded
        sharded_load_dir = get_sharded_load_dir(load_dir)
        if sharded_load_dir is None:
            print_rank_0("No sharded checkpoint found. Skipping loading modelopt_state.")
            return
        if _legacy_common_state_exists(f"{sharded_load_dir}/modelopt_state"):
            restore_sharded_modelopt_state_legacy([model], sharded_load_dir)
        else:
            restore_sharded_modelopt_state([model], sharded_load_dir)


def load_modelopt_checkpoint(
    model,
    optimizer=None,
    opt_param_scheduler=None,
    strict: bool = True,
    additional_sharded_prefix: str = "",
    load_arg: str = "load",
) -> None:
    """Load a sharded (untar .nemo or megatron --use-dist-ckpt) or unsharded checkpoint.

    Essentially, the function is detecting whether the checkpoint is a .nemo sharded checkpoint.
    If so, we load the sharded state_dict with additional_sharded_prefix `model.`.
    This additional prefix is tha artifact of the lightning module wrapper. Once the sharded
    state_dict is loaded, we use a state_dict pre_hook to pop this additional prefix (`model.`)
    from all state_dict keys.

    If this is not a .nemo sharded checkpoint, then this function will simply call
    load_checkpoint. See megatron.checkpointing.load_checkpoint for explanation.

    Args:
        additional_sharded_prefix: append additional prefix to align the sharded checkpoint keys.
            When loading an .nemo sharded checkpoint, this is usually `model.`. Otherwise, this is
            typically an empty string.
    """

    def _remove_prefix_state_dict_pre_hook(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Pytorch state_dict pre_hook to remove prefix of the state_dict keys."""
        if additional_sharded_prefix is None:
            return
        key_rewrite_list = []
        for key, _ in state_dict.items():
            if key.startswith(additional_sharded_prefix):
                key_rewrite_list.append(key)
        for old_key in key_rewrite_list:
            new_key = old_key[len(additional_sharded_prefix) :]
            state_dict[new_key] = state_dict.pop(old_key)

    args = get_args()
    load_dir = getattr(args, load_arg)
    # main's get_sharded_load_dir() resolves the MLM tracker file only and returns just the
    # directory; the NeMo weight-dir probe that also produced a key prefix was removed, so
    # the prefix now comes solely from this function's additional_sharded_prefix argument.
    sharded_load_dir = get_sharded_load_dir(load_dir)

    unwrapped_model = unwrap_model(model)

    if args.ckpt_format == "torch":
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir, args, rank0=False
        )
        model_state_dict = state_dict["model"]
        unwrapped_model[0].load_state_dict(model_state_dict, strict=False)
        print_distributed_quant_summary(unwrapped_model[0])
    elif sharded_load_dir is not None and optimizer is None and opt_param_scheduler is None:
        sharded_state_dict_metadata = dist_checkpointing.load_content_metadata(sharded_load_dir)
        sharded_state_dict = unwrapped_model[0].sharded_state_dict(
            prefix=additional_sharded_prefix, metadata=sharded_state_dict_metadata
        )

        if additional_sharded_prefix:
            unwrapped_model[0]._register_load_state_dict_pre_hook(
                _remove_prefix_state_dict_pre_hook
            )
        model_state_dict = dist_checkpointing.load(
            sharded_state_dict, sharded_load_dir, strict=args.dist_ckpt_strictness
        )
        unwrapped_model[0].load_state_dict(model_state_dict, strict=False)
        print_distributed_quant_summary(unwrapped_model[0])
    else:
        _ = load_checkpoint(model, optimizer, opt_param_scheduler, strict=strict, load_arg=load_arg)


def load_kd_teacher_checkpoint(model) -> None:
    """Load the teacher checkpoint for ModelOpt distillation if the model has one."""
    args = get_args()
    if not getattr(args, "export_kd_teacher_load", None):
        return

    teacher = unwrap_model(model[0]).teacher_model
    print_rank_0(
        f"Loading teacher as {type(teacher).__name__} from {args.export_kd_teacher_load} ..."
    )
    # [WAR]: To avoid error out on loading teacher's checkpoint, we temporarily
    # set args.finetune to True while loading the teacher checkpoint.
    original_args_finetune, original_ckpt_format = args.finetune, args.ckpt_format
    args.finetune = True
    if args.export_kd_teacher_ckpt_format is not None:
        args.ckpt_format = args.export_kd_teacher_ckpt_format
    try:
        load_checkpoint([teacher], None, None, load_arg='export_kd_teacher_load')
    finally:
        args.finetune, args.ckpt_format = original_args_finetune, original_ckpt_format
    print_rank_0("... teacher loaded successfully.")
