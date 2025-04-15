# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core import dist_checkpointing
from megatron.training import get_args
from megatron.training.checkpointing import _load_base_checkpoint, load_checkpoint
from megatron.training.utils import print_rank_0, unwrap_model

try:
    import modelopt
    from modelopt.torch.opt.plugins import (
        get_sharded_modelopt_state,
        restore_modelopt_state_metadata,
    )
    from modelopt.torch.opt.plugins.mcore_dist_checkpointing import _get_gpt_sharded_modelopt_state
except ImportError as e:
    raise ImportError("Required `\"nvidia-modelopt[torch]\"` is not installed!") from e


NEMO_WEIGHT_DIR_NAMES = {
    "model_weights": "model.", 
    "weights": "module.",
}


def get_sharded_load_dir(load_dir: str) -> Tuple[str, str]:
    """
    """
    sharded_prefix = ""
    sharded_load_dir = None
    # Read the tracker file and set the iteration if this is a MLM sharded checkpoint.
    tracker_filename = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    # If no tracker file, assuming that it is a NeMo sharded checkpoint.
    if os.path.isfile(tracker_filename):
        with open(tracker_filename, 'r') as f:
            metastring = f.read().strip()
            try:
                iteration = int(metastring)
                sharded_load_dir = Path(load_dir) / 'iter_{:07d}'.format(iteration)
            except ValueError:
                sharded_load_dir = Path(load_dir) / metastring
    else:
        for nemo_dir_name, prefix in NEMO_WEIGHT_DIR_NAMES.items():
            nemo_weight_dir = Path(load_dir) / nemo_dir_name
            if os.path.isdir(nemo_weight_dir):
                sharded_prefix = prefix
                sharded_load_dir = nemo_weight_dir
                break

    if sharded_load_dir is None:
        raise ValueError("{} is not a MLM or NeMo sharded checkpoint!".format(load_dir))

    return sharded_load_dir, sharded_prefix


def load_modelopt_state(load_dir: Optional[str] = None, model: Optional[nn.Module] = None) -> Dict:
    """Loading modelopt_state without loading the model.

    If --use-dist-ckpt, we try to load from the sharded modelopt_state. This will not load the model
    state_dict. Otherwise, if the checkpoint is not sharded, we load the base checkpoint (that
    contains the model state as well) and extract the modelopt_state.

    Args:
        load_dir: optionally provide a different loading path
        model: required when loading a sharded checkpoint
    """
    args = get_args()

    if load_dir is None:
        load_dir = args.load

    if args.use_dist_ckpt:
        assert model is not None, "`model` argument required when `args.use_dist_ckpt is True`"

        sharded_load_dir, _ = get_sharded_load_dir(load_dir)
        modelopt_state_dir = sharded_load_dir / "modelopt_state"
        if modelopt_state_dir.exists():
            common_modelopt_state = torch.load(modelopt_state_dir / "common.pt")
            extra_kwargs = {}
            for mode, mode_cfg in common_modelopt_state["modelopt_state_dict"]:
                if mode == "medusa":
                    extra_kwargs.update({"num_medusa_heads": mode_cfg["config"]["medusa_num_heads"]})
                if mode == "eagle" and modelopt.__version__ >= "0.20.0":
                    print("eagle_mode", mode_cfg["config"])
                    extra_kwargs.update({"num_eagle_layers": mode_cfg["config"]["eagle_num_layers"]})
            print_rank_0("Loading sharded modelopt_state ({})".format(modelopt_state_dir))
            modelopt_state = restore_modelopt_state_metadata(
                dist_checkpointing.load(
                    _get_gpt_sharded_modelopt_state(
                        num_layers=args.num_layers, **extra_kwargs
                    ),
                    modelopt_state_dir,
                )
            )
            return modelopt_state
        else:
            print_rank_0(
                "sharded modelopt_state ({}) does not exist!".format(modelopt_state_dir)
            )
            return {}
    else:
        print_rank_0("Loading modelopt_state from base checkpoint ({})".format(load_dir))
        try:
            state_dict, _, _ = _load_base_checkpoint(args.load, rank0=False)
        except Exception:
            print_rank_0("Failed to load base checkpoint via megatron _load_base_checkpoint!")
            return {}
        if state_dict is None:
            return {}
        return state_dict.get("modelopt_state", {})


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
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
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
    sharded_load_dir, additional_sharded_prefix = get_sharded_load_dir(load_dir)

    unwrapped_model = unwrap_model(model)

    if args.ckpt_format == "torch":
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir, args, rank0=False,
        )
        model_state_dict = state_dict["model"]
        unwrapped_model[0].load_state_dict(model_state_dict, strict=False)
    elif sharded_load_dir.exists() and optimizer is None and opt_param_scheduler is None:
        sharded_state_dict = unwrapped_model[0].sharded_state_dict(prefix=additional_sharded_prefix)
        if additional_sharded_prefix:
            unwrapped_model[0]._register_load_state_dict_pre_hook(
                _remove_prefix_state_dict_pre_hook
            )
        model_state_dict = dist_checkpointing.load(sharded_state_dict, sharded_load_dir)
        unwrapped_model[0].load_state_dict(model_state_dict, strict=False)
    else:
        _ = load_checkpoint(model, optimizer, opt_param_scheduler, strict=strict, load_arg=load_arg)
