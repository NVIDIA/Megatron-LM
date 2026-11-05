# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
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
from megatron.core.dist_checkpointing.dict_utils import nested_values
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
from megatron.training.models.distillation import DistillationConfig
from megatron.training.utils import print_rank_0

logger = logging.getLogger(__name__)


def has_modelopt_state(checkpoint_path: str, ckpt_format: str) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint and contains nontrivial state.

    NOTE: Ignores distillation (KD) state, which is deprecated and unused.

    Args:
        checkpoint_path: Path to the checkpoint directory
        ckpt_format: Checkpoint format used to load the state.

    Returns:
        True if modelopt_state exists and contains nontrivial state, False otherwise
    """

    def _has_nontrivial_modelopt_state(modelopt_state: dict) -> bool:
        """Check whether modelopt_state contains state beyond the (deprecated, unused) KD mode."""
        modes = modelopt_state.get("modelopt_state_dict", [])
        if len(modes) == 1 and modes[0][0] == "kd_loss":
            # Ignore KD state.
            modes = modes[:-1]
        return len(modes) > 0

    try:
        if ckpt_format == "torch":
            # Non-sharded
            state_dict, _, _, _ = _load_base_checkpoint(checkpoint_path, get_args(), rank0=False)
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
            modelopt_state = dist_checkpointing.load_common_state_dict(
                str(modelopt_checkpoint_name)
            )
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


def load_modelopt_state(
    model: nn.Module, load_dir: Optional[str] = None, *, ckpt_format: str | None = None
) -> None:
    """Loading modelopt_state without loading the model.

    If distributed checkpointing in use, we try to load from the sharded modelopt_state. This will not
    load the model state_dict. Otherwise, if the checkpoint is not sharded, we load the base checkpoint
    (which contains the model state as well) and extract the modelopt_state.

    Args:
        model: the model to load the modelopt_state into
        load_dir: optionally provide a different loading path
        ckpt_format: explicit checkpoint format; defaults to the runtime format
    """
    args = get_args()
    load_dir = load_dir or args.load
    ckpt_format = ckpt_format or args.ckpt_format

    if ckpt_format == "torch":
        # Non-sharded
        print_rank_0(f"Loading ModelOpt state from base checkpoint ({load_dir})")
        try:
            state_dict, _, _, _ = _load_base_checkpoint(load_dir, args, rank0=False)
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
        restore_sharded_modelopt_state([model], sharded_load_dir)


def resolve_checkpoint_load_dir(checkpoint) -> Optional[str]:
    """The directory ``load_checkpoint`` reads for ``checkpoint``.

    That is ``checkpoint.load``, or ``checkpoint.pretrained_checkpoint`` when ``load`` holds no
    checkpoint, as when finetuning with ``--pretrained-checkpoint`` and an empty ``--load``.
    """
    if checkpoint is None:
        return None
    pretrained_dir = getattr(checkpoint, "pretrained_checkpoint", None)
    if pretrained_dir is not None and not checkpoint_exists(checkpoint.load):
        return pretrained_dir
    return checkpoint.load


def _layernorm_key_remaps(model: nn.Module) -> set[tuple[str, str]]:
    """The ``(module-name fragment, checkpoint-key fragment)`` pairs ``model``'s layers remap.

    Read from the layer specs' ``sharded_state_dict_keys_map`` (e.g. ``input_layernorm.`` ->
    ``self_attention.linear_qkv.layer_norm_``), which the ModelOpt-local spec uses to load and
    save layernorms under the fused Transformer-Engine keys.
    """
    remaps = set()
    for module in model.modules():
        submodules = getattr(module, "submodules_config", None)
        keys_map = getattr(submodules, "sharded_state_dict_keys_map", None) or {}
        remaps.update(keys_map.items())
    return remaps


# Where load_legacy_layernorm_keys keeps a layer's original submodules_config.
_REMAPPED_SUBMODULES_ATTR = "_modelopt_remapped_submodules_config"


def load_legacy_layernorm_keys(model: nn.Module, load_dir: str, ckpt_format: str) -> bool:
    """Prepare ``model`` to load a distributed checkpoint that stores layernorms by module name.

    The ModelOpt-local spec loads and saves layernorms under the fused Transformer-Engine keys
    (e.g. ``self_attention.linear_qkv.layer_norm_weight``). A checkpoint saved without that
    remapping (``--export-te-mcore-model`` off) has ``input_layernorm.weight`` etc. instead,
    for the same tensors. If ``load_dir`` is such a checkpoint, turn off the remappings it
    lacks on ``model``'s layers so it loads as is, until ``restore_layernorm_key_remaps``;
    later saves then use the fused keys, converting the checkpoint. Remappings the checkpoint
    does use, such as a stock MLA spec's, are kept.

    Returns:
        Whether the checkpoint stores layernorms under their module names.
    """
    if ckpt_format != "torch_dist":
        return False
    remaps = _layernorm_key_remaps(model)
    if not remaps:
        return False
    sharded_load_dir = get_sharded_load_dir(load_dir)
    if sharded_load_dir is None:
        return False
    try:
        checkpoint_keys = set(dist_checkpointing.load_tensors_metadata(str(sharded_load_dir)))
        model_keys = {
            getattr(value, "key", None) for value in nested_values(model.sharded_state_dict())
        }
    except Exception as e:  # The check is advisory; the load itself reports real errors.
        logger.warning(f"Skipping the layernorm checkpoint-key check for {sharded_load_dir}: {e}")
        return False
    # The remappings whose keys the checkpoint has under the module names instead.
    legacy = {
        (unfused, fused)
        for key in model_keys
        if key is not None and key not in checkpoint_keys
        for unfused, fused in remaps
        if fused in key and key.replace(fused, unfused) in checkpoint_keys
    }
    if not legacy:
        return False
    print_rank_0(
        f"Checkpoint {sharded_load_dir} stores layernorm weights under their module names; "
        f"loading them as such. Checkpoints saved from this run use the fused "
        f"Transformer-Engine keys."
    )
    for module in model.modules():
        submodules = getattr(module, "submodules_config", None)
        keys_map = getattr(submodules, "sharded_state_dict_keys_map", None) or {}
        kept = {
            unfused: fused for unfused, fused in keys_map.items() if (unfused, fused) not in legacy
        }
        if len(kept) < len(keys_map):
            # Replace rather than modify: layers built from one spec share its submodules.
            setattr(module, _REMAPPED_SUBMODULES_ATTR, submodules)
            module.submodules_config = dataclasses.replace(
                submodules, sharded_state_dict_keys_map=kept
            )
    return True


def restore_layernorm_key_remaps(model) -> None:
    """Undo ``load_legacy_layernorm_keys`` once the checkpoint is loaded. No-op otherwise."""
    for stage in model if isinstance(model, list) else [model]:
        for module in stage.modules():
            original = module.__dict__.pop(_REMAPPED_SUBMODULES_ATTR, None)
            if original is not None:
                module.submodules_config = original


def load_kd_teacher_checkpoint(
    model, distillation: DistillationConfig | None = None, *, student_checkpoint=None
) -> None:
    """Load the teacher's weights after wrapping. No-op when the run has no distillation."""
    if distillation is None:
        return
    # load_checkpoint only warns about a missing checkpoint, which would leave the teacher
    # randomly initialized.
    if not checkpoint_exists(distillation.teacher_load):
        raise FileNotFoundError(
            f"No teacher checkpoint found at {distillation.teacher_load} (expected "
            f"{get_checkpoint_tracker_filename(distillation.teacher_load)})."
        )
    args = get_args()

    teacher = unwrap_model(model[0]).teacher_model
    ckpt_format = distillation.teacher_ckpt_format or (
        student_checkpoint.ckpt_format if student_checkpoint is not None else args.ckpt_format
    )
    transformations = getattr(distillation.teacher_model, "transformations", None)
    if transformations is not None and not transformations.default_te_spec:
        load_legacy_layernorm_keys(teacher, distillation.teacher_load, ckpt_format)
    print_rank_0(
        f"Loading teacher as {type(teacher).__name__} from {distillation.teacher_load} ..."
    )
    # load_checkpoint still uses runtime args. Adapt only for the duration of this call,
    # restoring even absent fields and failed loads so the student's state is unchanged.
    # Also clear the student's resume/finetune selectors, which would otherwise redirect the
    # teacher load to another iteration or directory.
    overrides = {
        "finetune": True,
        "export_kd_teacher_load": distillation.teacher_load,
        "ckpt_format": ckpt_format,
        "pretrained_checkpoint": None,
        "ckpt_step": None,
        "non_persistent_ckpt_type": None,
    }
    missing = object()
    original = {key: getattr(args, key, missing) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(args, key, value)
        load_checkpoint([teacher], None, None, load_arg="export_kd_teacher_load")
    finally:
        restore_layernorm_key_remaps(teacher)
        for key, value in original.items():
            if value is missing:
                delattr(args, key)
            else:
                setattr(args, key, value)
    print_rank_0("... teacher loaded successfully.")
