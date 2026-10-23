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
            state_dict, _, _ = _load_base_checkpoint(load_dir, rank0=False)
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


def load_kd_teacher_checkpoint(
    model, distillation: DistillationConfig | None = None, *, student_checkpoint=None
) -> None:
    """Load teacher weights after wrapping, using configuration or legacy CLI arguments."""
    args = get_args()
    if distillation is None:
        teacher_load = getattr(args, "export_kd_teacher_load", None)
        if not teacher_load:
            return
        distillation = DistillationConfig(
            teacher_load=teacher_load,
            teacher_ckpt_format=getattr(args, "export_kd_teacher_ckpt_format", None),
        )

    teacher = unwrap_model(model[0]).teacher_model
    print_rank_0(
        f"Loading teacher as {type(teacher).__name__} from {distillation.teacher_load} ..."
    )
    # load_checkpoint still uses runtime args. Adapt only for the duration of this call,
    # restoring even absent fields and failed loads so the student's state is unchanged.
    overrides = {
        "finetune": True,
        "export_kd_teacher_load": distillation.teacher_load,
        "ckpt_format": distillation.teacher_ckpt_format
        or (student_checkpoint.ckpt_format if student_checkpoint is not None else args.ckpt_format),
    }
    missing = object()
    original = {key: getattr(args, key, missing) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(args, key, value)
        load_checkpoint([teacher], None, None, load_arg="export_kd_teacher_load")
    finally:
        for key, value in original.items():
            if value is missing:
                delattr(args, key)
            else:
                setattr(args, key, value)
    print_rank_0("... teacher loaded successfully.")
