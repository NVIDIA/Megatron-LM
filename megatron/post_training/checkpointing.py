# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
    get_checkpoint_tracker_filename,
    get_load_checkpoint_path,
    load_checkpoint,
    resolve_checkpoint_load_dir,
)
from megatron.training.config import CheckpointConfig
from megatron.training.global_vars import get_run_config
from megatron.training.models.distillation import DistillationConfig
from megatron.training.utils import print_rank_0


def has_modelopt_state(checkpoint_path: str, *, checkpoint: CheckpointConfig) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint and contains nontrivial state.

    NOTE: Ignores distillation (KD) state, which is deprecated and unused.

    Args:
        checkpoint_path: Path to the checkpoint directory
        checkpoint: The checkpoint policy to inspect with (its format selects the reader): the
            registered section at run time, or, at startup before any run config exists, the
            section built from the arguments.

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
        if checkpoint.ckpt_format == "torch":
            # Non-sharded; the legacy reader is the one path that needs the arguments.
            state_dict, _, _, _ = _load_base_checkpoint(
                checkpoint_path, get_args(), rank0=False, checkpoint=checkpoint
            )
            if state_dict is None:
                return False
            modelopt_state = state_dict.get("modelopt_state")
            if modelopt_state is None:
                return False
            return _has_nontrivial_modelopt_state(modelopt_state)
        else:
            # Sharded
            if checkpoint_path is None:
                return False
            load_dir = Path(get_load_checkpoint_path(checkpoint_path, checkpoint))
            modelopt_checkpoint_name = load_dir / "modelopt_state"
            if not modelopt_checkpoint_name.is_dir():
                return False
            modelopt_state = dist_checkpointing.load_common_state_dict(str(modelopt_checkpoint_name))
            return _has_nontrivial_modelopt_state(modelopt_state)
    except Exception as e:
        print_rank_0(f"Failed to inspect checkpoint in {checkpoint_path}: {e}")
        return False


def load_modelopt_state(
    model: nn.Module, load_dir: Optional[str] = None, *, checkpoint: CheckpointConfig | None = None
) -> None:
    """Loading modelopt_state without loading the model.

    If distributed checkpointing in use, we try to load from the sharded modelopt_state. This will not
    load the model state_dict. Otherwise, if the checkpoint is not sharded, we load the base checkpoint
    (which contains the model state as well) and extract the modelopt_state.

    Args:
        model: the model to load the modelopt_state into
        load_dir: optionally provide a different loading path
        checkpoint: checkpoint policy; defaults to the registered run section
    """
    checkpoint = get_run_config().checkpoint if checkpoint is None else checkpoint
    if load_dir is None:
        load_dir, _ = resolve_checkpoint_load_dir(checkpoint)

    if checkpoint.ckpt_format == "torch":
        # Non-sharded
        print_rank_0(f"Loading ModelOpt state from base checkpoint ({load_dir})")
        try:
            state_dict, _, _, _ = _load_base_checkpoint(
                load_dir, get_args(), rank0=False, checkpoint=checkpoint
            )
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
        sharded_load_dir = (
            Path(get_load_checkpoint_path(load_dir, checkpoint)) if load_dir is not None else None
        )
        if sharded_load_dir is None or not sharded_load_dir.is_dir():
            print_rank_0("No sharded checkpoint found. Skipping loading modelopt_state.")
            return
        restore_sharded_modelopt_state([model], sharded_load_dir)


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
    checkpoint = get_run_config().checkpoint if student_checkpoint is None else student_checkpoint
    teacher = unwrap_model(model[0]).teacher_model
    print_rank_0(
        f"Loading teacher as {type(teacher).__name__} from {distillation.teacher_load} ..."
    )
    # The teacher's weights come from its own tracker, regardless of the student's resume
    # selectors. Per-call overrides leave both the arguments and the registered section alone.
    load_checkpoint(
        [teacher],
        None,
        None,
        load_dir=distillation.teacher_load,
        finetune=True,
        load_optim=False,
        load_rng=False,
        ckpt_format=distillation.teacher_ckpt_format or checkpoint.ckpt_format,
        ckpt_step=None,
        pretrained_checkpoint=None,
        non_persistent_ckpt_type=None,
    )
    print_rank_0("... teacher loaded successfully.")
