# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""ModelOpt pre-wrap hooks for ``ModelConfig.pre_wrap_hooks``.

Each ``*_hook`` factory returns a ``list[MegatronModule] -> list[MegatronModule]`` hook that
runs after the stock GPT/Hybrid builder has built the model stages and before they are
wrapped (Float16Module / DDP). They are registered, in this order, by
``megatron.post_training.model_config.modelopt_model_config_from_args``:

1. ``restore_modelopt_state_hook``: must come before any hook that converts the model.
2. ``qad_freeze_hook``
3. ``lsq_hook``
4. ``te_load_state_dict_hook``
5. ``kd_convert_hook``
"""

import dataclasses
import functools
import logging
import os
from argparse import Namespace
from typing import Callable

import modelopt.torch.distill as mtd
import modelopt.torch.distill.plugins.megatron as mtd_mcore
import torch
import yaml

from megatron.core.post_training.modelopt.gpt.state_dict_hooks import (
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.post_training.checkpointing import load_modelopt_state
from megatron.post_training.optimizer import install_lsq_optimizer_overrides
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.models.distillation import DistillationConfig
from megatron.training.models.hybrid import HybridModelConfig

logger = logging.getLogger(__name__)

PreWrapHook = Callable[[list[MegatronModule]], list[MegatronModule]]

# Student/teacher settings that must agree: the teacher stage runs inside the student's
# pipeline schedule and exchanges tensors with it.
_SHARED_STUDENT_TEACHER_ATTRS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "pipeline_dtype",
)


def on_model_init_device(hook: PreWrapHook, init_model_with_meta_device: bool) -> PreWrapHook:
    """Run ``hook`` under the device context the model stages were built in.

    With ``--init-model-with-meta-device`` the stages are built under ``torch.device("meta")``
    and materialized later. Tensors the hooks create (modelopt_state quantizers, the teacher)
    must be meta as well, otherwise they are allocated eagerly on the host.
    """
    if not init_model_with_meta_device:
        return hook

    @functools.wraps(hook)
    def wrapped(models: list[MegatronModule]) -> list[MegatronModule]:
        with torch.device("meta"):
            return hook(models)

    return wrapped


def restore_modelopt_state_hook(load_dir: str) -> PreWrapHook:
    """Restore ``modelopt_state`` from ``load_dir`` into each model stage.

    [IMPORTANT] This must happen before the model is returned to the caller.

    ModelOpt can create additional trainable parameters (e.g. for online speculative
    decoding training or PEFT). Hence resuming modelopt_state during checkpoint loading is already
    too late since Megatron creates the optimizer right after building the model, before loading
    the checkpoint. To ensure all trainable parameters are registered, we resume the
    modelopt_state (which transforms the model to have additional parameters) right after build.
    """

    def hook(models: list[MegatronModule]) -> list[MegatronModule]:
        for model in models:
            load_modelopt_state(model=model, load_dir=load_dir)
        return models

    return hook


def freeze_for_qad(model, target):
    """Select which side of an MTP model trains during QAD / MTP QAT.

    Splits parameters into the MTP heads (``mtp.layers.*``) and the base model,
    and freezes one side so controlled QAD+MTP experiments can be run:

    * ``"mtp"``  — train the MTP heads only, freeze the base. Used after QAD:
      load a quantized checkpoint, add MTP heads, and train them while the
      quantized base stays fixed (the production two-phase recipe).
    * ``"base"`` — train the base only, freeze the MTP heads. QAD on the base
      with the MTP head held at its init (e.g. measuring how well a frozen MTP
      head rides on a quantizing base).
    * ``"both"`` — train the base and the MTP heads together (QAD co-training).
    """
    if target not in ("mtp", "base", "both"):
        raise ValueError(f"qad train target must be one of mtp/base/both, got {target!r}")

    if target == "both":
        for param in model.parameters():
            param.requires_grad = True
        # Nothing is frozen, so no router expert_bias should be pinned.
        for module in model.modules():
            if hasattr(module, 'expert_bias'):
                module.frozen_expert_bias = False
        print_rank_0("QAD train target 'both': all parameters trainable")
        return

    train_mtp = target == "mtp"
    trainable, frozen = 0, 0
    for name, param in model.named_parameters():
        is_mtp = 'mtp.layers.' in name
        param.requires_grad = is_mtp == train_mtp
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1

    # The MoE router's expert bias is updated from load-balancing token counts in
    # finalize_model_grads._update_router_expert_bias, independently of requires_grad.
    # Setting requires_grad=False does NOT stop it, so the frozen side would keep
    # drifting. Flag the frozen side's routers so the update is skipped; the trainable
    # side's routers must keep updating (so we clear the flag there).
    frozen_bias = 0
    for name, module in model.named_modules():
        if hasattr(module, 'expert_bias'):
            is_mtp = 'mtp.layers.' in name
            freeze_this = is_mtp != train_mtp
            module.frozen_expert_bias = freeze_this
            if freeze_this:
                frozen_bias += 1
    print_rank_0(
        f"QAD train target '{target}': training {'MTP' if train_mtp else 'base'} "
        f"({trainable} trainable, {frozen} frozen, {frozen_bias} router expert_bias frozen)"
    )


def qad_freeze_hook(target: str) -> PreWrapHook:
    """Apply ``freeze_for_qad(model, target)`` to each model stage."""

    def hook(models: list[MegatronModule]) -> list[MegatronModule]:
        for model in models:
            freeze_for_qad(model, target)
        return models

    return hook


def _propagate_expert_allreduce_to_lsq_params(model):
    """Copy each weight's MCore ``allreduce`` flag onto its LSQ ``_amax_{pre,post}`` params.

    Without the flag, expert amax params get bucketed into the global data-parallel group
    instead of the expert-data-parallel group (wrong grad reduction under expert parallelism).
    """
    module_by_name = dict(model.named_modules())
    for name, param in model.named_parameters():
        if not (name.endswith("_amax_pre") or name.endswith("_amax_post")):
            continue
        marker = ".weight_quantizer"
        if marker not in name:
            continue
        linear = module_by_name.get(name.split(marker, 1)[0])
        if linear is None:
            continue
        allreduce = next(
            (
                w.allreduce
                for wname, w in linear.named_parameters(recurse=False)
                if "weight" in wname and hasattr(w, "allreduce")
            ),
            None,
        )
        if allreduce is not None:
            param.allreduce = allreduce


def lsq_hook(scale_lr: float | None) -> PreWrapHook:
    """Prepare LSQ (scale-learning QAD) parameters for DDP and the distributed optimizer.

    LSQ ``_amax_*`` params lack MCore's ``allreduce`` attribute; propagate it from each weight
    before DDP and the distributed optimizer are built. No-op without LSQ params.
    """

    def hook(models: list[MegatronModule]) -> list[MegatronModule]:
        for model in models:
            _propagate_expert_allreduce_to_lsq_params(model)
        if scale_lr is not None:
            install_lsq_optimizer_overrides()
        return models

    return hook


def te_load_state_dict_hook() -> PreWrapHook:
    """Register a ``load_state_dict`` pre-hook that handles known TE/local state_dict key mismatches."""

    def hook(models: list[MegatronModule]) -> list[MegatronModule]:
        for model in models:
            model._register_load_state_dict_pre_hook(mcore_gpt_load_te_state_dict_pre_hook)
        return models

    return hook


def _load_teacher_model_config(distillation: DistillationConfig, args: Namespace) -> Namespace:
    """Reads teacher config from a file.

    The config provided, either in the teacher checkpoint dir or via ``distillation.teacher_model_config``,
    should specify any model architecture settings which differ from the main student model's.
    The field names should match those returned by get_args() and not TransformerConfig.
    """
    if distillation.teacher_model_config is not None:
        config_path = distillation.teacher_model_config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Teacher model-config file ({config_path}) not found.")
    else:
        config_path = os.path.join(distillation.teacher_load, "model_config.yaml")
        if not os.path.exists(config_path):
            logger.warning(
                "No teacher config provided via --export-kd-teacher-model-config nor found at"
                f" {distillation.teacher_load}/model_config.yaml. Assuming teacher model architecture same as student's."
            )  # Useful for cases like QAD
            config_path = None

    args_dict = vars(args).copy()

    if config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        del args_dict["kv_channels"]  # not recalculated if present
        # Setting teacher Flextron fields to false if training with Flextron, can be overridden
        if "flextron" in args_dict:
            args_dict["flextron"] = False
        if "enable_router" in args_dict:
            args_dict["enable_router"] = False
        if "freeze_model" in args_dict:
            args_dict["freeze_model"] = False

        args_dict.update(config)

    # Backward compat: old checkpoints have hybrid_override_pattern but not hybrid_layer_pattern
    if (args_dict.get('hybrid_override_pattern') is not None
            and args_dict.get('hybrid_layer_pattern') is None):
        args_dict['hybrid_layer_pattern'] = args_dict['hybrid_override_pattern']

    return Namespace(**args_dict)


def _build_teacher(student_model_config, teacher_config_raw: Namespace, teacher_transformer, model):
    """Build the teacher stage matching ``model`` with the same builder class as the student."""
    overrides = {}
    if isinstance(student_model_config, HybridModelConfig):
        # Note: hybrid_override_pattern is remapped to hybrid_layer_pattern in
        # _load_teacher_model_config, so teacher_config_raw.hybrid_layer_pattern is always set here.
        overrides["hybrid_layer_pattern"] = teacher_config_raw.hybrid_layer_pattern

    # Only the architecture differs from the student. Layer specs given as callables
    # (e.g. the ModelOpt GPT spec) are re-evaluated against the teacher's transformer config.
    teacher_model_config = dataclasses.replace(
        student_model_config,
        transformer=teacher_transformer,
        distillation=None,
        pre_wrap_hooks=[],
        post_wrap_hooks=[],
        **overrides,
    )
    builder = teacher_model_config.get_builder_cls()(teacher_model_config)
    pg_collection = getattr(model, "pg_collection", None)
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return builder.build_model(
        pg_collection,
        pre_process=model.pre_process,
        post_process=model.post_process,
        vp_stage=getattr(model, "vp_stage", None),
    )


def kd_convert_hook(student_model_config, args: Namespace, te_mcore_model: bool) -> PreWrapHook:
    """Convert each model stage into a ModelOpt ``DistillationModel`` with its teacher stage.

    The teacher is built by the same ``ModelBuilder`` class as the student, from the student's
    model config with the architecture overridden by the teacher config.

    The teacher's weights are intentionally NOT loaded here. They are loaded by
    ``megatron.post_training.checkpointing.load_kd_teacher_checkpoint()`` alongside the student's
    checkpoint, after the mixed-precision wrapper (e.g. Float16Module) has been applied. Loading
    them before wrapping would lose the wrapper's dtype handling (e.g. fp32 MoE router params).

    Args:
        student_model_config: The student ``ModelConfig``; its ``distillation`` must be set.
        args: The argparse namespace the student config was derived from. Used (read-only)
            to derive the teacher's ``TransformerConfig``.
        te_mcore_model: Whether to register the TE state_dict key-remapping hook on the teacher.
    """
    distillation = student_model_config.distillation
    assert distillation is not None, "kd_convert_hook requires model_config.distillation"

    def hook(models: list[MegatronModule]) -> list[MegatronModule]:
        print_rank_0("Distillation: Enabled.")
        if len(models) != 1:
            raise ValueError("ModelOpt KD does not support virtual pipeline parallelism.")

        teacher_config_raw = _load_teacher_model_config(distillation, args)
        teacher_transformer = core_transformer_config_from_args(teacher_config_raw)
        for attr in _SHARED_STUDENT_TEACHER_ATTRS:
            student_value = getattr(student_model_config.transformer, attr)
            teacher_value = getattr(teacher_transformer, attr)
            if student_value != teacher_value:
                raise ValueError(
                    f"Student and teacher must have the same {attr}, got {student_value} "
                    f"(student) and {teacher_value} (teacher)."
                )

        converted = []
        for model in models:
            distill_cfg = mtd_mcore.setup_distillation_config(
                distillation.kd_cfg, student_cfg=model.config, teacher_cfg=teacher_transformer
            )
            teacher = _build_teacher(
                student_model_config, teacher_config_raw, teacher_transformer, model
            )
            if te_mcore_model:
                teacher._register_load_state_dict_pre_hook(mcore_gpt_load_te_state_dict_pre_hook)

            kd_config = {
                "teacher_model": teacher,
                "criterion": distill_cfg.criterion,
                "loss_balancer": distill_cfg.loss_balancer,
            }
            model = mtd.convert(model, mode=[("kd_loss", kd_config)])

            # Additional tweaks needed for MCore.
            # (accounts for sharded state, pipeline parallel, and potentially skipping LM loss)
            mtd_mcore.adjust_distillation_model_for_mcore(model, distill_cfg)
            converted.append(model)
        return converted

    # Lets later registrations (e.g. freeze hooks) be ordered before the conversion.
    hook.modelopt_kd_convert = True
    return hook
