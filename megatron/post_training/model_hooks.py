# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""ModelOpt pre-wrap hooks for ``ModelConfig.pre_wrap_hooks``.

Each ``*_hook`` factory returns a ``list[MegatronModule] -> list[MegatronModule]`` hook that
runs after the stock GPT/Hybrid builder has built the model stages and before they are
wrapped (Float16Module / DDP). ``modelopt_pre_wrap_hook`` reconstructs them from the
serializable configuration on each build, in this order:

1. ``restore_modelopt_state_hook``: must come before any hook that converts the model.
2. ``qad_freeze_hook``
3. ``lsq_hook``
4. ``te_load_state_dict_hook``
5. User ``pre_wrap_hooks`` (including training freezes)
6. ``kd_convert_hook``
"""

import dataclasses
import functools
from typing import Callable

import modelopt.torch.distill as mtd
import modelopt.torch.distill.plugins.megatron as mtd_mcore
import torch

from megatron.core.post_training.modelopt.gpt.state_dict_hooks import (
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.post_training.checkpointing import load_modelopt_state
from megatron.post_training.optimizer import install_lsq_optimizer_overrides
from megatron.training import print_rank_0
from megatron.training.models.base import compose_hooks

PreWrapHook = Callable[[list[MegatronModule]], list[MegatronModule]]

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


def restore_modelopt_state_hook(load_dir: str, ckpt_format: str | None = None) -> PreWrapHook:
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
            load_modelopt_state(model=model, load_dir=load_dir, ckpt_format=ckpt_format)
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


def _build_teacher(student_model_config, distillation, model):
    """Build the resolved teacher architecture on the same pipeline stage as the student."""
    teacher_model_config = distillation.teacher_model
    if teacher_model_config is None:
        raise ValueError("Distillation requires a resolved teacher_model configuration.")
    # Tokenizer initialization may resolve the student's vocabulary after CLI translation.
    # Student and teacher consume the same token IDs, as in the legacy builder.
    teacher_model_config = dataclasses.replace(
        teacher_model_config,
        vocab_size=student_model_config.vocab_size,
        should_pad_vocab=student_model_config.should_pad_vocab,
        pre_wrap_hooks=[],
        post_wrap_hooks=[],
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


def kd_convert_hook(student_model_config, distillation, remap_te_layernorm: bool) -> PreWrapHook:
    """Convert each model stage into a ModelOpt ``DistillationModel`` with its teacher stage.

    The teacher is built by the same ``ModelBuilder`` class as the student, from the student's
    model config with the architecture overridden by the teacher config.

    The teacher's weights are intentionally NOT loaded here. They are loaded by
    ``megatron.post_training.checkpointing.load_kd_teacher_checkpoint()`` alongside the student's
    checkpoint, after the mixed-precision wrapper (e.g. Float16Module) has been applied. Loading
    them before wrapping would lose the wrapper's dtype handling (e.g. fp32 MoE router params).

    Args:
        student_model_config: The student ``ModelConfig``.
        distillation: The run's ``DistillationConfig`` (teacher and distillation loss).
        remap_te_layernorm: Whether to register the TE layernorm key-remapping load hook on
            the teacher.
    """
    def hook(models: list[MegatronModule]) -> list[MegatronModule]:
        print_rank_0("Distillation: Enabled.")
        if len(models) != 1:
            raise ValueError("ModelOpt KD does not support virtual pipeline parallelism.")

        teacher_transformer = distillation.teacher_model.transformer

        converted = []
        for model in models:
            distill_cfg = mtd_mcore.setup_distillation_config(
                mtd_mcore.DistillationConfig(**distillation.loss_kwargs()),
                student_cfg=model.config,
                teacher_cfg=teacher_transformer,
            )
            teacher = _build_teacher(student_model_config, distillation, model)
            if remap_te_layernorm:
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

    return hook


def modelopt_pre_wrap_hook(model_config, checkpoint=None, distillation=None) -> PreWrapHook:
    """Rebuild setup hooks from configuration, keeping all student freezes before KD.

    Called by the stock builders on every build, including after deserialization.
    Runtime hooks are composed without modifying the config or duplicating registrations.
    """
    hooks = []
    options = model_config.transformations
    init_meta = model_config.transformer.init_model_with_meta_device

    def register(hook):
        hooks.append(on_model_init_device(hook, init_meta))

    if options is not None:
        if checkpoint is not None and checkpoint.load is not None:
            register(restore_modelopt_state_hook(checkpoint.load, checkpoint.ckpt_format))
        if options.qad_train_target is not None:
            register(qad_freeze_hook(options.qad_train_target))
        register(lsq_hook(options.lsq_scale_lr))
        if not options.default_te_spec:
            register(te_load_state_dict_hook())

    hooks.extend(model_config.pre_wrap_hooks)
    if distillation is not None:
        remap_te_layernorm = options is not None and not options.default_te_spec
        register(kd_convert_hook(model_config, distillation, remap_te_layernorm))
    return compose_hooks(hooks)
