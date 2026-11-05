# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from megatron.training.models.base import ModelConfig


def load_distillation_loss_yaml(path: str) -> dict[str, Any]:
    """Read a ModelOpt distillation-config YAML (``--export-kd-cfg``) into a dict."""
    import yaml

    with open(path) as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Distillation config {path} must be a mapping, got {type(values)}.")
    return values


@dataclass(kw_only=True)
class DistillationConfig:
    """ModelOpt knowledge distillation: the teacher and the distillation loss.

    Lives on the training run container (``PretrainConfigContainer.distillation``). The teacher
    is built, and the student converted into a ``DistillationModel``, by
    ``megatron.post_training.model_hooks.kd_convert_hook``; the teacher's weights are loaded
    after wrapping by ``megatron.post_training.checkpointing.load_kd_teacher_checkpoint``.
    """

    teacher_load: str
    """Path to the teacher checkpoint. Also used to find ``model_config.yaml`` when
    ``teacher_model_config`` is not given."""

    teacher_model: ModelConfig | None = None
    """Resolved teacher architecture. CLI translation resolves the teacher YAML once;
    serialized configs can then rebuild the teacher without the original CLI namespace.
    """

    teacher_model_config: str | None = None
    """YAML overriding the student's architecture settings for the teacher. Field names
    follow the argparse names (e.g. ``num_layers``), not ``TransformerConfig``.

    Only for ``--export-kd-teacher-model-config``: it is resolved into ``teacher_model`` at
    CLI translation time. YAML run configs should give the teacher as ``teacher_model``, a
    ``ModelConfig`` of its own, rather than as argparse-name overrides."""

    teacher_ckpt_format: str | None = None
    """Checkpoint format of the teacher, if different from the student's."""

    loss: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for ModelOpt's ``modelopt.torch.distill.plugins.megatron.DistillationConfig``
    (the contents of the ``--export-kd-cfg`` YAML). Passed through as is, so its fields follow
    the installed ModelOpt version; ModelOpt validates them when the student is converted.
    Empty means ModelOpt's default logits-only distillation."""

    def validate(self) -> None:
        if not self.teacher_load:
            raise ValueError("Distillation requires a teacher checkpoint.")
        if self.teacher_model is None:
            raise ValueError("Distillation requires a resolved teacher_model configuration.")
        self.teacher_model.validate()
        if not isinstance(self.loss, dict):
            raise ValueError(f"Distillation loss settings must be a mapping, got {self.loss!r}.")


def validate_distillation_student(distillation: DistillationConfig, student: ModelConfig) -> None:
    """Check that ``student`` can be distilled from ``distillation``'s teacher.

    Call after ``distillation.validate()``, which ensures the teacher config is resolved.
    """
    config = student.transformer
    teacher = distillation.teacher_model.transformer
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("ModelOpt KD does not support virtual pipeline parallelism.")
    if config.tp_comm_overlap:
        raise ValueError("ModelOpt distillation is incompatible with tp_comm_overlap.")
    if config.cross_entropy_fusion_impl == "te":
        raise ValueError("ModelOpt distillation is incompatible with TE cross entropy.")
    attrs = [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "pipeline_dtype",
    ]
    # A dense teacher has no expert parallelism, so only an MoE teacher must match the student's.
    if teacher.num_moe_experts is not None:
        attrs.append("expert_model_parallel_size")
    for attr in attrs:
        student_value = getattr(config, attr)
        teacher_value = getattr(teacher, attr)
        if student_value != teacher_value:
            raise ValueError(
                f"Student and teacher must have the same {attr}, got "
                f"{student_value} (student) and {teacher_value} (teacher)."
            )
