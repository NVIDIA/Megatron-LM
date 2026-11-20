# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from megatron.training.models.base import ModelConfig


@dataclass(kw_only=True)
class CachedLogitsConfig:
    """Existing offline teacher-logit recording and student-logit loss options."""

    logits_save_dir: str | None = None
    """Directory in which to record teacher logits; None disables recording."""
    logits_save_top_k: int = 128
    """Maximum number of teacher logits saved per token."""
    logits_save_top_p: float | None = None
    """Optional cumulative probability threshold for saved teacher logits."""
    logits_save_top_p_min_k: int = 1
    """Minimum number of logits retained with top-p recording."""
    logits_save_dtype: str = "fp16"
    """Storage dtype for recorded logits."""
    logits_load_dir: str | None = None
    """Directory containing cached teacher logits; None uses the ordinary LM loss."""
    logits_load_decode_threads: int = 4
    """Number of threads decoding cached logits."""
    logits_load_msc_prefetch_depth: int = 2
    """Number of cached-logit files prefetched through MSC."""
    logits_load_kd_loss_alpha: float = 1.0
    """Weight of the cached-logit distillation loss."""
    logits_load_ignore_errors: bool = False
    """Skip cached-logit read errors, following the existing CLI policy."""
    logits_load_ignore_hash: bool = False
    """Skip cached-logit token-hash verification."""


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
    """Path to the teacher checkpoint."""

    teacher_model: ModelConfig | None = None
    """Resolved teacher architecture. CLI translation resolves the teacher YAML once;
    serialized configs can then rebuild the teacher without the original CLI namespace.
    """

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
