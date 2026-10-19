# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.training.models.base import ModelConfig


@dataclass(kw_only=True)
class DistillationConfig:
    """Declarative description of a ModelOpt knowledge-distillation teacher.

    Purely configuration: the teacher is constructed (and converted into a
    ``DistillationModel``) by ``megatron.post_training.model_hooks.kd_convert_hook``,
    which builds it with the same ``ModelBuilder`` class as the student.
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
    follow the argparse names (e.g. ``num_layers``), not ``TransformerConfig``."""

    teacher_ckpt_format: str | None = None
    """Checkpoint format of the teacher, if different from the student's."""

    kd_cfg: str | None = None
    """Path to a ModelOpt distillation-config YAML. Uses the defaults if ``None``."""

    def validate(self, student: ModelConfig) -> None:
        if not self.teacher_load:
            raise ValueError("Distillation requires a teacher checkpoint.")
        if self.teacher_model is None:
            raise ValueError("Distillation requires a resolved teacher_model configuration.")
        if self.teacher_model.distillation is not None:
            raise ValueError("Nested teacher distillation is not supported.")
        config = student.transformer
        if config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("ModelOpt KD does not support virtual pipeline parallelism.")
        if config.tp_comm_overlap:
            raise ValueError("ModelOpt distillation is incompatible with tp_comm_overlap.")
        if config.cross_entropy_fusion_impl == "te":
            raise ValueError("ModelOpt distillation is incompatible with TE cross entropy.")
        for attr in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "pipeline_dtype",
        ):
            student_value = getattr(config, attr)
            teacher_value = getattr(self.teacher_model.transformer, attr)
            if student_value != teacher_value:
                raise ValueError(
                    f"Student and teacher must have the same {attr}, got "
                    f"{student_value} (student) and {teacher_value} (teacher)."
                )
