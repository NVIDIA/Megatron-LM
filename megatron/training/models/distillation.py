# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass


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

    teacher_model_config: str | None = None
    """YAML overriding the student's architecture settings for the teacher. Field names
    follow the argparse names (e.g. ``num_layers``), not ``TransformerConfig``."""

    teacher_ckpt_format: str | None = None
    """Checkpoint format of the teacher, if different from the student's."""

    kd_cfg: str | None = None
    """Path to a ModelOpt distillation-config YAML. Uses the defaults if ``None``."""
