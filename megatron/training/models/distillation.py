# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from megatron.training.models.base import ModelConfig

# Fields mirrored from ModelOpt's ``modelopt.torch.distill.plugins.megatron.DistillationConfig``.
_LOSS_FIELDS = (
    "intermediate_layer_pairs",
    "logit_layers",
    "skip_lm_loss",
    "kd_loss_scale",
    "logit_kl_temperature",
    "logit_kl_topk",
)


def load_distillation_loss_yaml(path: str) -> dict[str, Any]:
    """Read the loss settings of a ModelOpt distillation-config YAML (``--export-kd-cfg``)."""
    import yaml

    with open(path) as f:
        values = yaml.safe_load(f) or {}
    unknown = set(values) - set(_LOSS_FIELDS)
    if unknown:
        raise ValueError(f"Unknown distillation settings in {path}: {sorted(unknown)}")
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

    # Distillation loss. Defaults match ModelOpt's; only non-default values are passed to it.
    intermediate_layer_pairs: list[tuple[str, ...]] = field(default_factory=list)
    """``(student_layer, teacher_layer[, "cosine" | "mse"])`` pairs for intermediate losses."""

    logit_layers: tuple[str, str] = ("output_layer", "output_layer")
    """Student and teacher layers whose outputs are the logits."""

    skip_lm_loss: bool = True
    """Train on the distillation loss only, without the language-model loss."""

    kd_loss_scale: float = 1.0
    """Relative scale of the distillation loss when ``skip_lm_loss`` is False."""

    logit_kl_temperature: float = 1.0
    """Temperature of the logit KL-divergence loss."""

    logit_kl_topk: int | None = None
    """If set, compute the logit KL-divergence over the teacher's top-k logits only."""

    kd_cfg: str | None = None
    """Deprecated: path to a ModelOpt distillation-config YAML. Its values are read into the
    loss fields above on construction, and the path is then dropped."""

    def __post_init__(self):
        if self.kd_cfg is not None:
            warnings.warn(
                "DistillationConfig.kd_cfg is deprecated; set the distillation loss fields "
                "directly.",
                DeprecationWarning,
                stacklevel=3,
            )
            for name, value in load_distillation_loss_yaml(self.kd_cfg).items():
                if getattr(self, name) != _LOSS_DEFAULTS[name]:
                    raise ValueError(f"{name} is set both directly and in kd_cfg ({self.kd_cfg}).")
                setattr(self, name, value)
            self.kd_cfg = None
        # Normalize YAML lists so serialized configs compare equal after a round trip.
        self.logit_layers = tuple(self.logit_layers)
        self.intermediate_layer_pairs = [tuple(pair) for pair in self.intermediate_layer_pairs]

    def loss_kwargs(self) -> dict[str, Any]:
        """Non-default loss settings, as keyword arguments for ModelOpt's ``DistillationConfig``."""
        return {
            name: getattr(self, name)
            for name in _LOSS_FIELDS
            if getattr(self, name) != _LOSS_DEFAULTS[name]
        }

    def validate(self) -> None:
        if not self.teacher_load:
            raise ValueError("Distillation requires a teacher checkpoint.")
        if self.teacher_model is None:
            raise ValueError("Distillation requires a resolved teacher_model configuration.")
        if len(self.logit_layers) != 2:
            raise ValueError(f"logit_layers must be a (student, teacher) pair: {self.logit_layers}")
        if any(len(pair) not in (2, 3) for pair in self.intermediate_layer_pairs):
            raise ValueError(f"Invalid intermediate_layer_pairs: {self.intermediate_layer_pairs}")
        if self.kd_loss_scale <= 0 or self.logit_kl_temperature <= 0:
            raise ValueError("kd_loss_scale and logit_kl_temperature must be positive.")

    def validate_student(self, student: ModelConfig) -> None:
        """Check that the student can be distilled from this teacher."""
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


_LOSS_DEFAULTS: dict[str, Any] = {
    f.name: f.default_factory() if f.default_factory is not MISSING else f.default
    for f in fields(DistillationConfig)
    if f.name in _LOSS_FIELDS
}
