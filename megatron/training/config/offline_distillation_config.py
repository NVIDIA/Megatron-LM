# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration for producing or consuming an offline teacher-logits cache."""

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True)
class LogitsSaveConfig:
    """Teacher-side cache generation; independent of whether the model is frozen."""

    directory: str
    """Directory or MSC URL to write teacher log-probability shards."""

    top_k: int = 128
    """Maximum number of vocabulary entries to retain per token."""

    top_p: float | None = None
    """Optional nucleus threshold applied after top-k selection."""

    min_k: int = 1
    """Minimum entries retained per token when top-p truncation is enabled."""

    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    """On-disk dtype of the retained log-probabilities."""

    def validate(self) -> None:
        """Check cache settings before attaching model hooks."""
        if not self.directory:
            raise ValueError("teacher_save requires a directory.")
        if self.top_k < 1:
            raise ValueError("teacher_save.top_k must be positive.")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("teacher_save.top_p must be in (0, 1].")
        if not 1 <= self.min_k <= self.top_k:
            raise ValueError("teacher_save.min_k must be between 1 and top_k.")
        if self.dtype not in ("fp16", "bf16", "fp32"):
            raise ValueError("teacher_save.dtype must be fp16, bf16 or fp32.")


@dataclass(kw_only=True)
class LogitsLoadConfig:
    """Student-side cache reading and offline KD loss."""

    directory: str
    """Directory or MSC URL containing the teacher log-probability shards."""

    loss_alpha: float = 1.0
    """Loss mixture: alpha * KD loss + (1 - alpha) * language-model loss."""

    decode_threads: int = 4
    """Threads for decompressing and decoding cached teacher targets."""

    msc_prefetch_depth: int = 2
    """Number of remote tar shards to prefetch; zero disables prefetch."""

    ignore_errors: bool = False
    """Fall back to language-model loss when computing the KD loss fails."""

    ignore_hash: bool = False
    """Skip validation of the cache's dataset-identity hash."""

    def validate(self) -> None:
        """Check reader and loss settings before opening the cache."""
        if not self.directory:
            raise ValueError("student_load requires a directory.")
        if not 0.0 <= self.loss_alpha <= 1.0:
            raise ValueError("student_load.loss_alpha must be in [0, 1].")
        if self.decode_threads < 1:
            raise ValueError("student_load.decode_threads must be positive.")
        if self.msc_prefetch_depth < 0:
            raise ValueError("student_load.msc_prefetch_depth must be nonnegative.")


@dataclass(kw_only=True)
class OfflineDistillationConfig:
    """Exactly one phase of offline logits KD; no live teacher or ModelOpt dependency."""

    teacher_save: LogitsSaveConfig | None = None
    """Produce a teacher cache. Mutually exclusive with student_load."""

    student_load: LogitsLoadConfig | None = None
    """Train against a teacher cache. Mutually exclusive with teacher_save."""

    def validate(self) -> None:
        """Require exactly one phase, then validate its settings."""
        if (self.teacher_save is None) == (self.student_load is None):
            raise ValueError(
                "offline_distillation requires exactly one of teacher_save or student_load."
            )
        config = self.teacher_save if self.teacher_save is not None else self.student_load
        config.validate()
