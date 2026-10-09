# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.training.distillation.cached_logits_loss import LossFuncCallable, StudentLogitsCapture
from megatron.training.distillation.logits_saver import (
    LogitsSaverHooks,
    begin_logits_attempt,
    build_logits_saver,
    check_logits_saver_failure,
    commit_logits_attempt,
    get_logits_saver,
)

__all__ = [
    "LossFuncCallable",
    "LogitsSaverHooks",
    "StudentLogitsCapture",
    "begin_logits_attempt",
    "build_logits_saver",
    "check_logits_saver_failure",
    "commit_logits_attempt",
    "get_logits_saver",
]
