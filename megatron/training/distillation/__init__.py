# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.training.distillation.cached_logits_loss import LossFuncCallable, StudentLogitsCapture
from megatron.training.distillation.logits_saver import (
    LogitsSaverHooks,
    check_logits_saver_failure,
    get_logits_saver,
)

__all__ = [
    "LossFuncCallable",
    "LogitsSaverHooks",
    "StudentLogitsCapture",
    "check_logits_saver_failure",
    "get_logits_saver",
]
