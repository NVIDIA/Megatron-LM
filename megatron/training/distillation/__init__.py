# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.training.distillation.cached_logits_loss import LossFuncCallable
from megatron.training.distillation.logits_saver import LogitsSaverHooks, get_logits_saver

__all__ = [
    "LossFuncCallable",
    "LogitsSaverHooks",
    "get_logits_saver",
]
