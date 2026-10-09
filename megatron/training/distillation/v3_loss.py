# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KD loss with explicitly paired targets, without an independent iterator."""

from typing import Any

import torch

from megatron.core import parallel_state as mpu
from megatron.training import get_args

from .cached_logits_loss import add_kd_loss, lm_loss_and_report, topk_kl_div


def paired_loss(loss_mask: Any, output_tensor: Any, model: Any, kd_batch: dict | None) -> tuple:
    """Reuse the cached-logits LM/KD loss and reporting for one replay microbatch."""
    loss_lm, num_tokens, report = lm_loss_and_report(loss_mask, output_tensor)
    if not model.training:
        return loss_lm, num_tokens, report
    if kd_batch is None:
        raise RuntimeError("Training v3 replay requires targets paired with this forward batch")
    logits = kd_batch["logits"]
    values, indices = kd_batch["values"], kd_batch["indices"]
    if tuple(logits.shape[:2]) != tuple(values.shape[:2]):
        raise ValueError("Student logits do not match their v3 batch's token mapping")
    # The sparse KL stabilizes FP32 logits in place. Keep the model output
    # intact when its LM loss backward also needs that tensor.
    kl_logits = logits.clone() if logits.dtype == torch.float32 else logits
    per_token = topk_kl_div(
        kl_logits,
        values,
        indices,
        mpu.get_tensor_model_parallel_world_size(),
        mpu.get_tensor_model_parallel_rank(),
        mpu.get_tensor_model_parallel_group(),
        add_ghost_token=True,
    )
    alpha = get_args().logits_load_kd_loss_alpha
    return add_kd_loss(loss_lm, num_tokens, report, per_token, loss_mask, alpha)
