# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KD loss with explicitly paired targets, without an independent iterator."""

from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.training import get_args

from .cached_logits_loss import topk_kl_div


def paired_loss(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Any, kd_batch: dict | None
) -> tuple:
    """Reuse sparse KL and loss reporting for one immutable replay microbatch."""
    weights = loss_mask.reshape(-1).float()
    loss_lm = (output_tensor.reshape(-1).float() * weights).sum()
    num_tokens = loss_mask.sum().detach().clone().to(torch.int)
    report = {"lm loss": torch.cat((loss_lm.detach().reshape(1), num_tokens.reshape(1)))}
    if not model.training:
        return loss_lm, num_tokens, report
    if kd_batch is None:
        raise RuntimeError("Training v3 replay requires targets paired with this forward batch")
    logits = kd_batch["logits"]
    values, indices = kd_batch["values"], kd_batch["indices"]
    if tuple(logits.shape[:2]) != tuple(values.shape[:2]):
        raise ValueError("Student logits do not match their v3 batch's token mapping")
    # The legacy KL stabilizes FP32 logits in place. Keep the model output
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
    kd = (per_token.reshape(-1) * weights).sum()
    dist.all_reduce(kd, group=mpu.get_tensor_model_parallel_group())
    alpha = get_args().logits_load_kd_loss_alpha
    total = alpha * kd + (1 - alpha) * loss_lm
    report["logits distillation loss"] = torch.cat((kd.detach().reshape(1), num_tokens.reshape(1)))
    report["total loss"] = torch.cat((total.detach().reshape(1), num_tokens.reshape(1)))
    return total, num_tokens, report
