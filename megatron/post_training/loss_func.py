# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT loss function(s)."""

import torch

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args
from megatron.training.utils import unwrap_model


def _mask_loss(output_tensor, loss_mask):
    """Apply mask to the unreduced loss tensor."""
    args = get_args()

    if isinstance(output_tensor, tuple):
        # Special distillation flags indicating whether to perform additional tensor-parallel adjustments.
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
    else:
        tp_reduce, is_sequence_parallel = False, False

    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = parallel_state.get_tensor_model_parallel_rank()
        loss_mask = torch.tensor_split(loss_mask, args.tensor_model_parallel_size, dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    if tp_reduce or is_sequence_parallel:
        # Losses on parallel tensors require extra all-reduce to sync across MP ranks.
        torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())

    return loss


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel):
    """Loss function (with KD Loss support).

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
        model (GPTModel): The model (can be wrapped)
    """
    args = get_args()

    # Unwrap for both Distillation and LANA
    model = unwrap_model(model)

    # Standard lm loss
    loss_lm = _mask_loss(output_tensor, loss_mask)
    loss = loss_lm
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss_lm.clone().detach().view(1), num_tokens.view(1)])}

    if model.training and args.export_kd_teacher_load:
        # [ModelOpt]: Handle knowledge distillation
        losses = model.compute_kd_loss(
            student_loss=loss_lm,
            loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
        )
        loss = losses["kd_loss"]

        report["total loss"] = torch.cat([losses["kd_loss"].clone().detach().view(1), num_tokens.view(1)])
        report["logits distillation loss"] = torch.cat([losses["logits_loss"].clone().detach().view(1), num_tokens.view(1)])
        report["intermediate distillation loss"] = torch.cat([losses["intermediate_loss"].clone().detach().view(1), num_tokens.view(1)])

    return loss, num_tokens, report
