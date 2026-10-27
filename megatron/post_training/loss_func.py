# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT loss function(s)."""

from typing import Optional

import torch

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.process_groups_config import warn_global_process_group_fallback
from megatron.core.utils import get_attr_wrapped_model, get_pg_rank, get_pg_size, unwrap_model
from megatron.training import get_args


def _mask_loss(output_tensor, loss_mask, tp_group: Optional[torch.distributed.ProcessGroup] = None):
    """Apply mask to the unreduced loss tensor.

    Args:
        output_tensor: The unreduced loss, or a distillation tuple ``(loss, tp_reduce,
            is_sequence_parallel)`` whose flags mark a loss split across tensor-parallel ranks.
        loss_mask: Mask over the tokens of the loss.
        tp_group: Tensor-parallel group of the model. A sequence-parallel loss takes its slice of
            the mask by the rank in this group, and a split loss is all-reduced over it. None
            falls back to the global tensor-parallel group, with a deprecation warning, when a
            distillation loss needs the group.
    """
    if isinstance(output_tensor, tuple):
        # Special distillation flags indicating whether to perform additional tensor-parallel adjustments.
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
    else:
        tp_reduce, is_sequence_parallel = False, False

    if (tp_reduce or is_sequence_parallel) and tp_group is None:
        warn_global_process_group_fallback(
            "megatron.post_training.loss_func._mask_loss",
            "tp_group",
            deprecated_in="0.21",
            removed_in="0.23",
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()

    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = get_pg_rank(tp_group)
        loss_mask = torch.tensor_split(loss_mask, get_pg_size(tp_group), dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    if tp_reduce or is_sequence_parallel:
        # Losses on parallel tensors require extra all-reduce to sync across MP ranks.
        torch.distributed.all_reduce(loss, group=tp_group)

    return loss


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel):
    """Loss function (with KD Loss support).

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
        model (GPTModel): The model (can be wrapped). Distillation losses split across
            tensor-parallel ranks are reduced over the tensor-parallel group of its
            ``pg_collection``. It may be None when knowledge distillation is off; a split loss
            then falls back to the global tensor-parallel group with a deprecation warning.
    """
    args = get_args()

    # Unwrap for both Distillation and LANA
    model = unwrap_model(model)
    tp_group = None if model is None else get_attr_wrapped_model(model, "pg_collection").tp

    # Standard lm loss
    loss_lm = _mask_loss(output_tensor, loss_mask, tp_group=tp_group)
    loss = loss_lm
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss_lm.clone().detach().view(1), num_tokens.view(1)])}

    if args.export_kd_teacher_load:
        # [ModelOpt]: Handle knowledge distillation
        losses = model.compute_kd_loss(
            student_loss=loss_lm,
            loss_reduction_fn=lambda x: _mask_loss(x, loss_mask, tp_group=tp_group),
        )

        report["total loss"] = torch.cat([losses["kd_loss"].clone().detach().view(1), num_tokens.view(1)])
        report["logits distillation loss"] = torch.cat([losses["logits_loss"].clone().detach().view(1), num_tokens.view(1)])
        report["intermediate distillation loss"] = torch.cat([losses["intermediate_loss"].clone().detach().view(1), num_tokens.view(1)])

        if model.training:
            loss = losses["kd_loss"]

    return loss, num_tokens, report
