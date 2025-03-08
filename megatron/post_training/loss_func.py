# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT loss function(s)."""

import os

import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args
from megatron.training.utils import average_losses_across_data_parallel_group, unwrap_model


def _mask_loss(output_tensor, loss_mask):
    """Apply mask to the unreduced loss tensor."""
    args = get_args()

    if isinstance(output_tensor, tuple):
        # Special distillation flag indicating whether to perform an additional tensor-parallel reduction.
        output_tensor, tp_reduce = output_tensor
    else:
        tp_reduce = False

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    if tp_reduce and args.tensor_model_parallel_size > 1:
        # Losses such as KL-Div require extra all-reduce to ensure same values across MP-TP partitions.
        loss = torch.sum(tensor_parallel.gather_from_tensor_model_parallel_region(loss.reshape(1)))

    return loss


def _allreduce_loss(loss):
    """Reduce loss for reporting purposes."""
    args = get_args()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, averaged_loss[0]


def loss_func(loss_mask: torch.Tensor, model: GPTModel, output_tensor: torch.Tensor):
    """Loss function (with KD Loss support).

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        model (GPTModel): The model (can be wrapped)
        output_tensor (Tensor): The tensor with the losses
    """
    args = get_args()

    # Unwrap for both Distillation and LANA
    model = unwrap_model(model)

    # Standard lm loss
    loss_lm = _mask_loss(output_tensor, loss_mask)
    loss_lm, loss_lm_avg = _allreduce_loss(loss_lm)

    loss, report = loss_lm, {'lm loss': loss_lm_avg}

    if model.training and args.export_kd_teacher_load and not args.export_kd_finalize:
        # [ModelOpt]: Handle knowledge distillation
        loss_kd = model.compute_kd_loss(
            student_loss=loss, loss_reduction_fn=lambda x: _mask_loss(x, loss_mask)
        )
        loss_kd, loss_kd_avg = _allreduce_loss(loss_kd)

        # Still logs original loss for baseline-comparison purposes.
        loss, report["kd loss"] = loss_kd, loss_kd_avg

    return loss, report
