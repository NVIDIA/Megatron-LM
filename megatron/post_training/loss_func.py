# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT loss function(s)."""

import os

import torch

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args
from megatron.training.utils import average_losses_across_data_parallel_group, unwrap_model


def _mask_loss(output_tensor, loss_mask):
    """Apply mask to the unreduced loss tensor."""
    args = get_args()

    if isinstance(output_tensor, tuple):
        # Special distillation flags indicating whether to perform additional tensor-parallel adjustments.
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
    else:
        tp_reduce, is_sequence_parallel = False, False

    num_tokens = loss_mask.sum().float()
    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = parallel_state.get_tensor_model_parallel_rank()
        loss_mask = torch.tensor_split(loss_mask, args.tensor_model_parallel_size, dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()

    loss = torch.cat([torch.sum(losses * loss_mask).view(1), num_tokens.view(1)])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
    loss = loss[0] / loss[1]

    if tp_reduce or is_sequence_parallel:
        # Losses on parallel tensors require extra all-reduce to sync across MP ranks.
        torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())

    return loss


def _allreduce_losses(losses):
    """Reduce losses across all GPUs."""
    args = get_args()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        for loss in losses:
            assert not loss.isnan(), (
                f'Rank {global_rank}: found NaN in local forward loss calculation. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

    # Reduce loss for logging.
    # TODO(aanoosheh): This should ideally be done with num_tokens separately reduced and averaged.
    return average_losses_across_data_parallel_group(losses)


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
    loss_lm_avg = _allreduce_losses([loss_lm])[0]

    loss, report = loss_lm, {'lm loss': loss_lm_avg}

    if model.training and args.export_kd_teacher_load:
        # [ModelOpt]: Handle knowledge distillation
        losses = model.compute_kd_loss(
            student_loss=loss_lm,
            loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
        )
        loss = losses["kd_loss"]

        losses_avg = _allreduce_losses([losses["kd_loss"], losses["logits_loss"], losses["intermediate_loss"]])
        report["kd loss"] = losses_avg[0]
        report["logits distillation loss"] = losses_avg[1]
        report["intermediate distillation loss"] = losses_avg[2]

    return loss, report
