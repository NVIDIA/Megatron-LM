# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Flextron loss function(s).

Combines lm loss with the router's budget loss, optional KD distillation
loss, and per-budget reporting (full-model vs sub-budget breakdown).
"""

import torch

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args
from megatron.training.utils import unwrap_model


def _mask_loss(output_tensor, loss_mask):
    """Apply mask to the unreduced loss tensor."""
    args = get_args()
    if isinstance(output_tensor, tuple) and len(output_tensor) == 2:
        (output_tensor, (param_loss, extra_reporting_dict)) = output_tensor
        tp_reduce, is_sequence_parallel = False, False
    elif isinstance(output_tensor, tuple):
        # Special distillation flags indicating whether to perform additional tensor-parallel adjustments.
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
        param_loss = None
    else:
        tp_reduce, is_sequence_parallel = False, False
        param_loss = None

    num_tokens = loss_mask.sum().float()

    if param_loss is not None:
        if param_loss > 0:
            pass
        else:
            param_loss = -args.router_beta * param_loss

    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = parallel_state.get_tensor_model_parallel_rank()
        loss_mask = torch.tensor_split(loss_mask, args.tensor_model_parallel_size, dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    alpha = args.loss_alpha
    if not args.freeze_router and param_loss is not None:
        param_loss_item = param_loss[0] * num_tokens * alpha
        # add param loss to lm loss
        loss += param_loss_item
    else:
        param_loss_item = None

    if tp_reduce or is_sequence_parallel:
        # Losses on parallel tensors require extra all-reduce to sync across MP ranks.
        torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())

    if param_loss_item is not None:
        return loss, param_loss_item
    else:
        return loss


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: GPTModel,
    selected_budget: float = None,
):
    """Loss function (with KD Loss support).

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
        model (GPTModel): The model (can be wrapped)
        selected_budget (float): The budget value used for this forward pass
    """
    args = get_args()

    # Unwrap for both Distillation and LANA
    model = unwrap_model(model)

    # Standard lm loss
    out_mask_loss = _mask_loss(output_tensor, loss_mask)

    if isinstance(out_mask_loss, tuple):
        loss_lm, param_loss_item = out_mask_loss
    else:
        # assert args.freeze_router, "Param loss None is not supported without freezing router"
        loss_lm = out_mask_loss
        param_loss_item = torch.tensor(0.0, device=loss_lm.device, dtype=loss_lm.dtype)

    loss = loss_lm
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    # Protect against division by zero when all tokens are masked.
    num_tokens = torch.clamp(num_tokens, min=1)
    # Report (value, num_tokens) as local-rank values; the training loop performs the
    # DP+CP all-reduce on report-dict tuples (training.py: token-weighted reduction).
    report = {
        'lm loss': ((loss_lm.detach() - param_loss_item.detach()).view(1), num_tokens),
        'param loss item': (param_loss_item.detach().view(1), num_tokens),
    }

    # Add per-model LM loss breakdown for logging only when KD is NOT active
    kd_active = model.training and args.export_kd_teacher_load
    if not kd_active:
        try:
            is_full_model = (param_loss_item is None) or (param_loss_item.detach().abs() == 0)
        except Exception:
            is_full_model = False
        zero_num = torch.zeros_like(report['lm loss'][0])
        zero_den = torch.zeros_like(num_tokens)
        if is_full_model:
            report['lm loss (full)'] = report['lm loss']
            report['lm loss (budget)'] = (zero_num, zero_den)
        else:
            report['lm loss (budget)'] = report['lm loss']
            report['lm loss (full)'] = (zero_num, zero_den)

    if model.training and args.export_kd_teacher_load:
        # [ModelOpt]: Handle knowledge distillation.
        # The installed balancer with skip_lm_loss=True drops student_loss (param_loss) from
        # the total. Add loss_lm back manually to restore the router gradient signal.
        losses = model.compute_kd_loss(
            student_loss=loss_lm, loss_reduction_fn=lambda x: _mask_loss(x, loss_mask)
        )
        loss = losses["kd_loss"] + param_loss_item
        # All-gather logits_loss across DP ranks so we can mask by selected_budget below.
        logits_loss = losses["logits_loss"].detach()
        dp_world_size = torch.distributed.get_world_size(
            group=parallel_state.get_data_parallel_group()
        )
        logits_loss_gathered = [torch.zeros_like(logits_loss) for _ in range(dp_world_size)]
        torch.distributed.all_gather(
            logits_loss_gathered, logits_loss, group=parallel_state.get_data_parallel_group()
        )
        logits_loss_gathered = torch.stack(logits_loss_gathered)

        total_loss_report = losses["kd_loss"].detach() + param_loss_item.detach()
        report["total loss"] = (total_loss_report, num_tokens)

        # Log KD loss split into full vs budget similar to LM loss breakdown.
        try:
            is_full_model_kd = (param_loss_item is None) or (param_loss_item.detach().abs() == 0)
        except Exception:
            is_full_model_kd = False
        zero_num_kd = torch.zeros_like(total_loss_report)
        zero_den_kd = torch.zeros_like(num_tokens)
        if is_full_model_kd:
            report["kd loss (full)"] = (total_loss_report, num_tokens)
            report["kd loss (budget)"] = (zero_num_kd, zero_den_kd)
        else:
            report["kd loss (budget)"] = (total_loss_report, num_tokens)
            report["kd loss (full)"] = (zero_num_kd, zero_den_kd)
        report["logits distillation loss"] = (losses["logits_loss"].detach(), num_tokens)
        report["intermediate distillation loss"] = (
            losses["intermediate_loss"].detach(),
            num_tokens,
        )

        local_budget = torch.tensor(
            [selected_budget], dtype=torch.float32, device=logits_loss.device
        )
        budgets_gathered = [torch.zeros_like(local_budget) for _ in range(dp_world_size)]
        torch.distributed.all_gather(
            budgets_gathered, local_budget, group=parallel_state.get_data_parallel_group()
        )
        budgets_gathered = torch.cat(budgets_gathered)

        # Create a binary mask where gathered budgets are equal to selected_budget (with 1e-6 tolerance)
        budget_mask = (budgets_gathered - selected_budget).abs() < 1e-6
        logits_loss_gathered_selected = logits_loss_gathered[budget_mask].sum() / budget_mask.sum()
        budget_num_tokens = (
            num_tokens.float() * budget_mask.sum() / budget_mask.shape[0] / budget_mask.sum()
        )

        corrected_budget_list = list(set(args.budget_list))

        for temp_budget in corrected_budget_list:
            report[f"logits distillation loss {temp_budget:.3f}"] = (
                torch.tensor(0.0, device=logits_loss.device, dtype=torch.float32),
                torch.tensor(0.0, device=logits_loss.device, dtype=torch.float32),
            )
        index_of_selected_budget = corrected_budget_list.index(selected_budget)
        all_budget_logit = torch.zeros(
            len(corrected_budget_list), device=logits_loss.device, dtype=logits_loss.dtype
        )
        all_budget_tokens = torch.zeros(
            len(corrected_budget_list), device=logits_loss.device, dtype=logits_loss.dtype
        )

        all_budget_logit[index_of_selected_budget] = logits_loss_gathered_selected
        all_budget_tokens[index_of_selected_budget] = budget_num_tokens

        for i in range(len(corrected_budget_list)):
            report[f"logits distillation loss {corrected_budget_list[i]:.3f}"] = (
                all_budget_logit[i],
                all_budget_tokens[i],
            )

    # Convert all items in report dict to a single (value, num_tokens) tensor.
    for key, val in report.items():
        assert isinstance(val, tuple), "Value is not a tuple"
        report[key] = torch.tensor(
            [val[0], val[1].view(1)], device=loss_lm.device, dtype=loss_lm.dtype
        )

    return loss, num_tokens, report
