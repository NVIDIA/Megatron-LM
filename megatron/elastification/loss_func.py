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
    if isinstance(output_tensor, tuple) and len(output_tensor) == 2:
        (output_tensor, (param_loss, extra_reporting_dict)) = output_tensor
        tp_reduce, is_sequence_parallel = False, False
    elif isinstance(output_tensor, tuple):
        # Special distillation flags indicating whether to perform additional tensor-parallel adjustments.
        #(tensor([[0, 0, 0,  ..., 0, 0, 0]], device='cuda:0'), (tensor([0.2500], device='cuda:0', grad_fn=<ToCopyBackward0>), {}))
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
        param_loss = None
    else:
        tp_reduce, is_sequence_parallel = False, False
        param_loss = None

    num_tokens = loss_mask.sum().float()
    
    # Sharath: param loss for flextron copied from Ali
    if param_loss is not None:
        if param_loss > 0:
            #param_loss_report = param_loss.detach().clone()
            pass
        else:
            #param_loss_report = param_loss.detach().clone()
            param_loss = -args.router_beta * param_loss

    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = parallel_state.get_tensor_model_parallel_rank()
        loss_mask = torch.tensor_split(loss_mask, args.tensor_model_parallel_size, dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Ali: param loss for flextron
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


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel, selected_budget: float = None):
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

    param_loss_item_avg = None
    if param_loss_item is not None:
        torch.distributed.all_reduce(param_loss_item, group=parallel_state.get_data_parallel_group())

    loss = loss_lm
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    # Protect against division by zero when all tokens are masked.
    num_tokens = torch.clamp(num_tokens, min=1)
    reporting_loss_lm = torch.cat([loss_lm.clone().detach().view(1), num_tokens.view(1)])
    torch.distributed.all_reduce(reporting_loss_lm, group=parallel_state.get_data_parallel_group())
    report = {'lm loss': ((reporting_loss_lm[0].clone().detach().view(1)-param_loss_item.clone().detach().view(1)).detach().clone(), reporting_loss_lm[1]),
              'param loss item': (param_loss_item.clone().detach().view(1), reporting_loss_lm[1])}

    # Add per-model LM loss breakdown for logging only when KD is NOT active
    kd_active = model.training and args.export_kd_teacher_load
    if not kd_active:
        try:
            is_full_model = (param_loss_item is None) or (param_loss_item.detach().abs() == 0)
        except Exception:
            is_full_model = False
        zero_num = report['lm loss'][0].clone().detach() * 0.0
        zero_den = report['lm loss'][1].clone().detach() * 0.0
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
        # loss_lm = ce_loss + param_loss_item (pre-DP-all-reduce), matching the dev-branch
        # formula: total = original_loss + logits_kd + scaled_intermed.
        losses = model.compute_kd_loss(
            student_loss=loss_lm,
            loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
        )
        loss = losses["kd_loss"] + param_loss_item
        # losses_avg = _allreduce_losses([losses["kd_loss"], losses["logits_loss"], losses["intermediate_loss"]])
        # [kd_loss, logits_loss, intermed_loss] — local values, all-reduced below.
        # param_loss_item is already all-reduced (line 113), so keep it separate to
        # avoid double-reduction. Combine after both reductions are done.
        reporting_losses = torch.stack([losses["kd_loss"].detach(), losses["logits_loss"].detach(), losses["intermediate_loss"].detach()]).to(device=reporting_loss_lm.device)
        # All-gather logits_loss across all data parallel ranks into a single tensor
        logits_loss = losses["logits_loss"].detach()
        dp_world_size = torch.distributed.get_world_size(group=parallel_state.get_data_parallel_group())
        logits_loss_gathered = [torch.zeros_like(logits_loss) for _ in range(dp_world_size)]
        torch.distributed.all_gather(
            logits_loss_gathered,
            logits_loss,
            group=parallel_state.get_data_parallel_group()
        )
        logits_loss_gathered = torch.stack(logits_loss_gathered)
        torch.distributed.all_reduce(reporting_losses, group=parallel_state.get_data_parallel_group())

        # True total = logits_kd (already DP-reduced in reporting_losses[0]) + param_loss (already DP-reduced).
        total_loss_report = reporting_losses[0] + param_loss_item.detach()
        report["total loss"] = (total_loss_report, reporting_loss_lm[1])

        # Log KD loss split into full vs budget similar to LM loss breakdown.
        try:
            is_full_model_kd = (param_loss_item is None) or (param_loss_item.detach().abs() == 0)
        except Exception:
            is_full_model_kd = False
        zero_num_kd = total_loss_report.clone().detach() * 0.0
        zero_den_kd = reporting_loss_lm[1].clone().detach() * 0.0
        if is_full_model_kd:
            report["kd loss (full)"] = (total_loss_report, reporting_loss_lm[1])
            report["kd loss (budget)"] = (zero_num_kd, zero_den_kd)
        else:
            report["kd loss (budget)"] = (total_loss_report, reporting_loss_lm[1])
            report["kd loss (full)"] = (zero_num_kd, zero_den_kd)
        report["logits distillation loss"] = (reporting_losses[1], reporting_loss_lm[1])
        report["intermediate distillation loss"] = (reporting_losses[2], reporting_loss_lm[1])

        local_budget = torch.tensor(
            [selected_budget],
            dtype=torch.float32,
            device=logits_loss.device
        )
        budgets_gathered = [torch.zeros_like(local_budget) for _ in range(dp_world_size)]
        torch.distributed.all_gather(
            budgets_gathered,
            local_budget,
            group=parallel_state.get_data_parallel_group()
        )
        budgets_gathered = torch.cat(budgets_gathered)

        # Create a binary mask where gathered budgets are equal to selected_budget (with 1e-6 tolerance)
        budget_mask = (budgets_gathered - selected_budget).abs() < 1e-6
        logits_loss_gathered_selected = logits_loss_gathered[budget_mask].sum()/budget_mask.sum()
        budget_num_tokens = reporting_loss_lm[1] * budget_mask.sum()/budget_mask.shape[0]/budget_mask.sum()

        corrected_budget_list = list(set(args.budget_list))

        for temp_budget in corrected_budget_list:
            report[f"logits distillation loss {temp_budget:.3f}"] = (
                torch.tensor(0.0, device=logits_loss.device, dtype=torch.float32),
                torch.tensor(0.0, device=logits_loss.device, dtype=torch.float32)
            )
        index_of_selected_budget = corrected_budget_list.index(selected_budget)
        all_budget_logit = torch.zeros(len(corrected_budget_list), device=logits_loss.device, dtype=logits_loss.dtype)
        all_budget_tokens = torch.zeros(len(corrected_budget_list), device=logits_loss.device, dtype=logits_loss.dtype)

        all_budget_logit[index_of_selected_budget] = logits_loss_gathered_selected
        all_budget_tokens[index_of_selected_budget] = budget_num_tokens
        torch.distributed.all_reduce(all_budget_logit, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(all_budget_tokens, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM)

        for i in range(len(corrected_budget_list)):
            report[f"logits distillation loss {corrected_budget_list[i]:.3f}"] = (all_budget_logit[i], all_budget_tokens[i])
        
        # Add per-budget lm loss tracking following same pattern as logits distillation loss
        # Only do this if lm loss is enabled (not being skipped by distillation config)

        # current_lm_loss = reporting_loss_lm[0] - param_loss_item.clone().detach()
        # lm_loss_enabled = current_lm_loss.abs() > 1e-8  # Check if lm loss is meaningful (not just zeros)
        
        # if lm_loss_enabled:
        #     for temp_budget in corrected_budget_list:
        #         report[f"lm loss {temp_budget:.3f}"] = (
        #             torch.tensor(0.0, device=logits_loss.device, dtype=logits_loss.dtype),
        #             torch.tensor(0.0, device=logits_loss.device, dtype=logits_loss.dtype)
        #         )
            
        #     # All-gather lm loss across all data parallel ranks
        #     lm_loss_gathered = [torch.zeros_like(current_lm_loss) for _ in range(dp_world_size)]
        #     torch.distributed.all_gather(
        #         lm_loss_gathered,
        #         current_lm_loss,
        #         group=parallel_state.get_data_parallel_group()
        #     )
        #     lm_loss_gathered = torch.stack(lm_loss_gathered)
            
        #     # Select lm loss for current budget
        #     lm_loss_gathered_selected = lm_loss_gathered[budget_mask].sum()/budget_mask.sum()
            
        #     # Create tensors to track lm loss values per budget
        #     all_budget_lm_loss = torch.zeros(len(corrected_budget_list), device=logits_loss.device, dtype=logits_loss.dtype)
        #     all_budget_lm_tokens = torch.zeros(len(corrected_budget_list), device=logits_loss.device, dtype=logits_loss.dtype)
            
        #     # Set values for current selected budget
        #     all_budget_lm_loss[index_of_selected_budget] = lm_loss_gathered_selected
        #     all_budget_lm_tokens[index_of_selected_budget] = budget_num_tokens
            
        #     # All-reduce across data parallel group
        #     torch.distributed.all_reduce(all_budget_lm_loss, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM)
        #     torch.distributed.all_reduce(all_budget_lm_tokens, group=parallel_state.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM)
            
        #     # Update report with per-budget lm loss values
        #     for i in range(len(corrected_budget_list)):
        #         report[f"lm loss {corrected_budget_list[i]:.3f}"] = (all_budget_lm_loss[i], all_budget_lm_tokens[i])

        # # added by Sharath
        # report["param loss"] = param_loss_item_avg
        # report["lm loss"] = reporting_loss_lm[0]

    # Convert all items in report dict to torch.tensor
    for key, val in report.items():
        # If value is a tuple, convert both items to tensors (if not already)
        assert isinstance(val, tuple), "Value is not a tuple"
        report[key] = torch.tensor(val, device=reporting_loss_lm[0].device, dtype=reporting_loss_lm[0].dtype)

    return loss, num_tokens, report
