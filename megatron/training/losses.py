# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Loss selection and shared validation for GPT and Hybrid training entry points."""

from functools import partial
from typing import TYPE_CHECKING, Callable

import torch

from megatron.core.rerun_state_machine import get_rerun_state_machine

if TYPE_CHECKING:
    from megatron.training.config import PretrainConfigContainer
    from megatron.training.config.resilience_config import RerunStateMachineConfig


LossResult = tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
LossFunction = Callable[..., LossResult]


def language_model_loss(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, *, model=None
) -> LossResult:
    """Return the masked LM loss sum, token count and local reporting totals."""
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    return loss, num_tokens, report


def _checked_loss(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    model=None,
    loss_func: LossFunction,
    checks: "RerunStateMachineConfig",
) -> LossResult:
    """Apply the run's rerun checks to the selected algorithm's loss."""
    loss, num_tokens, report = loss_func(loss_mask, output_tensor, model=model)
    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if checks.check_for_nan_in_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=True,
        )
    # Check for spiky loss
    if checks.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=checks.spiky_loss_factor,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are deterministic
            fatal=False,
        )

    return loss, num_tokens, report


def build_loss_func(cfg: "PretrainConfigContainer") -> LossFunction:
    """Build the configured loss callback, defaulting to ordinary LM loss.

    The caller owns the callback and passes it to forward_step. The offline callback
    retains its reader across microbatches; no state is attached to the model or config.
    Online KD is selected only by the explicit distillation config, never model inspection.
    """
    loss_func = language_model_loss

    offline_kd_cfg = cfg.offline_distillation
    if offline_kd_cfg is not None and offline_kd_cfg.student_load is not None:
        from megatron.training.distillation import LossFuncCallable

        load_cfg = offline_kd_cfg.student_load
        loss_func = LossFuncCallable(
            logprobs_dir=load_cfg.directory,
            decode_threads=load_cfg.decode_threads,
            msc_prefetch_depth=load_cfg.msc_prefetch_depth,
            kd_loss_alpha=load_cfg.loss_alpha,
            ignore_errors=load_cfg.ignore_errors,
            ignore_hash=load_cfg.ignore_hash,
        )
    elif cfg.distillation is not None:
        from megatron.post_training.loss_func import loss_func as modelopt_loss

        # ModelOpt retains its own train/eval behavior and parallel reductions.
        loss_func = modelopt_loss

    return partial(_checked_loss, loss_func=loss_func, checks=cfg.rerun_state_machine)
