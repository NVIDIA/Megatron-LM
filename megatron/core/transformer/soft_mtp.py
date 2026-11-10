# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Soft-MTP: balance observed-token supervision and NTP self-distillation.

``soft_mtp_weight`` (α) mixes sample-based MTP representation gradients from observed
tokens with teacher-averaged gradients based on the NTP prediction for the same target.
That prediction is treated as a fixed target when computing the MTP gradient.
α = 0 is standard MTP; α = 1 is Soft-MTP. These auxiliary gradients reach the
MTP modules and shared backbone.

The router recovers the mixture from existing cross-entropy gradients in backward. Forward
values, reported losses, the ordinary NTP gradient, and the shared output projection's
hard-label gradient stay unchanged. The overall MTP loss weight remains independent of α.
This module applies the current α; scheduling it belongs to the training layer.

Reference: "Multi-Token Prediction Done Right", Sec. 5.1 (https://arxiv.org/abs/2610.38331).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def validate_soft_mtp_config(config: TransformerConfig) -> None:
    """Reject settings that Soft-MTP does not support or has not been validated with.

    Called only while Soft-MTP routing is active (α > 0 in a training forward with
    gradients), so inference with the same config is unaffected.
    """
    if not config.mtp_num_layers:
        raise ValueError("soft_mtp_weight > 0 requires mtp_num_layers > 0.")
    loss_weight = config.mtp_loss_scaling_factor
    if loss_weight is None or not math.isfinite(loss_weight) or loss_weight < 0:
        raise ValueError(
            "soft_mtp_weight > 0 requires a finite, nonnegative mtp_loss_scaling_factor."
        )
    if config.mtp_grad_scale_func is not None:
        raise ValueError(
            "soft_mtp_weight > 0 requires the same backward scale for NTP and MTP losses, "
            "so mtp_grad_scale_func must be None."
        )
    unsupported = {
        "mtp_loss_early_backward": config.mtp_loss_early_backward,
        "mtp_detach_heads": config.mtp_detach_heads,
        "freeze_base_model_for_mtp": config.freeze_base_model_for_mtp,
        "overlap_moe_expert_parallel_comm": config.overlap_moe_expert_parallel_comm,
        "fp8_output_proj": config.fp8_output_proj,
        "quant_recipe": config.quant_recipe is not None,
        "cuda_graph_impl": config.cuda_graph_impl != "none",
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError(f"soft_mtp_weight > 0 does not support {', '.join(enabled)}.")


class SoftMTPGradientRouter(torch.autograd.Function):
    """Pass main states through and adjust each MTP head's gradient during backward.

    Let p be an MTP head's prediction, q the NTP prediction for the same target,
    y the hard target, and c_k the head's loss coefficient. The MTP gradient treats q
    as a fixed target; the main model still receives its own NTP training gradient.

    With shared linear output projection W, subtract ``α * c_k * W^T(q - y)`` from
    the head's hard-label gradient ``c_k * W^T(p - y)``. This gives
    ``c_k * [(1 - α) W^T(p - y) + α W^T(p - q)]``: observed-token supervision mixed
    with NTP self-distillation. The output projection retains hard-label supervision.

    ``corrections`` has shape [heads, batch, sequence], with α * c_k on valid
    positions and zero elsewhere. The incoming NTP gradient already includes the loss mask.
    ``source_count`` undoes a main-loss mean and is one for summed losses. ``roll`` aligns
    each head with the NTP prediction for the same target, respecting document and CP
    boundaries. No vocabulary-sized tensors are saved.
    """

    @staticmethod
    def forward(
        ctx,
        main_hidden_states: Tensor,
        corrections: Tensor,
        source_count: Tensor,
        roll: Callable[[Tensor], Tensor],
        sequence_parallel_group: Optional[torch.distributed.ProcessGroup],
        *head_hidden_states: Tensor,
    ) -> Tensor:
        """Keep forward values unchanged and record how to route gradients in backward.

        Each head is an input so autograd can send a correction back through its graph.
        Save this forward's coefficients so a later α update cannot change its backward.
        """
        # Treat this forward's correction coefficients and source count as constants in backward.
        ctx.save_for_backward(corrections.detach(), source_count.detach())
        # Keep the current document/CP alignment callback and SP group for backward.
        ctx.roll = roll
        ctx.sequence_parallel_group = sequence_parallel_group
        # Return corrections in each head's dtype without saving its activation tensor.
        ctx.head_dtypes = tuple(head.dtype for head in head_hidden_states)
        return main_hidden_states

    @staticmethod
    def backward(ctx, main_hidden_states_grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Preserve the main NTP gradient and return one correction per MTP head.

        Reconstruct any SP slices, undo any main-loss mean, and align the NTP gradient
        with each head's targets. Autograd adds each correction to that head's ordinary
        hard-label gradient before propagating through its layers and the backbone.
        """
        corrections, source_count = ctx.saved_tensors
        sequence_parallel_group = ctx.sequence_parallel_group
        sequence_parallel_size = (
            1 if sequence_parallel_group is None else sequence_parallel_group.size()
        )

        # Reconstruct the full CP-local sequence before shifting token positions.
        routed_gradient = main_hidden_states_grad
        if sequence_parallel_size > 1:
            routed_gradient = routed_gradient.new_empty(
                (sequence_parallel_size * routed_gradient.shape[0], *routed_gradient.shape[1:])
            )
            torch.distributed.all_gather_into_tensor(
                routed_gradient, main_hidden_states_grad.contiguous(), group=sequence_parallel_group
            )

        # Scale in FP32. Undoing a mean multiplies by the token count, which can overflow FP16
        # before the correction coefficients scale it back down. Low-precision rounding then
        # happens once, in the final cast to each head's dtype. This cannot recover precision
        # already lost in the output projection's input gradient.
        if routed_gradient.dtype in (torch.float16, torch.bfloat16):
            routed_gradient = routed_gradient.float()
        # Undo the token-count division, then put sequence last for the label roll.
        routed_gradient = routed_gradient * source_count
        routed_gradient = routed_gradient.movedim(0, -1).contiguous()

        head_corrections = []
        for correction_coefficients, head_dtype in zip(corrections, ctx.head_dtypes):
            # Shift once per head depth to align matching targets in logical sequence order.
            routed_gradient = ctx.roll(routed_gradient)
            # Match [sequence, batch, hidden] with [sequence, batch, 1] coefficients.
            # The minus sign subtracts the aligned NTP-gradient copy from the head gradient.
            head_correction = routed_gradient.movedim(-1, 0) * (-correction_coefficients).transpose(
                0, 1
            ).unsqueeze(-1)
            if sequence_parallel_size > 1:
                # Return only the sequence slice owned by this SP rank.
                head_correction = head_correction.chunk(sequence_parallel_size, dim=0)[
                    sequence_parallel_group.rank()
                ]
            # A contiguous copy in the head's dtype gives the local correction its own storage.
            head_corrections.append(
                head_correction.to(
                    dtype=head_dtype, memory_format=torch.contiguous_format, copy=True
                )
            )

        # Preserve the NTP gradient; the four routing inputs receive None, followed by
        # one correction per head in the same order as the forward inputs.
        return main_hidden_states_grad, None, None, None, None, *head_corrections
