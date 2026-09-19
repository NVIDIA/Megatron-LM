# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Modality-specific auxiliary-loss-free routing using the standard MoE dispatcher."""

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.router import Router


class ModalityBalance(nn.Module):
    """One modality's bias and batch counters, consumed by finalize_model_grads."""

    def __init__(self, num_experts) -> None:
        super().__init__()
        self.register_buffer(
            "expert_bias",
            mark_keep_in_fp32(
                torch.zeros(num_experts, dtype=torch.float32, device=torch.cuda.current_device())
            ),
        )
        self.register_buffer(
            "local_tokens_per_expert", torch.zeros(num_experts, dtype=torch.int64), persistent=False
        )
        self.frozen_expert_bias = False


class ModalityRouter(Router):
    """Select using modality biases, but weight experts using unbiased sqrt-softplus scores."""

    def __init__(self, config, pg_collection, is_mtp_layer=False) -> None:
        super().__init__(config, pg_collection, is_mtp_layer)
        if config.moe_router_score_function != "sqrtsoftplus":
            raise ValueError("V4.1 modality routing requires sqrtsoftplus scores")
        if config.moe_aux_loss_coeff:
            raise ValueError("V4.1 modality routing uses auxiliary-loss-free balancing")
        self.text_balance = ModalityBalance(config.num_moe_experts)
        self.image_balance = ModalityBalance(config.num_moe_experts)

    def routing(self, logits, image_mask=None, padding_mask=None):
        """Route sequence-major tokens with separate, independently updated biases."""
        scores = F.softplus(logits.float().reshape(-1, self.num_experts)).sqrt()
        image = (
            torch.zeros(scores.shape[0], device=scores.device, dtype=torch.bool)
            if image_mask is None
            else image_mask.reshape(-1)
        )
        bias = torch.where(
            image[:, None],
            self.image_balance.expert_bias.float(),
            self.text_balance.expert_bias.float(),
        )
        indices = (scores + bias).topk(self.config.moe_router_topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.config.moe_router_topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.config.moe_router_topk_scaling_factor
        probs = torch.zeros_like(scores).scatter(-1, indices, weights)
        route = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, indices, True)
        if padding_mask is not None:
            route = route & ~padding_mask.reshape(-1, 1)
            probs = probs * route
        if self.training and torch.is_grad_enabled() and self.config.moe_router_enable_expert_bias:
            with torch.no_grad():
                self.text_balance.local_tokens_per_expert.add_((route & ~image[:, None]).sum(0))
                self.image_balance.local_tokens_per_expert.add_((route & image[:, None]).sum(0))
        return probs.to(logits.dtype), route

    def forward(self, hidden_states, padding_mask=None, image_mask=None):
        """Compute FP32 router logits with the inherited router projection."""
        return self.routing(self.gating(hidden_states), image_mask, padding_mask)


def multimodal_moe_forward(moe, hidden_states, image_mask, padding_mask=None):
    """Compose standard dispatch/expert/combine operations with explicit modality inputs.

    The mask is never placed on a module, so overlapping forwards cannot overwrite
    another microbatch's routing metadata.
    """
    shared = moe.shared_experts_compute(hidden_states)
    padding = None if padding_mask is None else padding_mask.transpose(0, 1)
    probs, route = moe.router(hidden_states, padding, image_mask.transpose(0, 1))
    hidden_states, probs = moe.preprocess(hidden_states, probs, route)
    dispatched, probs = moe.dispatch(hidden_states, probs)
    output, bias = moe.routed_experts_compute(dispatched, probs)
    output = moe.combine(output)
    return moe.postprocess(output, shared), bias
