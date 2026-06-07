"""MoE router implementations: TopKRouter (softmax) and SigmoidTopKRouter.

Internals call the atomic free functions in Megatron-Core's
`megatron.core.transformer.moe.moe_utils` (plan `docs/moe_mc_wrap_plan.md`
D3/D4). The outer classes keep the flat-kwargs + `ParallelState` constructor
style of megatron.lite primitives — no `TransformerConfig`, no mpu globals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from megatron.core.transformer.moe.moe_utils import (  # pyright: ignore[reportMissingImports]
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
)

from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel import ParallelState


def _ordered_topk_from_routing_map(
    probs_dense: torch.Tensor,
    routing_map: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_ids = torch.arange(
        probs_dense.size(-1),
        device=probs_dense.device,
        dtype=torch.long,
    ).expand_as(routing_map)
    masked_ids = torch.where(
        routing_map,
        expert_ids,
        torch.full_like(expert_ids, probs_dense.size(-1)),
    )
    topk_indices = torch.sort(masked_ids, dim=-1).values[:, :topk]
    topk_scores = torch.gather(probs_dense, dim=-1, index=topk_indices)
    return topk_scores, topk_indices


class TopKRouter(nn.Module):
    """TopK gating with optional high-precision router logits/probabilities."""

    def __init__(
        self,
        config,
        ps: ParallelState,
        *,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = True,
        use_pre_softmax: bool = False,
        moe_router_fusion: bool = False,
        router_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if router_bias_rate > 0:
            raise NotImplementedError(
                "expert-bias EMA is not implemented in the primitive router; "
                "use load_balancing_type='none' or extend ParallelState."
            )
        self.topk = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.aux_loss_coeff = config.router_aux_loss_coef
        self.router_bias_rate = router_bias_rate
        self.compute_aux_loss = compute_aux_loss
        self.use_pre_softmax = use_pre_softmax
        self.moe_router_fusion = moe_router_fusion
        self.router_dtype = router_dtype

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

        self._aux_loss_group = ps.tp_group if ps.tp_size > 1 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_dtype = self.router_dtype or x.dtype
        logits = router_gating_linear(x, self.gate.weight, None, router_dtype)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        if self.moe_router_fusion:
            probs_dense, _ = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.use_pre_softmax,
                score_function="softmax",
                fused=True,
            )
            topk_scores, topk_indices = torch.topk(probs_dense, k=self.topk, dim=-1)
        else:
            probs_dense, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.use_pre_softmax,
                score_function="softmax",
                fused=False,
            )
            topk_scores, topk_indices = _ordered_topk_from_routing_map(
                probs_dense,
                routing_map,
                self.topk,
            )
        if self.router_dtype is None:
            topk_scores = topk_scores.to(x.dtype)

        if self.compute_aux_loss and self.training and torch.is_grad_enabled():
            routing_map, aux_scores = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                score_function="softmax",
                fused=self.moe_router_fusion,
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(
                    group=self._aux_loss_group
                )
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


class SigmoidTopKRouter(nn.Module):
    """Sigmoid-based TopK router for DeepSeek V3."""

    def __init__(
        self,
        config,
        ps: ParallelState,
        *,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = True,
        use_pre_softmax: bool = False,
        moe_router_fusion: bool = False,
    ):
        super().__init__()
        if router_bias_rate > 0:
            raise NotImplementedError(
                "expert-bias EMA is not implemented in the primitive router; "
                "use load_balancing_type='none' or extend ParallelState."
            )
        self.topk = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.aux_loss_coeff = config.aux_loss_alpha
        self.scaling_factor = config.routed_scaling_factor
        self.router_bias_rate = router_bias_rate
        self.compute_aux_loss = compute_aux_loss
        self.use_pre_softmax = use_pre_softmax
        self.moe_router_fusion = moe_router_fusion

        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )

        self._aux_loss_group = ps.tp_group if ps.tp_size > 1 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        probs_dense, routing_map = topk_routing_with_score_function(
            logits,
            self.topk,
            score_function="sigmoid",
            expert_bias=self.expert_bias.to(logits.dtype),
            scaling_factor=(self.scaling_factor or None),
            fused=self.moe_router_fusion,
        )
        topk_scores, topk_indices = _ordered_topk_from_routing_map(
            probs_dense,
            routing_map,
            self.topk,
        )
        topk_scores = topk_scores.to(logits.dtype)

        if self.compute_aux_loss and self.training and torch.is_grad_enabled():
            _, aux_scores = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                score_function="sigmoid",
                fused=self.moe_router_fusion,
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(
                    group=self._aux_loss_group
                )
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


__all__ = ["SigmoidTopKRouter", "TopKRouter"]
