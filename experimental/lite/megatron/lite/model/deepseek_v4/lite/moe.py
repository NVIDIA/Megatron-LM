# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.mlp import SwiGLUMLP
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.primitive.utils.moe import topk_routing_with_score_function


class DeepseekV4Router(SigmoidTopKRouter):
    """DS4 router whose observable slot order matches rollout's dsv4_topk."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x).view(-1, self.num_experts)
        routing_kwargs = {}
        if self.num_groups is not None and self.group_topk is not None:
            routing_kwargs = {
                "num_groups": self.num_groups,
                "group_topk": self.group_topk,
            }
        topk_scores, topk_indices = topk_routing_with_score_function(
            logits,
            self.topk,
            use_pre_softmax=self.use_pre_softmax,
            score_function=self.score_function,
            expert_bias=self.expert_bias.to(logits.dtype),
            scaling_factor=(self.scaling_factor or None),
            fused=False,
            dense_output=True,
            **routing_kwargs,
        )
        if self.router_replay is not None:
            selected = self.router_replay.select_indices(topk_indices)
            if selected is not topk_indices:
                scores = torch.sqrt(F.softplus(logits.float()))
                topk_scores = scores.gather(-1, selected.long())
                if self.topk > 1:
                    topk_scores = topk_scores / topk_scores.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-20)
                topk_scores = topk_scores * self.scaling_factor
                topk_indices = selected
        return topk_scores.to(logits.dtype), topk_indices


class DeepseekV4MoE(nn.Module):
    """Model-specific assembly over shared router, Experts, dispatcher, and shared MLP.

    Allowlist reason: this owns DS4 hash routing wiring, while expert compute stays shared.
    """

    dispatcher_cls = TokenDispatcher

    def __init__(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        *,
        layer_idx: int,
        use_deepep: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.is_hash_layer = layer_idx < config.num_hash_layers
        self.gate = DeepseekV4Router(config, ps, compute_aux_loss=False)
        if self.is_hash_layer:
            self.gate.register_buffer(
                "tid2eid",
                torch.zeros(config.vocab_size, self.topk, dtype=torch.int64),
                persistent=True,
            )
        else:
            self.gate._non_persistent_buffers_set.discard("expert_bias")
        self.experts = self._build_experts(config, ps)
        shared_intermediate = config.n_shared_experts * config.moe_intermediate_size
        self.shared_experts = (
            SwiGLUMLP(
                config.hidden_size,
                shared_intermediate,
                swiglu_limit=config.swiglu_limit,
            )
            if config.n_shared_experts > 0
            else None
        )
        self.dispatcher = self.dispatcher_cls(
            config.n_routed_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
        )

    def _build_experts(self, config: DeepseekV4Config, ps: ParallelState) -> nn.Module:
        return Experts(config, ps)

    def _hash_route(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate.gate(x).view(-1, self.gate.num_experts)
        if self.gate.score_function == "sqrtsoftplus":
            scores = F.softplus(logits.float()).sqrt()
        else:
            scores = logits.float().sigmoid()
        indices = self.gate.tid2eid[input_ids.reshape(-1).to(torch.int64)]
        weights = scores.gather(1, indices)
        # R3's rollout tensor contains one column for every DS4 MoE layer,
        # including the input-deterministic hash layers.  Consume/replay those
        # columns here so later learned-router layers retain the same global
        # layer index as vLLM.  Replay changes only indices; normalization and
        # route scaling below still use this actor's live scores.
        if self.gate.router_replay is not None:
            weights, indices = self.gate.router_replay.apply(scores, weights, indices)
        if self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return (weights * self.route_scale).to(dtype=x.dtype), indices

    def _route(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_hash_layer and input_ids is not None:
            return self._hash_route(x, input_ids)
        return self.gate(x)

    def _shared_expert_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(x)

    def forward(self, x: torch.Tensor, *, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        weights, indices = self._route(x_flat, input_ids)
        dispatched, tpe, permuted_probs = self.dispatcher.dispatch(x_flat, weights, indices)
        del weights, indices
        self.dispatcher.wait_dispatch_event()
        out = self.experts(
            dispatched,
            tpe,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        out = self.dispatcher.combine(out)
        if self.shared_experts is not None:
            out = out + self._shared_expert_forward(x_flat)
        return out.view(shape)
