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


class DeepseekV4MoE(nn.Module):
    """Model-specific assembly over shared router, Experts, dispatcher, and shared MLP.

    Allowlist reason: this owns DS4 hash routing wiring, while expert compute stays shared.
    """

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
        self.gate = SigmoidTopKRouter(config, ps, compute_aux_loss=False)
        if self.is_hash_layer:
            self.gate.register_buffer(
                "tid2eid",
                torch.zeros(config.vocab_size, self.topk, dtype=torch.int64),
                persistent=True,
            )
        else:
            self.gate._non_persistent_buffers_set.discard("expert_bias")
        self.experts = Experts(config, ps)
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
        self.dispatcher = TokenDispatcher(
            config.n_routed_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
        )

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
        if self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return (weights * self.route_scale).to(dtype=x.dtype), indices

    def forward(self, x: torch.Tensor, *, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        if self.is_hash_layer and input_ids is not None:
            weights, indices = self._hash_route(x_flat, input_ids)
        else:
            weights, indices = self.gate(x_flat)
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
            out = out + self.shared_experts(x_flat)
        return out.view(shape)
