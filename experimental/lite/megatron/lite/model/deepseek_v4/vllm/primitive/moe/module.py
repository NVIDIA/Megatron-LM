from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.moe import DeepseekV4MoE as LiteDeepseekV4MoE
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import (
    block_fp8_linear,
    gate_linear,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import fixed_route_vjp
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.grouped import (
    VLLMGroupedMoEWithBF16Backward,
)
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
    DeploymentBlockFP8Adapter,
)


def _kernel_topk_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights if weights.dtype == torch.float32 else weights.float()


def _learned_route(logits, bias, scale):
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    return dsv4_topk(logits, bias, torch.int64, scale)


def _batch_invariant_gate_logits(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    if hidden_states.shape[0] <= 16:
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
            is_available,
            ll_bf16_gemm,
        )

        if is_available():
            return ll_bf16_gemm(hidden_states, weight)
    return torch.mm(hidden_states, weight.T, out_dtype=torch.float32)


def _hash_route(logits, token_ids, tid2eid, *, topk, renormalize, scale):
    from vllm._custom_ops import topk_hash_softplus_sqrt

    weights = logits.new_empty((logits.shape[0], topk))
    ids = torch.empty_like(weights, dtype=torch.int64)
    token_expert = torch.empty_like(ids, dtype=torch.int32)
    topk_hash_softplus_sqrt(
        weights,
        ids,
        token_expert,
        logits,
        renormalize,
        scale,
        None,
        token_ids,
        tid2eid,
        None,
    )
    return weights, ids


class _VLLMVisibleExperts(Experts):
    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
        tokens_per_expert_list: list[int] | None = None,
    ) -> torch.Tensor:
        if tokens_per_expert_list is not None:
            tokens_per_expert = torch.tensor(
                tokens_per_expert_list,
                device=tokens_per_expert.device,
                dtype=tokens_per_expert.dtype,
            )
        w13 = tuple(
            getattr(self.fc1, f"weight{i}") for i in range(self.num_local_experts)
        )
        w2 = tuple(
            getattr(self.fc2, f"weight{i}") for i in range(self.num_local_experts)
        )
        return VLLMGroupedMoEWithBF16Backward.apply(
            hidden_states,
            tokens_per_expert,
            0.0,
            *w13,
            *w2,
        )


class DeepseekV4MoE(LiteDeepseekV4MoE):

    dispatcher_cls = VLLMAlignedNormalDeepEPDispatcher

    def __init__(
        self,
        config: DeepseekV4Config,
        ps=None,
        *,
        layer_idx: int,
        cache_deployment_weights: bool = False,
    ):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            require_batch_invariant_quant_kernel,
        )

        # ``deepseek_v4.vllm`` is an alignment implementation, not a generic
        # MoE backend.  Validate its required numerical kernel while building
        # the model so a missing or incompatible library can never turn into a
        # later, layout-dependent fallback.
        require_batch_invariant_quant_kernel()
        ps = ps or ParallelState()
        super().__init__(
            config,
            ps,
            layer_idx=layer_idx,
            use_deepep=True,
        )
        self.config = config
        self.ps = ps
        self.shared_gate_up_fp8 = DeploymentBlockFP8Adapter(
            cache_weight=cache_deployment_weights
        )
        self.shared_down_fp8 = DeploymentBlockFP8Adapter(
            cache_weight=cache_deployment_weights
        )

    def clear_deployment_weight_cache(self) -> None:
        self.shared_gate_up_fp8.clear_cache()
        self.shared_down_fp8.clear_cache()

    def _build_experts(self, config: DeepseekV4Config, ps: ParallelState) -> nn.Module:
        return _VLLMVisibleExperts(config, ps)

    def _shared_expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = block_fp8_linear(
            self.shared_gate_up_fp8,
            hidden_states,
            self.shared_experts.gate_up.weight,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        return block_fp8_linear(
            self.shared_down_fp8,
            F.silu(gate) * up,
            self.shared_experts.down.weight,
        )

    def _replay_route(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        replay = self.gate.router_replay
        if replay is None:
            return weights, ids
        selected = replay.select_indices(ids)
        if selected is ids:
            return weights, ids
        dense = torch.sqrt(F.softplus(logits.float()))
        weights = dense.gather(-1, selected.long())
        if self.config.norm_topk_prob and selected.size(-1) > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.config.routed_scaling_factor
        return weights.to(dtype=logits.dtype), selected

    def _route(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_hash_layer and input_ids is None:
            raise NotImplementedError("hash MoE requires explicit input_ids")
        logits = gate_linear(
            lambda value: _batch_invariant_gate_logits(
                value, self.gate.gate.weight
            ),
            hidden_states,
            self.gate.gate.weight,
        ).float().contiguous()
        if self.is_hash_layer:
            token_ids = input_ids.reshape(-1).to(dtype=torch.int32)
            tid2eid = self.gate.tid2eid.to(dtype=torch.int32).contiguous()

            def hash_route(value):
                weights, ids = _hash_route(
                    value,
                    token_ids,
                    tid2eid,
                    topk=self.config.num_experts_per_tok,
                    renormalize=self.config.norm_topk_prob,
                    scale=self.config.routed_scaling_factor,
                )
                return self._replay_route(value, weights, ids)

            weights, ids = fixed_route_vjp(
                hash_route,
                logits,
                renormalize=self.config.norm_topk_prob,
                route_scale=self.config.routed_scaling_factor,
            )
        else:
            correction_bias = self.gate.expert_bias.float().contiguous()

            def learned_route(value):
                weights, ids = _learned_route(
                    value, correction_bias, self.config.routed_scaling_factor
                )
                return self._replay_route(value, weights, ids)

            weights, ids = fixed_route_vjp(
                learned_route,
                logits,
                renormalize=True,
                route_scale=self.config.routed_scaling_factor,
            )
        return _kernel_topk_weights(weights), ids.to(dtype=torch.int64)
