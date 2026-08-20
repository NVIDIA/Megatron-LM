from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.moe import DeepseekV4MoE as LiteDeepseekV4MoE
from megatron.lite.model.deepseek_v4.vllm.runtime_metadata import MoEKernelMetadata
from megatron.lite.model.deepseek_v4.vllm.primitive import (
    block_fp8_linear,
    fixed_route_vjp,
    gate_linear as training_gate_linear,
)
from megatron.lite.primitive.alignment.vllm_grouped_moe import (
    VLLMGroupedMoEWithBF16Backward,
)
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    DeploymentBlockFP8Adapter,
)


def _kernel_topk_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights if weights.dtype == torch.float32 else weights.float()


def _learned_route(logits, bias, scale):
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    return dsv4_topk(logits, bias, torch.int64, scale)


def _gate_output(gate, hidden_states):
    result = gate(hidden_states)
    return result[0] if isinstance(result, tuple) else result


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
        if permuted_probs is None:
            permuted_probs = hidden_states.new_zeros(
                hidden_states.shape[0], dtype=torch.float32
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
            permuted_probs,
            0.0,
            *w13,
            *w2,
        )


class DeepseekV4MoE(LiteDeepseekV4MoE):

    def __init__(
        self,
        config: DeepseekV4Config,
        ps=None,
        *,
        layer_idx: int,
        use_deepep: bool = False,
    ):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            is_batch_invariant_quant_kernel_enabled,
        )

        # ``deepseek_v4.vllm`` is an alignment implementation, not a generic
        # MoE backend.  Validate its required numerical kernel while building
        # the model so a missing or incompatible library can never turn into a
        # later, layout-dependent fallback.
        if not is_batch_invariant_quant_kernel_enabled():
            raise RuntimeError(
                "DeepSeek V4 vLLM requires "
                "VLLM_BATCH_INVARIANT_KERNEL_LIB; refusing a non-BI MoE fallback"
            )
        ps = ps or ParallelState()
        super().__init__(
            config,
            ps,
            layer_idx=layer_idx,
            use_deepep=use_deepep,
            deepep_align_to_low_latency=True,
        )
        self.config = config
        self.ps = ps
        self.use_deepep = use_deepep
        self.shared_gate_up_fp8 = DeploymentBlockFP8Adapter(cache_weight=True)
        self.shared_down_fp8 = DeploymentBlockFP8Adapter(cache_weight=True)

    def clear_deployment_weight_cache(self) -> None:
        self.shared_gate_up_fp8.clear_cache()
        self.shared_down_fp8.clear_cache()

    def _build_experts(self, config: DeepseekV4Config, ps: ParallelState) -> nn.Module:
        return _VLLMVisibleExperts(config, ps)

    def _visible_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        metadata: MoEKernelMetadata | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("MoE requires flat [tokens, hidden]")
        if self.is_hash_layer and input_ids is None:
            raise NotImplementedError("hash MoE requires explicit input_ids")
        if metadata is None or metadata.gate_linear is None:
            raise NotImplementedError("MoE requires an explicit vLLM GateLinear")
        gate_linear = metadata.gate_linear
        if isinstance(gate_linear, nn.Module) and hasattr(gate_linear, "weight"):
            # Bind the caller-constructed vLLM GateLinear to this model's BF16
            # master rather than retaining a second persistent parameter.
            gate_linear._parameters["weight"] = self.gate.gate.weight
        logits = training_gate_linear(
            lambda value: _gate_output(gate_linear, value),
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

            topk_weights, topk_ids = fixed_route_vjp(
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

            topk_weights, topk_ids = fixed_route_vjp(
                learned_route,
                logits,
                renormalize=True,
                route_scale=self.config.routed_scaling_factor,
            )

        if not self.use_deepep:
            raise NotImplementedError(
                "MoE requires normal DeepEP transport aligned to low-latency semantics"
            )
        dispatched, tokens_per_expert, permuted_probs = self.dispatcher.dispatch(
            hidden_states,
            _kernel_topk_weights(topk_weights),
            topk_ids.to(dtype=torch.int64),
        )
        self.dispatcher.wait_dispatch_event()
        output = self.experts(
            dispatched,
            tokens_per_expert,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        output = self.dispatcher.combine(output)
        if self.shared_experts is not None:
            output = output + self._visible_shared_experts(hidden_states)
        return output
