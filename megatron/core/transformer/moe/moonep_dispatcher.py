# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token dispatcher backed by MoonEP's perfectly balanced expert-parallel kernels."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moonep_manager import (
    MoonEPLayerBuffers,
    MoonEPManager,
    get_or_create_moonep_manager,
)
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


class _MoonEPDispatch(torch.autograd.Function):
    """Scatter tokens to expert-grouped slots and prefetch the duplicated experts.

    The backward is MoonEP's combine: it K-sums each token's dispatched gradient
    copies back to token-major and, on the same pass, gathers the permuted probs
    gradient. Duplicated experts' wgrad is reduced to its home rank here because
    this node runs after the expert backward.
    """

    @staticmethod
    def forward(
        ctx, hidden_sh, probs_sk, topk_sk, tokens_per_expert, manager, layer_buffers, plan_holder
    ):
        """Run MoonEP planning plus dispatch, then fill the local prefetch slots."""
        buffer = manager.ensure_buffer(hidden_sh.shape[0])
        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
            hidden_sh.contiguous(), probs_sk, topk_sk, tokens_per_expert
        )
        manager.prefetch(layer_buffers, plan)
        plan_holder.append(plan)
        ctx.manager = manager
        ctx.layer_buffers = layer_buffers
        ctx.plan = plan
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    @staticmethod
    def backward(ctx, grad_hidden_nvsh, grad_route_weights_nvs, _grad_cu_seqlens):
        """Combine token gradients back and reduce duplicated experts' weight gradients."""
        manager = ctx.manager
        if grad_route_weights_nvs is None:
            grad_route_weights_nvs = torch.zeros_like(grad_hidden_nvsh[:, 0], dtype=torch.float32)
        grad_hidden_sh, grad_probs_sk, _ = manager.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden_nvsh.contiguous(),
            route_weights_nvs=grad_route_weights_nvs.contiguous().float(),
        )
        manager.reduce_grad(ctx.layer_buffers, ctx.plan)
        return grad_hidden_sh, grad_probs_sk, None, None, None, None, None


class _MoonEPCombine(torch.autograd.Function):
    """K-sum expert outputs back to token-major.

    The backward re-dispatches the output gradient with the saved plan. It first
    restores this microbatch's prefetched weights, because the prefetch pool is
    shared across layers and has since been overwritten by other layers' forwards.
    """

    @staticmethod
    def forward(ctx, expert_output_nvsh, manager, layer_buffers, plan):
        """Gather expert outputs from the NVL buffer into token-major order."""
        output_sh, _, _ = manager.buffer.combine(
            plan=plan, hidden_nvsh=expert_output_nvsh.contiguous()
        )
        ctx.manager = manager
        ctx.layer_buffers = layer_buffers
        ctx.plan = plan
        return output_sh

    @staticmethod
    def backward(ctx, grad_output_sh):
        """Restore replica weights, then scatter the output gradient back to VM group order."""
        manager = ctx.manager
        manager.prefetch(ctx.layer_buffers, ctx.plan)
        manager.reset_grad_accumulators(ctx.layer_buffers)
        grad_expert_output_nvsh, _, _, _ = manager.buffer.dispatch(
            grad_output_sh.contiguous(), plan=ctx.plan
        )
        return grad_expert_output_nvsh, None, None, None


class MoEMoonEPTokenDispatcher(MoETokenDispatcher):
    """Dispatch tokens with MoonEP so every rank computes exactly S*K tokens.

    MoonEP plans a small set of redundant experts from the current router output,
    prefetches their weights over NVLink, and writes tokens straight into their
    expert-grouped destination slots. The local expert axis therefore holds this
    rank's own experts followed by the prefetch slots.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        layer_number: Optional[int] = None,
    ) -> None:
        """
        Initialize the MoonEP token dispatcher.

        Args:
            num_local_experts (int): Number of physical experts on this device.
            local_expert_indices (List[int]): Global indices of this device's master experts.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
            layer_number (int, optional): One-based MoE layer number, used to key MoonEP's
                per-layer symmetric weight and gradient storage.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        if self.tp_size != 1:
            raise ValueError("MoonEP currently requires expert_tensor_parallel_size=1.")
        if layer_number is None:
            raise ValueError("MoonEP MoE layers require a stable layer_number.")

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        self.layer_number = layer_number
        self.manager: MoonEPManager = get_or_create_moonep_manager(config, self.ep_group)
        self.layer_buffers: MoonEPLayerBuffers = self.manager.register_layer(layer_number)
        self.num_experts = config.num_moe_experts

        self.hidden_shape: Optional[torch.Size] = None
        self.plan = None
        self.cu_seqlens: Optional[torch.Tensor] = None
        self.num_dispatched_tokens: Optional[int] = None
        self.num_padded_slots: Optional[int] = None

    def reset_transient_forward_state(self) -> None:
        """Release the per-forward plan and offsets; the backward reads autograd-saved state."""
        self.plan = None
        self.cu_seqlens = None

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Flatten tokens and convert the multihot routing map into MoonEP's topk format."""
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        num_tokens = hidden_states.shape[0]

        probs = probs.reshape(num_tokens, self.num_experts)
        self.topk_probs, topk_indices = torch.topk(probs, self.config.moe_router_topk, dim=-1)
        self.topk_indices = topk_indices.to(torch.int32).contiguous()
        # Count from the indices actually handed to MoonEP so the planner and the
        # dispatch agree even when the router emits zero-probability slots.
        self.router_tokens_per_expert = (
            torch.bincount(topk_indices.flatten(), minlength=self.num_experts)
            .to(torch.int32)
            .contiguous()
        )

        self.manager.stage_master_weights(self.layer_buffers)
        return hidden_states, self.topk_probs

    def token_dispatch(self, hidden_states: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """Run MoonEP planning, dispatch and weight prefetch."""
        plan_holder: List = []
        dispatched, route_weights_nvs, cu_seqlens = _MoonEPDispatch.apply(
            hidden_states,
            self.topk_probs.float().contiguous(),
            self.topk_indices,
            self.router_tokens_per_expert,
            self.manager,
            self.layer_buffers,
            plan_holder,
        )
        self.plan = plan_holder[0]
        self.cu_seqlens = cu_seqlens
        return dispatched, route_weights_nvs

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Slice the padded NVL buffer down to the rows this rank's group GEMM consumes."""
        tokens_per_expert = self.manager.local_tokens_per_expert(self.cu_seqlens)
        counts = tokens_per_expert.tolist()
        total = sum(counts)
        self.num_padded_slots = hidden_states.shape[0]
        self.num_dispatched_tokens = total
        return hidden_states[:total], tokens_per_expert, probs[:total]

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """Pad expert outputs back to MoonEP's static [NvS, H] slot count."""
        padded = hidden_states.new_zeros(self.num_padded_slots, hidden_states.shape[-1])
        padded[: self.num_dispatched_tokens] = hidden_states
        return padded

    def token_combine(self, hidden_states: torch.Tensor):
        """K-sum expert outputs back to token-major order."""
        return _MoonEPCombine.apply(hidden_states, self.manager, self.layer_buffers, self.plan)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """Restore the original hidden-state shape."""
        return hidden_states.view(self.hidden_shape)
