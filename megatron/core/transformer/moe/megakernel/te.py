"""NCCL EP dispatch, grouped experts, and combine in one TE operation sequence."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed import ProcessGroup

from megatron.core.extensions.transformer_engine import (
    TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT,
    _get_fp8_autocast_for_quant_recipe,
    _get_fp8_autocast_recipe_for_quant_params,
)
from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def routing_map_to_te_inputs(
    probs: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact authoritative boolean routes, retaining zero-weight selected experts."""
    if routing_map.dtype != torch.bool or routing_map.shape != probs.shape:
        raise ValueError("TE MoE requires a boolean routing map with the same shape as probs")
    num_experts = probs.shape[-1]
    expert_indices = torch.arange(num_experts, device=routing_map.device, dtype=torch.int32)
    candidates = expert_indices.expand_as(routing_map).masked_fill(~routing_map, num_experts)
    indices = candidates.topk(topk, dim=-1, largest=False).values
    valid = indices < num_experts
    weights = probs.gather(-1, indices.clamp_max(num_experts - 1).long())
    return weights.float().masked_fill(~valid, 0).contiguous(), indices.masked_fill(~valid, -1)


class TEMegakernel(MegakernelBackend):
    """Compose TE operations while native MCore experts own all trainable parameters."""

    def __init__(
        self,
        *,
        config: TransformerConfig,
        ep_group: ProcessGroup,
        routed_experts: nn.Module,
        shared_experts: nn.Module | None,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        try:
            from transformer_engine.pytorch import ep, ops

            required = (ops.MoeDispatch, ops.MoeCombine, ep.EpConfig, ep.EpBuffer)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "moe_megakernel_backend='te' requires TE with NCCL EP, MoeDispatch, "
                "MoeCombine, and operation-fuser channel support"
            ) from exc
        del required
        from megatron.core.transformer.moe.experts import TEGroupedMLP

        if (
            not isinstance(routed_experts, TEGroupedMLP)
            or not routed_experts._is_fused_impl_supported()
        ):
            raise ValueError("The TE MoE backend requires op-fuser-compatible TEGroupedMLP experts")
        if ep_group.size() < 2:
            raise ValueError("The TE MoE backend requires at least two expert-parallel ranks")
        self.config = config
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self._native_experts = (routed_experts, shared_experts)
        self._sequence = None
        self._ep_config = None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Native expert modules own the canonical checkpoint entries."""
        return {}

    def _initialize_ep(self, num_tokens: int) -> None:
        """Size the NCCL EP context once and verify its explicit process-group owner."""
        from transformer_engine.common.recipe import MXFP8BlockScaling
        from transformer_engine.pytorch import ep

        if self._ep_config is not None:
            if not ep.is_ep_bootstrapped() or ep.get_ep_group() is not self.ep_group:
                raise RuntimeError("The TE MoE NCCL EP context was finalized or replaced")
            return
        options = self.config.moe_megakernel_backend_config or {}
        settings = getattr(ep, "_BOOTSTRAP_SETTINGS", None)
        if "max_tokens_per_rank" in options:
            max_tokens = options["max_tokens_per_rank"]
        elif settings is not None:
            max_tokens = settings["max_tokens_per_rank"]
        else:
            bound = torch.tensor(num_tokens, device="cuda", dtype=torch.int64)
            torch.distributed.all_reduce(
                bound, op=torch.distributed.ReduceOp.MAX, group=self.ep_group
            )
            max_tokens = int(bound.item())
        max_tokens = max(64, max_tokens + -max_tokens % 64)
        alignment = TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT
        capacity = None
        if self.config.moe_expert_rank_capacity_factor is not None:
            capacity = int(
                max_tokens
                * self.config.moe_router_topk
                * self.config.moe_expert_rank_capacity_factor
            )
            capacity += -capacity % alignment
        if not ep.is_ep_bootstrapped():
            ep.ep_bootstrap(
                self.ep_group,
                num_experts=self.config.num_moe_experts,
                max_tokens_per_rank=max_tokens,
                hidden_dim=self.config.hidden_size,
                num_topk=self.config.moe_router_topk,
                recv_capacity_per_rank=capacity,
                max_num_sms=self.config.moe_flex_dispatcher_num_sms or 0,
                zero_copy=False,
                drop_on_overflow=False,
            )
        if ep.get_ep_group() is not self.ep_group or ep.get_ep_drop_on_overflow():
            raise RuntimeError("The TE MoE backend requires its own non-dropping NCCL EP context")
        settings = getattr(ep, "_BOOTSTRAP_SETTINGS", None)
        expected = {
            "top_k": self.config.moe_router_topk,
            "hidden_dim": self.config.hidden_size,
            "num_local_experts": self.num_local_experts,
            "max_tokens_per_rank": max_tokens,
            "recv_capacity_per_rank": capacity,
            "zero_copy": False,
        }
        if settings is None or any(settings.get(name) != value for name, value in expected.items()):
            raise RuntimeError("TE MoE layers must use matching process-wide NCCL EP settings")
        self._ep_config = ep.EpConfig(
            top_k=self.config.moe_router_topk,
            hidden_dim=self.config.hidden_size,
            num_local_experts=self.num_local_experts,
            max_tokens_per_rank=max_tokens,
            recv_capacity_per_rank=capacity,
            ep_group=self.ep_group,
            alignment=alignment,
            dispatch_fwd_quant_recipe=(
                MXFP8BlockScaling() if self.config.moe_dispatch_fwd_dtype == "mxfp8" else None
            ),
            combine_bwd_quant_recipe=(
                MXFP8BlockScaling() if self.config.moe_combine_bwd_dtype == "mxfp8" else None
            ),
        )

    def _make_sequence(self, buffer):
        """Wrap native expert operations in dispatch/combine with internal metadata channels."""
        from transformer_engine.pytorch import ops

        experts, _ = self._native_experts
        fc1, activation, fc2 = experts._make_fused_ops()
        fc1.split_alignment = self._ep_config.alignment
        fc2.split_alignment = self._ep_config.alignment
        dispatch = ops.MoeDispatch(self._ep_config, buffer)
        combine = ops.MoeCombine(self._ep_config, buffer)
        dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=False)
        dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=False)
        fc1.set_extra_input_channel(0, "tokens_per_expert")
        activation.set_extra_input_channel(0, "routing_weights")
        fc2.set_extra_input_channel(0, "tokens_per_expert")
        sequence = ops.Sequential(dispatch, fc1, activation, fc2, combine)
        sequence.register_forward_pre_hook(experts._make_fused_impl_pre_forward_hook())
        return sequence

    def forward(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        """Run NCCL EP and grouped MLP, retaining per-forward routing state for backward."""
        from transformer_engine.pytorch import ep

        if hidden_states.dtype != torch.bfloat16:
            raise TypeError("The TE MoE backend requires BF16 hidden states")
        shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, shape[-1]).contiguous()
        self._initialize_ep(tokens.shape[0])
        config = self._ep_config
        if tokens.shape[0] > config.max_tokens_per_rank:
            raise ValueError(
                "TE MoE input exceeds max_tokens_per_rank; set "
                "moe_megakernel_backend_config['max_tokens_per_rank'] to the per-rank maximum"
            )
        buffer = ep.EpBuffer(
            top_k=config.top_k,
            max_tokens_per_rank=config.max_tokens_per_rank,
            hidden_dim=config.hidden_dim,
            num_local_experts=config.num_local_experts,
            recv_capacity_per_rank=config.recv_capacity_per_rank,
            alignment=config.alignment,
            device=tokens.device,
            dispatch_fwd_quant_recipe=config.dispatch_fwd_quant_recipe,
            combine_bwd_quant_recipe=config.combine_bwd_quant_recipe,
        )
        if self._sequence is None:
            self._sequence = (self._make_sequence(buffer),)
        (sequence,) = self._sequence
        sequence[0].buffer = buffer
        sequence[-1].buffer = buffer
        sequence.train(self.training)
        weights, indices = routing_map_to_te_inputs(
            probs.reshape(-1, self.config.num_moe_experts),
            routing_map.reshape(-1, self.config.num_moe_experts),
            config.top_k,
        )
        experts, shared_experts = self._native_experts
        fc1_recipe = _get_fp8_autocast_recipe_for_quant_params(
            experts.linear_fc1.te_quant_params, experts.linear_fc1.training
        )
        fc2_recipe = _get_fp8_autocast_recipe_for_quant_params(
            experts.linear_fc2.te_quant_params, experts.linear_fc2.training
        )
        if fc1_recipe != fc2_recipe:
            raise ValueError("The TE MoE backend requires matching FC1 and FC2 precision overrides")
        context = (
            nullcontext() if fc1_recipe is None else _get_fp8_autocast_for_quant_recipe(fc1_recipe)
        )
        with context:
            output = sequence(tokens, indices, weights, indices).view(shape)
        if shared_experts is not None:
            output = output + shared_experts(hidden_states)
        return output
