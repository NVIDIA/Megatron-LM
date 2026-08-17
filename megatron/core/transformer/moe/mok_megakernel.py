# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal Mixture-of-Kittens adapter for MCore MoE training experiments.

This module intentionally keeps the integration narrow: MCore owns routing,
parameters, DDP, and the optimizer, while MoK replaces dispatch, routed/shared
expert computation, and combine. Checkpoint conversion and fused accumulation
into ``main_grad`` are deliberately out of scope for the first E2E study.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


def _copy_parameter_attributes(dst: nn.Parameter, src: torch.Tensor, *, allreduce: bool) -> None:
    """Copy the parameter metadata MCore uses for optimizer/DDP classification."""
    dst.allreduce = allreduce
    for name in (
        "sequence_parallel",
        "tensor_model_parallel",
        "partition_dim",
        "partition_stride",
        "shared",
    ):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _dequantize_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a logical TE/plain parameter as an ordinary BF16 tensor."""
    tensor = tensor.detach()
    dequantize = getattr(tensor, "dequantize", None)
    if callable(dequantize):
        tensor = dequantize()
    return tensor.to(dtype=torch.bfloat16)


def _indexed_grouped_weight(linear: nn.Module, index: int, num_experts: int) -> torch.Tensor:
    """Return one logical expert weight from either TE grouped layout."""
    if not getattr(linear, "single_grouped_weight", False):
        return getattr(linear, f"weight{index}")

    weight = linear.weight
    split_quantized = getattr(weight, "split_into_quantized_tensors", None)
    if callable(split_quantized):
        return split_quantized()[index]
    if weight.ndim >= 3 and weight.shape[0] == num_experts:
        return weight[index]
    if weight.shape[0] % num_experts != 0:
        raise RuntimeError(
            f"Cannot split grouped weight with shape {tuple(weight.shape)} "
            f"into {num_experts} experts"
        )
    return weight.narrow(0, index * (weight.shape[0] // num_experts), weight.shape[0] // num_experts)


def _new_bf16_parameter(
    shape: Iterable[int], reference: torch.Tensor, *, allreduce: bool, zero: bool = False
) -> nn.Parameter:
    data = torch.empty(tuple(shape), dtype=torch.bfloat16, device=reference.device)
    if zero:
        data.zero_()
    param = nn.Parameter(data)
    _copy_parameter_attributes(param, reference, allreduce=allreduce)
    return param


class _MoKAutograd(torch.autograd.Function):
    """Autograd bridge from MCore parameters to MoK's functional API."""

    @staticmethod
    def forward(
        ctx,
        module: "MoKMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        routed_gate: torch.Tensor,
        routed_up: torch.Tensor,
        routed_down: torch.Tensor,
        shared_gate: torch.Tensor,
        shared_up: torch.Tensor,
        shared_down: torch.Tensor,
    ) -> torch.Tensor:
        from mok import functional

        workspace = functional.get_workspace(
            module.mok_config,
            module.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=top_experts.shape[1],
        )
        schedule = functional.build_schedule(
            workspace,
            module.mok_config,
            top_experts,
            num_local_experts=module.num_local_experts,
        )
        gate_q, up_q, down_q = module.quantized_routed_weights()
        gate_forward = gate_q[:2] if isinstance(gate_q, tuple) else gate_q
        up_forward = up_q[:2] if isinstance(up_q, tuple) else up_q
        down_forward = down_q[:2] if isinstance(down_q, tuple) else down_q
        output, forward_context = functional.forward(
            module.mok_config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_down,
            gate_forward,
            up_forward,
            down_forward,
            swiglu_limit=module.swiglu_limit,
        )

        ctx.module = module
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.quantized_weights = (gate_q, up_q, down_q)
        ctx.save_for_backward(
            x,
            router_weights,
            routed_gate,
            routed_up,
            routed_down,
            shared_gate,
            shared_up,
            shared_down,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from mok import functional

        (
            x,
            router_weights,
            routed_gate,
            routed_up,
            routed_down,
            shared_gate,
            shared_up,
            shared_down,
        ) = ctx.saved_tensors
        gate_q, up_q, down_q = ctx.quantized_weights
        backward_gate = gate_q
        backward_up = up_q
        backward_down = down_q[2:] if isinstance(down_q, tuple) else down_q
        (
            d_x,
            d_router_weights,
            d_routed_gate,
            d_routed_up,
            d_routed_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        ) = functional.backward(
            ctx.module.mok_config,
            ctx.workspace,
            ctx.schedule,
            ctx.forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_down,
            backward_gate,
            backward_up,
            backward_down,
            swiglu_limit=ctx.module.swiglu_limit,
        )

        ctx.module = None
        ctx.workspace = None
        ctx.schedule = None
        ctx.forward_context = None
        ctx.quantized_weights = None
        return (
            None,
            d_x,
            d_router_weights,
            None,
            d_routed_gate,
            d_routed_up,
            d_routed_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        )


class MoKMegakernel(nn.Module):
    """Own MCore trainable weights in the layouts consumed by the MoK kernel."""

    def __init__(
        self,
        config,
        ep_group,
        routed_experts: nn.Module,
        shared_experts: nn.Module,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        try:
            from mok.functional import MoKConfig
        except ImportError as exc:
            raise ImportError(
                "--use-mok-megakernel requires the latest mixture-of-kittens package "
                "on PYTHONPATH"
            ) from exc

        if config.moe_mlp_glu_interleave_size is not None:
            raise ValueError("MoK weight import requires non-interleaved MCore routed FC1 weights")
        if config.moe_shared_expert_glu_interleave_size is not None:
            raise ValueError("MoK weight import requires non-interleaved shared FC1 weights")
        if config.moe_shared_expert_gate:
            raise ValueError("MoK does not support MCore's optional shared-expert output gate")

        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size
        self.shared_intermediate_size = config.moe_shared_expert_intermediate_size
        self.topk = config.moe_router_topk
        self.swiglu_limit = config.activation_func_clamp_value
        self.use_mxfp8_weights = config.mok_use_mxfp8_weights
        self.mok_config = MoKConfig(
            fwd_num_comm_sms=config.mok_fwd_num_comm_sms,
            bwd_num_comm_sms=config.mok_bwd_num_comm_sms,
            minibatch_size=config.mok_minibatch_size,
            macrobatch_size=config.mok_macrobatch_size,
            schedule_capacity_multiplier=config.mok_schedule_capacity_multiplier,
            all_gather_top_experts_chunk_bytes=config.mok_all_gather_top_experts_chunk_bytes,
            scale_router_before_fc2=config.mok_scale_router_before_fc2,
        )

        self._import_routed_weights(routed_experts)
        self._import_shared_weights(shared_experts)

        # MegatronModule.set_is_first_microbatch discovers this attribute and resets it
        # once per optimizer iteration, matching TE's weight-cache lifecycle.
        self.is_first_microbatch = True
        self._quantized_cache: tuple[Any, Any, Any] | None = None
        self._quantized_versions: tuple[int, int, int] | None = None

    @torch.no_grad()
    def _import_routed_weights(self, experts: nn.Module) -> None:
        fc1 = experts.linear_fc1
        fc2 = experts.linear_fc2
        fc1_ref = _indexed_grouped_weight(fc1, 0, self.num_local_experts)
        fc2_ref = _indexed_grouped_weight(fc2, 0, self.num_local_experts)
        i, h, e = self.intermediate_size, self.hidden_size, self.num_local_experts

        self.routed_gate_weight = _new_bf16_parameter((e, i, h), fc1_ref, allreduce=False)
        self.routed_up_weight = _new_bf16_parameter((e, i, h), fc1_ref, allreduce=False)
        self.routed_down_weight = _new_bf16_parameter((e, h, i), fc2_ref, allreduce=False)

        for expert_idx in range(e):
            source_fc1 = _dequantize_bf16(
                _indexed_grouped_weight(fc1, expert_idx, self.num_local_experts)
            ).reshape(2 * i, h)
            source_fc2 = _dequantize_bf16(
                _indexed_grouped_weight(fc2, expert_idx, self.num_local_experts)
            ).reshape(h, i)
            self.routed_gate_weight[expert_idx].copy_(source_fc1[:i])
            self.routed_up_weight[expert_idx].copy_(source_fc1[i:])
            self.routed_down_weight[expert_idx].copy_(source_fc2)

    @torch.no_grad()
    def _import_shared_weights(self, shared: nn.Module) -> None:
        fc1_ref = shared.linear_fc1.weight
        fc2_ref = shared.linear_fc2.weight
        routed_i = self.intermediate_size
        shared_i = self.shared_intermediate_size
        h = self.hidden_size

        # Upstream MoK currently has one intermediate-size template parameter for
        # both routed and shared experts. Zero-padding is mathematically inert and
        # keeps the DSv4-Pro shared MLP (2048) equivalent inside a routed-I=3072
        # kernel. The extra shared compute is reported as a known POC overhead.
        self.shared_gate_weight = _new_bf16_parameter(
            (routed_i, h), fc1_ref, allreduce=True, zero=True
        )
        self.shared_up_weight = _new_bf16_parameter(
            (routed_i, h), fc1_ref, allreduce=True, zero=True
        )
        self.shared_down_weight = _new_bf16_parameter(
            (h, routed_i), fc2_ref, allreduce=True, zero=True
        )
        source_fc1 = _dequantize_bf16(fc1_ref).reshape(2 * shared_i, h)
        source_fc2 = _dequantize_bf16(fc2_ref).reshape(h, shared_i)
        self.shared_gate_weight[:shared_i].copy_(source_fc1[:shared_i])
        self.shared_up_weight[:shared_i].copy_(source_fc1[shared_i:])
        self.shared_down_weight[:, :shared_i].copy_(source_fc2)

    @torch.no_grad()
    def quantized_routed_weights(self):
        """Refresh normal+transposed MXFP8 copies once per optimizer iteration."""
        if not self.use_mxfp8_weights:
            return (
                self.routed_gate_weight,
                self.routed_up_weight,
                self.routed_down_weight,
            )

        from mok.ops import mxfp8_quantize

        versions = (
            self.routed_gate_weight._version,
            self.routed_up_weight._version,
            self.routed_down_weight._version,
        )
        if (
            self._quantized_cache is None
            or self.is_first_microbatch
            or versions != self._quantized_versions
        ):
            self._quantized_cache = (
                mxfp8_quantize(self.routed_gate_weight, True, True),
                mxfp8_quantize(self.routed_up_weight, True, True),
                mxfp8_quantize(self.routed_down_weight, True, True),
            )
            self._quantized_versions = versions
            self.is_first_microbatch = False
        return self._quantized_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        del routing_map  # Router side effects/losses are already attached to probs.
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        router_weights, top_experts = torch.topk(
            probs.reshape(x.shape[0], -1), self.topk, dim=-1, sorted=False
        )
        router_weights = router_weights.to(dtype=torch.float32).contiguous()
        top_experts = top_experts.to(dtype=torch.int64).contiguous()

        output = _MoKAutograd.apply(
            self,
            x,
            router_weights,
            top_experts,
            self.routed_gate_weight,
            self.routed_up_weight,
            self.routed_down_weight,
            self.shared_gate_weight,
            self.shared_up_weight,
            self.shared_down_weight,
        )
        return output.view(original_shape)
