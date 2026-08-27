# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK functional runtime and its MCore autograd bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    finish_weight_gradient as _finish_weight_gradient,
)

if TYPE_CHECKING:
    from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel


class _MoKAutograd(torch.autograd.Function):
    """Autograd bridge from MCore parameters to MoK's functional API."""

    @staticmethod
    def forward(
        ctx,
        module: "MoKMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        from mok import functional

        num_routed_parameters = len(module.autograd_routed_parameters)
        if len(parameters) != num_routed_parameters + 2:
            raise RuntimeError(
                "MOK autograd parameter count mismatch: "
                f"got {len(parameters)}, expected {num_routed_parameters + 2}"
            )
        routed_parameters = parameters[:num_routed_parameters]
        shared_fc1, shared_fc2 = parameters[num_routed_parameters:]
        shared_gate, shared_up, shared_down = module.shared_weight_views(shared_fc1, shared_fc2)

        workspace = functional.get_workspace(
            module.mok_config,
            module.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=top_experts.shape[1],
        )
        schedule = functional.build_schedule(
            workspace, module.mok_config, top_experts, num_local_experts=module.num_local_experts
        )
        prepared_gate, prepared_up, prepared_down = module.quantized_routed_weights()
        if module.use_mxfp8_weights and module.native_single_grouped_weights:
            gate_forward = prepared_gate[:2]
            up_forward = prepared_up[:2]
            down_forward = prepared_down[:2]
        else:
            # BF16 single-weight and all non-single representations are already
            # directly consumable by MOK; only single-weight MXFP8 needs tuple slicing.
            gate_forward = prepared_gate
            up_forward = prepared_up
            down_forward = prepared_down
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
        ctx.quantized_weights = (prepared_gate, prepared_up, prepared_down)
        ctx.save_for_backward(x, router_weights, *routed_parameters, shared_fc1, shared_fc2)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from mok import functional

        x, router_weights, *parameters = ctx.saved_tensors
        num_routed_parameters = len(ctx.module.autograd_routed_parameters)
        shared_fc1, shared_fc2 = parameters[num_routed_parameters:]
        shared_gate, shared_up, shared_down = ctx.module.shared_weight_views(
            shared_fc1, shared_fc2
        )
        prepared_gate, prepared_up, prepared_down = ctx.quantized_weights
        if ctx.module.use_mxfp8_weights and ctx.module.native_single_grouped_weights:
            backward_gate = prepared_gate
            backward_up = prepared_up
            backward_down = prepared_down[2:]
        else:
            backward_gate = prepared_gate
            backward_up = prepared_up
            backward_down = prepared_down
        direct_wgrad_accumulation = ctx.module.fuse_wgrad_accumulation
        main_grads = None
        main_grad_storage_tables = None
        if direct_wgrad_accumulation:
            main_grads, main_grad_storage_tables = ctx.module.main_grad_arguments()
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
            main_grads=main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
        )

        if ctx.module.fuse_wgrad_accumulation:
            routed_parameter_grads = ctx.module.finish_routed_weight_gradients()
            d_shared_fc1 = _finish_weight_gradient(ctx.module.shared_fc1_weight)
            d_shared_fc2 = _finish_weight_gradient(ctx.module.shared_fc2_weight)
        else:
            # Materialized routed gradients are only supported by the original
            # dense/single-grouped interface.
            routed_parameter_grads = (d_routed_gate, d_routed_down)
            d_shared_fc1 = torch.cat((d_shared_gate, d_shared_up), dim=0)
            d_shared_fc2 = d_shared_down

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
            *routed_parameter_grads,
            d_shared_fc1,
            d_shared_fc2,
        )
