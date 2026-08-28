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


def _gate_up_weight_arguments(
    shared_fc1: torch.Tensor, routed_fc1, intermediate_size: int
):
    """Adapt MCore-owned combined FC1 weights to MOK's gate/up API.

    Shared FC1 is a dense two-dimensional parameter, so gate and up are
    ordinary zero-copy row views. Routed FC1 may span multiple experts or
    descriptor-backed allocations; passing the same combined payload for gate
    and up preserves its storage metadata, and MOK applies the existing logical
    gate/up offsets when the two arguments alias.
    """
    expected_rows = 2 * intermediate_size
    if shared_fc1.ndim != 2 or shared_fc1.shape[0] != expected_rows:
        raise RuntimeError(
            "MOK shared FC1 shape mismatch: "
            f"got {tuple(shared_fc1.shape)}, expected [{expected_rows}, hidden_size]"
        )
    shared_gate = shared_fc1.narrow(0, 0, intermediate_size)
    shared_up = shared_fc1.narrow(0, intermediate_size, intermediate_size)
    return shared_gate, shared_up, routed_fc1, routed_fc1


def _gate_up_main_grad_arguments(main_grads, main_grad_storage_tables, intermediate_size: int):
    """Adapt canonical FC1/FC2 main-grad buffers to MOK's six-buffer order."""
    shared_fc1, routed_fc1, shared_fc2, routed_fc2 = main_grads
    shared_gate, shared_up, routed_gate, routed_up = _gate_up_weight_arguments(
        shared_fc1, routed_fc1, intermediate_size
    )
    split_main_grads = (
        shared_gate,
        routed_gate,
        shared_up,
        routed_up,
        shared_fc2,
        routed_fc2,
    )
    if main_grad_storage_tables is None:
        return split_main_grads, None
    routed_fc1_table, routed_fc2_table = main_grad_storage_tables
    return split_main_grads, (routed_fc1_table, routed_fc1_table, routed_fc2_table)


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
        prepared_fc1, prepared_fc2 = module.quantized_routed_weights()
        if module.use_mxfp8_weights and module.native_single_grouped_weights:
            # Single-weight MXFP8 forward consumes only rowwise data and scale.
            fc1_forward = prepared_fc1[:2]
            fc2_forward = prepared_fc2[:2]
        else:
            # Single-weight BF16 has no separate rowwise/columnwise physical tuple.
            # Non-single BF16/MXFP8 use SplitRoutedWeight objects consumed directly by MOK.
            fc1_forward = prepared_fc1
            fc2_forward = prepared_fc2
        shared_gate, shared_up, routed_gate, routed_up = _gate_up_weight_arguments(
            shared_fc1, fc1_forward, module.intermediate_size
        )
        output, forward_context = functional.forward(
            module.mok_config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_fc2,
            routed_gate,
            routed_up,
            fc2_forward,
            swiglu_limit=module.swiglu_limit,
        )

        ctx.module = module
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.quantized_weights = (prepared_fc1, prepared_fc2)
        ctx.save_for_backward(x, router_weights, *routed_parameters, shared_fc1, shared_fc2)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from mok import functional

        x, router_weights, *parameters = ctx.saved_tensors
        num_routed_parameters = len(ctx.module.autograd_routed_parameters)
        shared_fc1, shared_fc2 = parameters[num_routed_parameters:]
        prepared_fc1, prepared_fc2 = ctx.quantized_weights
        if ctx.module.use_mxfp8_weights and ctx.module.native_single_grouped_weights:
            backward_fc1 = prepared_fc1
            backward_fc2 = prepared_fc2[2:]
        else:
            backward_fc1 = prepared_fc1
            backward_fc2 = prepared_fc2
        shared_gate, shared_up, routed_gate, routed_up = _gate_up_weight_arguments(
            shared_fc1, backward_fc1, ctx.module.intermediate_size
        )
        main_grads, main_grad_storage_tables = ctx.module.main_grad_arguments()
        main_grads, main_grad_storage_tables = _gate_up_main_grad_arguments(
            main_grads, main_grad_storage_tables, ctx.module.intermediate_size
        )
        d_x, d_router_weights, *_ = functional.backward(
            ctx.module.mok_config,
            ctx.workspace,
            ctx.schedule,
            ctx.forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_fc2,
            routed_gate,
            routed_up,
            backward_fc2,
            swiglu_limit=ctx.module.swiglu_limit,
            main_grads=main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
        )

        routed_parameter_grads = ctx.module.finish_routed_weight_gradients()
        d_shared_fc1 = _finish_weight_gradient(ctx.module.shared_fc1_weight)
        d_shared_fc2 = _finish_weight_gradient(ctx.module.shared_fc2_weight)

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
