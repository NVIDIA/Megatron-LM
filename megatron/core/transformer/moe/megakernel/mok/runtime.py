# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK functional runtime and its MCore autograd bridge."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, NamedTuple

import torch

from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    finish_weight_gradient as _finish_weight_gradient,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    main_grad_buffer as _main_grad_buffer,
)

if TYPE_CHECKING:
    from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel


def _gate_up_weight_arguments(shared_fc1: torch.Tensor, routed_fc1, intermediate_size: int):
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
    split_main_grads = (shared_gate, routed_gate, shared_up, routed_up, shared_fc2, routed_fc2)
    if main_grad_storage_tables is None:
        return split_main_grads, None
    routed_fc1_table, routed_fc2_table = main_grad_storage_tables
    return split_main_grads, (routed_fc1_table, routed_fc1_table, routed_fc2_table)


class _MoKWeightArguments(NamedTuple):
    """Non-owning references in the external MOK forward/backward ABI order.

    Routed arguments may be tensors, MXFP8 tuples, or split-weight descriptors.
    This transient tuple neither copies payloads nor registers/caches parameters.
    """

    shared_gate: torch.Tensor
    shared_up: torch.Tensor
    shared_down: torch.Tensor
    routed_gate: object
    routed_up: object
    routed_down: object

    @classmethod
    def from_combined(cls, shared_gate_up, shared_fc2, routed_fc1, routed_fc2):
        """Pair the shared FC1 gate/up views with the caller's combined routed FC1 view.

        ``shared_gate_up`` is ``MoKMegakernel.shared_fc1_gate_up()``, whose row views are cached
        so that MOK receives the same tensor objects on every call. Routed FC1 is passed as both
        gate and up; MOK applies the logical gate/up offsets when the two arguments alias.
        """
        shared_gate, shared_up = shared_gate_up
        return cls(shared_gate, shared_up, shared_fc2, routed_fc1, routed_fc1, routed_fc2)


# Keyword options of mok.functional.backward that this adapter sets, resolved once per installed
# backward function (older MOK builds accept fewer options).
_BACKWARD_OPTIONS: dict = {}


def _backward_options(backward) -> dict:
    """Return the ``mok.functional.backward`` keyword options to pass, if MOK supports them.

    ``zero_unrouted_router_grads=False``: RoutingMapToIndices.backward reads the router-weight
    gradient only at routed slots, so MOK need not zero the unrouted ones.
    """
    options = _BACKWARD_OPTIONS.get(backward)
    if options is None:
        try:
            parameters = inspect.signature(backward).parameters
        except (TypeError, ValueError):
            parameters = {}
        options = {}
        if "zero_unrouted_router_grads" in parameters:
            options["zero_unrouted_router_grads"] = False
        _BACKWARD_OPTIONS[backward] = options
    return options


class _MoKAutograd(torch.autograd.Function):
    """Autograd bridge from MCore parameters to MoK's functional API."""

    @staticmethod
    def forward(
        ctx,
        module: "MoKMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        shared_output_gate_weight: torch.Tensor | None,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Run MOK forward and retain the state needed by backward."""
        from mok import functional

        num_routed_parameters = len(module.autograd_routed_parameters)
        if len(parameters) != num_routed_parameters + 2:
            raise RuntimeError(
                "MOK autograd parameter count mismatch: "
                f"got {len(parameters)}, expected {num_routed_parameters + 2}"
            )
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
        fc1_weight_view, fc2_weight_view = module.quantized_routed_weights()
        if module.use_mxfp8_weights and module.native_single_grouped_weights:
            # Single-weight MXFP8 forward consumes only rowwise data and scale.
            fc1_forward = fc1_weight_view[:2]
            fc2_forward = fc2_weight_view[:2]
        else:
            # Single-weight BF16 has no separate rowwise/columnwise physical tuple.
            # Non-single BF16/MXFP8 use SplitRoutedWeight objects consumed directly by MOK.
            fc1_forward = fc1_weight_view
            fc2_forward = fc2_weight_view
        # Cached gate/up views of shared FC1 (shared_fc1 is module.shared_fc1_weight).
        weight_args = _MoKWeightArguments.from_combined(
            module.shared_fc1_gate_up(), shared_fc2, fc1_forward, fc2_forward
        )
        gate_kwargs = {}
        if shared_output_gate_weight is not None:
            gate_kwargs["shared_output_gate_weight"] = shared_output_gate_weight
        output, forward_context = functional.forward(
            module.mok_config,
            workspace,
            schedule,
            x,
            router_weights,
            *weight_args,
            swiglu_limit=module.swiglu_limit,
            **gate_kwargs,
        )

        ctx.module = module
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.routed_weight_views = (fc1_weight_view, fc2_weight_view)
        # Backward reads routed weights through ctx.routed_weight_views and their main grads
        # through ctx.module; the routed parameters are inputs only so autograd delivers a
        # gradient to each one's AccumulateGrad node, which fires the DDP hooks.
        ctx.save_for_backward(x, router_weights, shared_output_gate_weight, shared_fc1, shared_fc2)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Run MOK backward and return gradients for its autograd inputs."""
        from mok import functional

        # Shared FC1 stays saved for autograd's version check; MOK reads it through cached views.
        x, router_weights, shared_output_gate_weight, _, shared_fc2 = ctx.saved_tensors
        fc1_weight_view, fc2_weight_view = ctx.routed_weight_views
        if ctx.module.use_mxfp8_weights and ctx.module.native_single_grouped_weights:
            # FC1 backward may need rowwise weights for recompute and columnwise weights for
            # dgrad, while FC2 only needs columnwise weights for dgrad.
            backward_fc1 = fc1_weight_view
            backward_fc2 = fc2_weight_view[2:]
        else:
            backward_fc1 = fc1_weight_view
            backward_fc2 = fc2_weight_view
        weight_args = _MoKWeightArguments.from_combined(
            ctx.module.shared_fc1_gate_up(), shared_fc2, backward_fc1, backward_fc2
        )
        main_grads, main_grad_storage_tables = ctx.module.mok_main_grad_arguments()
        gate_kwargs = {}
        if shared_output_gate_weight is not None:
            gate_kwargs = {
                "shared_output_gate_weight": shared_output_gate_weight,
                "shared_output_gate_main_grad": _main_grad_buffer(
                    ctx.module.shared_output_gate_weight
                ),
            }
        d_x, d_router_weights, *_ = functional.backward(
            ctx.module.mok_config,
            ctx.workspace,
            ctx.schedule,
            ctx.forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            *weight_args,
            swiglu_limit=ctx.module.swiglu_limit,
            main_grads=main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
            **gate_kwargs,
            **_backward_options(functional.backward),
        )

        routed_parameter_grads = ctx.module.finish_routed_weight_gradients()
        d_shared_fc1 = _finish_weight_gradient(ctx.module.shared_fc1_weight)
        d_shared_fc2 = _finish_weight_gradient(ctx.module.shared_fc2_weight)
        d_output_gate = (
            _finish_weight_gradient(ctx.module.shared_output_gate_weight)
            if shared_output_gate_weight is not None
            else None
        )

        ctx.module = None
        ctx.workspace = None
        ctx.schedule = None
        ctx.forward_context = None
        ctx.routed_weight_views = None
        return (
            None,
            d_x,
            d_router_weights,
            None,
            d_output_gate,
            *routed_parameter_grads,
            d_shared_fc1,
            d_shared_fc2,
        )
