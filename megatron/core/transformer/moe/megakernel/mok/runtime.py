# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK runtime (CuTe MoK's API or the mok functional layer) and its MCore autograd bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from megatron.core.transformer.moe.megakernel.mok.package import mok_modules
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    finish_weight_gradient as _finish_weight_gradient,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    main_grad_buffer as _main_grad_buffer,
)

if TYPE_CHECKING:
    from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel


def _gate_up_weight_arguments(shared_fc1: torch.Tensor | None, routed_fc1, intermediate_size: int):
    """Adapt MCore-owned combined FC1 weights to MOK's gate/up API.

    Shared FC1 is a dense two-dimensional parameter, so gate and up are
    ordinary zero-copy row views. Routed FC1 may span multiple experts or
    descriptor-backed allocations; passing the same combined payload for gate
    and up preserves its storage metadata, and MOK applies the existing logical
    gate/up offsets when the two arguments alias.
    """
    if shared_fc1 is None:
        return None, None, routed_fc1, routed_fc1
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


# CuTe MoK plans by layer configuration: a plan owns its symmetric workspace, and every layer of one
# configuration shares it. A plan outgrown by a larger token count is retired, not freed.
_CUTEMOK_PLANS: dict = {}
_RETIRED_CUTEMOK_PLANS: list = []


def _cutemok_plan(module: "MoKMegakernel", api, x: torch.Tensor):
    """Return the CuTe MoK plan of ``module``'s configuration for ``x.shape[0]`` tokens per rank."""
    main_grad = getattr(module.routed_fc1_parameters[0], "main_grad", None)
    variant = dict(
        hidden_size=module.hidden_size,
        intermediate_size=module.intermediate_size,
        num_local_experts=module.num_local_experts,
        topk=module.topk,
        swiglu_clamped=module.swiglu_limit is not None,
        situ=module.situ_beta is not None,
        has_shared=module.shared_fc1_weight is not None,
        split_routed_weights=not module.native_single_grouped_weights,
        shared_output_gate=module.shared_output_gate_weight is not None,
        backward=main_grad is not None,
        main_grad_dtype=None if main_grad is None else main_grad.dtype,
        routed_precision="mxfp8" if module.use_mxfp8_weights else "bf16",
        native_columnwise=module.use_mxfp8_weights,
    )
    key = (id(module.ep_group), x.device, repr(module.mok_config), tuple(sorted(variant.items())))
    tokens = x.shape[0]
    cached = _CUTEMOK_PLANS.get(key)
    if cached is not None and cached[0] >= tokens:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CuTe MoK prepares its plan outside CUDA graph capture; run an eager step first"
        )
    if cached is not None:
        _RETIRED_CUTEMOK_PLANS.append(cached[1])
    plan = api.cutemok_prepare(
        module.mok_config, module.ep_group, device=x.device, max_tokens_per_rank=tokens, **variant
    )
    _CUTEMOK_PLANS[key] = (tokens, plan)
    return plan


def clear_cutemok_plans() -> None:
    """Drop the cached CuTe MoK plans once the calls using them have finished (on every rank)."""
    torch.cuda.synchronize()
    _CUTEMOK_PLANS.clear()
    _RETIRED_CUTEMOK_PLANS.clear()


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
    def from_combined(cls, shared_fc1, shared_fc2, routed_fc1, routed_fc2, intermediate_size):
        """Split only shared FC1 while retaining the caller's routed weight views."""
        shared_gate, shared_up, routed_gate, routed_up = _gate_up_weight_arguments(
            shared_fc1, routed_fc1, intermediate_size
        )
        return cls(shared_gate, shared_up, shared_fc2, routed_gate, routed_up, routed_fc2)


class _MoKAutograd(torch.autograd.Function):
    """Autograd bridge from MCore parameters to CuTe MoK's API or MoK's functional layer."""

    @staticmethod
    def forward(
        ctx,
        module: "MoKMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        shared_output_gate_weight: torch.Tensor | None,
        *parameters: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run MOK forward and retain the state needed by backward."""
        api = module.cutemok
        num_routed_parameters = len(module.autograd_routed_parameters)
        if len(parameters) != num_routed_parameters + 2:
            raise RuntimeError(
                "MOK autograd parameter count mismatch: "
                f"got {len(parameters)}, expected {num_routed_parameters + 2}"
            )
        routed_parameters = parameters[:num_routed_parameters]
        shared_fc1, shared_fc2 = parameters[num_routed_parameters:]
        if api is None:
            functional, _ = mok_modules()
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
        weight_args = _MoKWeightArguments.from_combined(
            shared_fc1, shared_fc2, fc1_forward, fc2_forward, module.intermediate_size
        )
        forward_kwargs = {}
        if shared_output_gate_weight is not None:
            forward_kwargs["shared_output_gate_weight"] = shared_output_gate_weight
        if module.situ_beta is not None:
            forward_kwargs["situ_beta"] = module.situ_beta
        if api is None:
            output, forward_context = functional.forward(
                module.mok_config,
                workspace,
                schedule,
                x,
                router_weights,
                *weight_args,
                swiglu_limit=module.swiglu_limit,
                **forward_kwargs,
            )
        else:
            # The plan takes the workspace's role; the forward also builds the schedule.
            workspace = _cutemok_plan(module, api, x)
            output, schedule, forward_context = api.cutemok_forward(
                x,
                top_experts,
                router_weights,
                *weight_args,
                plan=workspace,
                config=module.mok_config,
                swiglu_limit=module.swiglu_limit,
                **forward_kwargs,
            )

        ctx.module = module
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.routed_weight_views = (fc1_weight_view, fc2_weight_view)
        ctx.save_for_backward(
            x, router_weights, shared_output_gate_weight, *routed_parameters, shared_fc1, shared_fc2
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Run MOK backward and return gradients for its autograd inputs."""
        x, router_weights, shared_output_gate_weight, *parameters = ctx.saved_tensors
        num_routed_parameters = len(ctx.module.autograd_routed_parameters)
        shared_fc1, shared_fc2 = parameters[num_routed_parameters:]
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
            shared_fc1, shared_fc2, backward_fc1, backward_fc2, ctx.module.intermediate_size
        )
        main_grads, main_grad_storage_tables = ctx.module.main_grad_arguments()
        main_grads, main_grad_storage_tables = _gate_up_main_grad_arguments(
            main_grads, main_grad_storage_tables, ctx.module.intermediate_size
        )
        gate_kwargs = {}
        if shared_output_gate_weight is not None:
            gate_kwargs = {
                "shared_output_gate_weight": shared_output_gate_weight,
                "shared_output_gate_main_grad": _main_grad_buffer(
                    ctx.module.shared_output_gate_weight
                ),
            }
        api = ctx.module.cutemok
        if api is None:
            functional, _ = mok_modules()
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
            )
        else:
            d_x, d_router_weights, *_ = api.cutemok_backward(
                grad_output.contiguous(),
                x,
                router_weights,
                *weight_args,
                ctx.schedule,
                ctx.forward_context,
                plan=ctx.workspace,
                config=ctx.module.mok_config,
                swiglu_limit=ctx.module.swiglu_limit,
                main_grads=main_grads,
                main_grad_storage_tables=main_grad_storage_tables,
                **gate_kwargs,
            )

        routed_parameter_grads = ctx.module.finish_routed_weight_gradients()
        d_shared_fc1 = (
            _finish_weight_gradient(ctx.module.shared_fc1_weight)
            if shared_fc1 is not None
            else None
        )
        d_shared_fc2 = (
            _finish_weight_gradient(ctx.module.shared_fc2_weight)
            if shared_fc2 is not None
            else None
        )
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
