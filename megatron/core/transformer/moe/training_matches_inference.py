# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The training MoE forward of ``moe_training_matches_inference``.

The layer's output comes from a value pass through training's all-to-all dispatcher
with generation's FlashInfer expert kernel and its finalize arithmetic. The backward
comes from the layer's unchanged training forward, which the activation checkpoint
recomputes.
"""

from __future__ import annotations

import inspect
import weakref
from itertools import chain
from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import register_optimizer_step_post_hook

from megatron.core.inference.moe.flashinfer_mxfp8 import (
    flashinfer_routed_mxfp8_moe,
    prepare_routed_mxfp8_weights,
)
from megatron.core.inference.moe.flashinfer_nvfp4 import (
    flashinfer_routed_nvfp4_moe,
    prepare_routed_nvfp4_weights,
)
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.inference.quantization.utils import (
    _has_mxfp8_storage,
    _to_bf16,
    resolve_mxfp8_backend,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.moe.batch_invariant import unpermute_with_bf16_rank_partials
from megatron.core.transformer.utils import precision_aware_checkpoint
from megatron.core.typed_torch import apply_module

if TYPE_CHECKING:
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer


def training_matches_inference_forward(
    layer: MoELayer,
    training_forward: Callable,
    hidden_states: torch.Tensor,
    intermediate_tensors=None,
    padding_mask: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
):
    """Forward values from the value pass, gradients from the layer's training forward.

    Without grad, as in the log-prob forward, only the value pass runs. With grad, the
    activation checkpoint runs the value pass for the output, and its recompute runs
    ``training_forward``, which builds the backward.
    """
    if intermediate_tensors is not None:
        raise RuntimeError(
            "moe_training_matches_inference does not support partial MoE CUDA-graph capture; "
            "remove moe_router/moe_preprocess from cuda_graph_modules"
        )

    def forward(hidden_states, padding_mask):
        # The checkpoint's first call runs without grad, its recompute with grad.
        if torch.is_grad_enabled():
            return training_forward(hidden_states, None, padding_mask)
        return _value_pass(layer, hidden_states, padding_mask, input_ids)

    if torch.is_grad_enabled():
        return precision_aware_checkpoint(
            forward, layer.config, layer.tp_group, hidden_states, padding_mask
        )
    return forward(hidden_states, padding_mask)


def _value_pass(layer: MoELayer, hidden_states: torch.Tensor, padding_mask, input_ids):
    """The layer's training forward with generation's routes, expert kernel and finalize.

    Training's all-to-all dispatcher sends each route as its own row and the FlashInfer
    kernel computes every row. The batch-invariant unpermute then adds a token's rows on
    each EP rank as the kernel's finalize does in generation, and adds the ranks as
    generation's combine does.
    """
    shared_expert_output = None
    if not layer.run_shared_experts_after_router:
        shared_expert_output = layer.shared_experts_compute(hidden_states)
    probs, top_indices = _route(layer, hidden_states, input_ids)
    routing_probs = torch.zeros(
        probs.shape[0], layer.config.num_moe_experts, dtype=probs.dtype, device=probs.device
    ).scatter(1, top_indices, probs)
    routing_map = torch.zeros_like(routing_probs, dtype=torch.bool).scatter(1, top_indices, True)
    dispatch_input, routing_probs = layer.preprocess(
        hidden_states, routing_probs, routing_map, padding_mask
    )
    if layer.run_shared_experts_after_router:
        shared_expert_output = layer.shared_experts_compute(hidden_states)

    dispatcher = layer.token_dispatcher
    dispatched_input, routing_probs = layer.dispatch(dispatch_input, routing_probs)
    dispatched_input, tokens_per_expert, _ = dispatcher.dispatch_postprocess(
        dispatched_input, routing_probs
    )
    rows = expert_value_pass(layer.experts, dispatched_input, tokens_per_expert)
    rows = layer.combine(dispatcher.combine_preprocess(rows))
    # The routed MXFP8 kernel weighs routes by BF16 probabilities, per-token NVFP4 by FP32.
    if layer.experts._nvfp4_recipe is None:
        probs = probs.to(torch.bfloat16)
    # The inverse map lists a token's routes in expert order; reorder it to top-k order.
    route_rows = dispatcher.batch_invariant_inverse_permutation_mapping[0].gather(
        1, top_indices.argsort(dim=1).argsort(dim=1)
    )
    output = unpermute_with_bf16_rank_partials(
        rows,
        route_rows,
        top_indices,
        probs.float(),
        layer.config.num_moe_experts,
        dispatcher.ep_size,
    )
    return _postprocess(layer, output, shared_expert_output), None


def _route(layer: MoELayer, hidden_states: torch.Tensor, input_ids):
    """Route as generation does, returning [tokens, topk] probs and expert ids.

    The FlashInfer finalize sums each token's routes in the order listed, so the value
    pass needs generation's top-k order, which training's routing map does not keep.
    Generation's router path also leaves the training router's expert-bias token counts
    to the recompute. It takes [tokens, 1, hidden].
    """
    with InferenceMode.active():
        return layer.route(
            hidden_states.reshape(-1, 1, hidden_states.shape[-1]), input_ids=input_ids
        )


def _postprocess(layer: MoELayer, output: torch.Tensor, shared_expert_output):
    """``layer.postprocess`` on the unpermuted output, for the all-to-all dispatcher."""
    dispatcher = layer.token_dispatcher
    output = output.view(dispatcher.hidden_shape)
    if dispatcher.shared_experts is not None:
        output += dispatcher.shared_experts.get_output()
    dispatcher.probs = None
    if layer.config.moe_latent_size:
        if layer.config.moe_use_norm_before_up_proj:
            output = apply_module(layer.fc2_norm)(output)
        output, _ = layer.fc2_latent_proj(output)
    if shared_expert_output is not None:
        output = output + shared_expert_output
    return dispatcher.finalize_layer_output(output)


# The value pass's kernel weights per experts module, with the source versions they were
# built from, kept here rather than on the module. Weak keys, so a deleted model frees its
# entries.
_value_pass_weights: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_hooked_experts: weakref.WeakSet = weakref.WeakSet()
_optimizer_step_hook = None


def _drop_value_pass_weights(module, *_):
    _value_pass_weights.pop(module, None)


def release_inference_value_pass_weights(model: torch.nn.Module) -> None:
    """Free the value pass's cached kernel weights in ``model``.

    For a caller that needs the memory, for example before colocated generation. The
    next value pass rebuilds them from the parameters.
    """
    for module in model.modules():
        _drop_value_pass_weights(module)


def expert_value_pass(experts: InferenceGroupedMLP, hidden_states, tokens_per_expert):
    """Generation's FlashInfer kernel, on the training parameters, for each dispatched row.

    Each row is one route, sorted by local expert. The kernel gets it as a top-1 token
    routed to that expert with probability 1.0, so its finalize returns the expert's
    output row unchanged. The routed MXFP8 kernel has no tile config for a few rows per
    local expert, so its launches are padded with unrouted rows to more than 128 per
    local expert, where its tile is its largest; the row values do not depend on the
    tile.

    The kernel weights are built from the parameters and reused until an in-place write
    to them, an optimizer step, a ``state_dict()`` or checkpoint load, or
    ``release_inference_value_pass_weights``. TE writes into MXFP8 parameters, as in an
    optimizer step or a copy into a ``state_dict()`` tensor, bypass the version counter,
    so a process-wide optimizer step post-hook and state_dict and load_state_dict
    post-hooks on the experts, all registered here, drop the weights; the source
    tensors' versions catch other in-place writes. A reference-model swap that copies
    into ``state_dict()`` tensors, as NeMo-RL's does, calls ``state_dict()`` first.
    Neither the training pipeline nor the experts module keeps any value-pass state.
    """
    global _optimizer_step_hook
    versions = _source_versions(experts)
    cached = _value_pass_weights.get(experts)
    if cached is None or cached[0] != versions:
        if _optimizer_step_hook is None:
            _optimizer_step_hook = register_optimizer_step_post_hook(
                lambda *_: _value_pass_weights.clear()
            )
        if experts not in _hooked_experts:
            experts.register_load_state_dict_post_hook(_drop_value_pass_weights)
            experts.register_state_dict_post_hook(_drop_value_pass_weights)
            _hooked_experts.add(experts)
        weights = _build_value_pass_weights(experts)
        # Read again: finishing the parameter all-gather in the build can write them.
        cached = _value_pass_weights[experts] = (_source_versions(experts), weights)
    mxfp8_weights, nvfp4_weights = cached[1]

    num_rows = hidden_states.shape[0]
    num_local_experts = experts.num_local_experts
    local_expert_offset = experts.ep_group.rank() * num_local_experts
    launch_rows = num_rows
    if mxfp8_weights is not None:
        launch_rows = max(num_rows, 128 * num_local_experts + 1)
    if launch_rows == 0:
        return hidden_states
    routes = hidden_states.new_full((launch_rows, 1), -1, dtype=torch.int64)
    routes[:num_rows, 0] = torch.repeat_interleave(
        torch.arange(
            local_expert_offset,
            local_expert_offset + num_local_experts,
            device=hidden_states.device,
        ),
        torch.as_tensor(tokens_per_expert, device=hidden_states.device),
        output_size=num_rows,
    )
    probs = torch.ones_like(routes, dtype=torch.float32)
    if launch_rows > num_rows:
        hidden_states = torch.cat(
            [hidden_states, hidden_states.new_zeros(launch_rows - num_rows, hidden_states.shape[1])]
        )
    kwargs = dict(
        num_experts=num_local_experts * experts.ep_group.size(),
        local_expert_offset=local_expert_offset,
        activation_type=experts._flashinfer_activation_type.value,
    )
    if mxfp8_weights is not None:
        output = flashinfer_routed_mxfp8_moe(hidden_states, routes, probs, *mxfp8_weights, **kwargs)
    else:
        output = flashinfer_routed_nvfp4_moe(hidden_states, routes, probs, nvfp4_weights, **kwargs)
    return output[:num_rows]


def _expert_weights(experts: InferenceGroupedMLP):
    for linear_name, _ in type(experts)._EXPERT_WEIGHT_GROUPS:
        linear = getattr(experts, linear_name)
        for i in range(experts.num_local_experts):
            yield getattr(linear, f"weight{i}")


def _source_versions(experts: InferenceGroupedMLP) -> tuple:
    """The address and version of every tensor the kernel weights are built from.

    A TE MXFP8 parameter is represented by the FP8 data and scales it dequantizes from,
    since ``state_dict()`` hands out a new TE tensor over them rather than a view of the
    parameter.
    """
    versions = []
    for weight in _expert_weights(experts):
        if isinstance(weight, DTensor):
            raise RuntimeError(
                "moe_training_matches_inference does not support sharded (DTensor) expert "
                "parameters, as Megatron-FSDP and torch FSDP use; use Megatron DDP."
            )
        te = weight if hasattr(weight, "_rowwise_data") else weight.data
        if getattr(te, "_rowwise_data", None) is not None:
            tensors = (te._rowwise_data, te._rowwise_scale_inv)
        elif getattr(te, "_columnwise_data", None) is not None:
            tensors = (te._columnwise_data, te._columnwise_scale_inv)
        else:
            tensors = (weight,)
        versions.extend((tensor.data_ptr(), tensor._version) for tensor in tensors)
    return tuple(versions)


def _finish_param_gather(experts: InferenceGroupedMLP) -> None:
    """Run the expert linears' pre-forward hooks, which finish DDP's parameter all-gather.

    The value pass reads the weights without running linear_fc1 or linear_fc2. Only Megatron
    DDP's hooks are known to need nothing from that forward, so any other pre-forward hook
    on them is rejected rather than run out of context.
    """
    from megatron.core.distributed import distributed_data_parallel

    for submodule in chain(experts.linear_fc1.modules(), experts.linear_fc2.modules()):
        for hook_id, hook in list(submodule._forward_pre_hooks.items()):
            if (
                inspect.getmodule(hook) is not distributed_data_parallel
                or hook_id in submodule._forward_pre_hooks_with_kwargs
            ):
                raise RuntimeError(
                    "moe_training_matches_inference reads the expert weights without running "
                    "the expert linears, so their only supported pre-forward hooks are Megatron "
                    f"DDP's parameter all-gather hooks; {type(submodule).__name__} has {hook!r}."
                )
            hook(submodule, ())


def _build_value_pass_weights(experts: InferenceGroupedMLP):
    """The value pass's (MXFP8, NVFP4) kernel weights, one of them None, from the parameters."""
    _finish_param_gather(experts)
    weight_groups = type(experts)._EXPERT_WEIGHT_GROUPS
    mxfp8_weights = nvfp4_weights = None
    with torch.no_grad():
        if experts._nvfp4_recipe is not None:
            nvfp4_weights = prepare_routed_nvfp4_weights(
                *(
                    torch.stack(
                        [
                            getattr(getattr(experts, linear_name), f"weight{i}").detach()
                            for i in range(experts.num_local_experts)
                        ]
                    )
                    for linear_name, _ in weight_groups
                )
            )
        else:
            if not expert_params_use_te_mxfp8(experts):
                raise RuntimeError(
                    "moe_training_matches_inference runs the FlashInfer routed MXFP8 kernel "
                    "on TE MXFP8 expert parameters, but this MoE layer's are not: they are "
                    "BF16, or were already converted for generation, which needs its own "
                    "model instance. Select both expert projections of every MoE layer in "
                    "the TE MXFP8 precision recipe, or set inference_moe_nvfp4_recipe."
                )
            backend = resolve_mxfp8_backend(experts.inference_grouped_gemm_backend)
            mxfp8_weights = tuple(
                prepare_routed_mxfp8_weights(mxfp8_training_stack(experts, linear_name, backend))
                for linear_name, _ in weight_groups
            )
    return mxfp8_weights, nvfp4_weights


def expert_params_use_te_mxfp8(experts: InferenceGroupedMLP) -> bool:
    """Whether the expert parameters hold TE MXFP8 storage (training's MXFP8 form)."""
    return type(experts)._require_uniform_weight_format(
        (
            _has_mxfp8_storage(getattr(getattr(experts, linear_name), f"weight{expert_index}"))
            for linear_name, _ in type(experts)._EXPERT_WEIGHT_GROUPS
            for expert_index in range(experts.num_local_experts)
        ),
        "MXFP8",
    )


def mxfp8_training_stack(experts, linear_name: str, backend: str) -> MXFP8Tensor:
    """Requantize the expert parameters into generation's stacked MXFP8 layout.

    Same dequantize-then-``from_bf16`` route, per expert, as ``quantize_model_to_mxfp8``,
    so the bits match generation's.
    """
    linear = getattr(experts, linear_name)
    data = scale = dtype = None
    for i in range(experts.num_local_experts):
        bf16 = _to_bf16(getattr(linear, f"weight{i}"))
        quantized = MXFP8Tensor.from_bf16(bf16, backend=backend)
        if data is None:
            data = quantized.data.new_empty(experts.num_local_experts, *quantized.data.shape)
            scale = quantized.scale.new_empty(experts.num_local_experts, *quantized.scale.shape)
            dtype = bf16.dtype
        data[i].copy_(quantized.data)
        scale[i].copy_(quantized.scale)
    return MXFP8Tensor(data=data, scale=scale, dtype=dtype, backend=backend)
