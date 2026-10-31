# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The training MoE forward of ``moe_training_matches_inference``.

The layer's output comes from its generation forward, which runs generation's router,
inference token dispatcher and FlashInfer expert kernel. The backward comes from the
layer's unchanged training forward, which the activation checkpoint recomputes.
"""

from __future__ import annotations

import weakref
from itertools import chain
from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from megatron.core import utils
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
from megatron.core.transformer.moe.token_dispatcher_inference import (
    NCCLAllGatherDispatcher,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.utils import precision_aware_checkpoint

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
    """Forward values from the layer's generation forward, gradients from its training forward.

    Without grad, as in the log-prob forward, only the generation forward runs. With grad,
    the activation checkpoint runs the generation forward for the output, and its
    recompute runs ``training_forward``, which builds the backward.
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
        return _generation_forward(layer, hidden_states, input_ids)

    if torch.is_grad_enabled():
        return precision_aware_checkpoint(
            forward, layer.config, layer.tp_group, hidden_states, padding_mask
        )
    return forward(hidden_states, padding_mask)


def _generation_forward(layer: MoELayer, hidden_states: torch.Tensor, input_ids):
    """The layer's generation forward, on training's token counts."""
    try:
        with InferenceMode.active():
            layer.select_token_dispatcher()
            dispatcher = layer.token_dispatcher
            # Normally allocated by the inference context.
            if NCCLAllGatherDispatcher._valid_tokens_tensor is None:
                NCCLAllGatherDispatcher.allocate_buffers()
            chunked = _uses_nvls_buffers(dispatcher)
            if chunked:
                _allocate_nvls_buffers(layer, dispatcher)

            shared_expert_output = None
            if not layer.run_shared_experts_after_router:
                shared_expert_output = layer.shared_experts_compute(hidden_states)
            probs, routing_map = _route(layer, hidden_states, input_ids)
            dispatch_input, probs = layer.preprocess(hidden_states, probs, routing_map)
            if layer.run_shared_experts_after_router:
                shared_expert_output = layer.shared_experts_compute(hidden_states)

            if chunked:
                output = _chunked_dispatch_and_compute(layer, dispatch_input, probs)
            else:
                dispatched_input, probs = layer.dispatch(dispatch_input, probs)
                output = layer.combine(_routed_experts_compute(layer, dispatched_input, probs))
            return layer.postprocess(output, shared_expert_output), None
    finally:
        # Outside InferenceMode, this restores the training dispatcher for the recompute.
        layer.select_token_dispatcher()


def _route(layer: MoELayer, hidden_states: torch.Tensor, input_ids):
    """Route as generation does, returning [tokens, topk] probs and expert ids.

    The FlashInfer kernel sums each token's routes in the order listed, so the value pass
    must list them in generation's top-k order to match it bitwise. Generation's router
    takes [tokens, 1, hidden].
    """
    return layer.route(hidden_states.reshape(-1, 1, hidden_states.shape[-1]), input_ids=input_ids)


def _uses_nvls_buffers(dispatcher) -> bool:
    """Whether the dispatcher gathers through the NVLS buffers, which need EP > 1."""
    return (
        isinstance(dispatcher, NVLSAllGatherVDispatcher)
        and utils.get_pg_size(dispatcher.tp_ep_group) > 1
    )


def _allocate_nvls_buffers(layer: MoELayer, dispatcher: NVLSAllGatherVDispatcher):
    """Allocate the NVLS symmetric buffers as a generation context would.

    Their size is the row count of the FlashInfer launch. Without a generation context in
    this process they are sized for the default `max_tokens`; a trainer whose generation
    sets `max_tokens` allocates them first, sized the same way.
    """
    tp_ep_group = dispatcher.tp_ep_group
    if NVLSAllGatherVDispatcher._symm_rsv is not None:
        rows = NVLSAllGatherVDispatcher._symm_rsv["tensor"].shape[0]
        per_rank = NVLSAllGatherVDispatcher._per_rank_worst_case_token_count
        if rows != per_rank * utils.get_pg_size(tp_ep_group):
            raise RuntimeError(
                "moe_training_matches_inference requires the generation expert-parallel size"
            )
        return
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    NVLSAllGatherVDispatcher.allocate_buffers(
        per_rank_worst_case_token_count=DynamicInferenceContext.round_up_tokens(
            DynamicInferenceContext.DEFAULT_MAX_TOKENS
        )
        // utils.get_pg_size(layer.tp_group),
        topk=layer.config.moe_router_topk,
        hidden_size=layer.config.moe_latent_size or layer.config.hidden_size,
        tp_ep_group=tp_ep_group,
    )


def _chunked_dispatch_and_compute(layer: MoELayer, hidden_states, probs):
    """Dispatch, expert compute and combine in launches that fit the NVLS buffers.

    Every rank of the dispatcher's group runs the same number of launches, since each one
    is collective. Every launch refreshes the dispatch metadata, which generation
    refreshes only on its first MoE layer, and none applies generation's padding mask:
    every training row is real. Each launch hands the kernel only the gathered rows that
    can hold tokens rather than the whole buffer; the rows past the real ones are routed
    nowhere and are never combined. Both kernels are batch invariant in the row count, so
    this matches generation's full-buffer launches.

    The routed MXFP8 kernel sizes its token tile from rows * top_k / local experts and has
    no config for small tiles at some shapes, so MXFP8 launches keep enough rows for its
    largest tile, the one generation's full-buffer launches use.
    """
    dispatcher = layer.token_dispatcher
    capacity = NVLSAllGatherVDispatcher._per_rank_worst_case_token_count
    group_size = utils.get_pg_size(dispatcher.tp_ep_group)
    min_rows = 0
    if layer.experts._nvfp4_recipe is None:
        # Above 128 routes per local expert the kernel's tile is its largest, 256.
        min_rows = 128 * layer.num_local_experts // layer.config.moe_router_topk + 1
    routing_map = dispatcher.routing_map
    num_tokens = hidden_states.shape[0]
    max_tokens = torch.tensor([num_tokens], device=hidden_states.device, dtype=torch.int64)
    torch.distributed.all_reduce(
        max_tokens, op=torch.distributed.ReduceOp.MAX, group=dispatcher.tp_ep_group
    )
    real_token_count = NVLSAllGatherVDispatcher._real_token_count_tensor
    NVLSAllGatherVDispatcher._real_token_count_tensor = None
    outputs = []
    max_tokens = int(max_tokens.item())
    try:
        for start in range(0, max_tokens, capacity):
            chunk = slice(min(start, num_tokens), min(start + capacity, num_tokens))
            dispatcher.routing_map = routing_map[chunk]
            dispatcher._local_tokens = chunk.stop - chunk.start
            if not dispatcher._runs_metadata_sync:
                dispatcher.update_metadata(dispatcher._local_tokens)
            dispatched_input, chunk_probs = layer.dispatch(hidden_states[chunk], probs[chunk])
            rows = group_size * min(capacity, max_tokens - start)
            rows = min(max(rows, min_rows), dispatched_input.shape[0])
            dispatched_input, chunk_probs = dispatched_input[:rows], chunk_probs[:rows]
            dispatcher.routing_map = dispatcher.routing_map[:rows]
            outputs.append(
                layer.combine(_routed_experts_compute(layer, dispatched_input, chunk_probs))
            )
    finally:
        NVLSAllGatherVDispatcher._real_token_count_tensor = real_token_count
        dispatcher._local_tokens = num_tokens
    return torch.cat(outputs)


def _routed_experts_compute(layer: MoELayer, hidden_states, probs):
    """``MoELayer.routed_experts_compute`` on generation's dispatcher, with the experts'
    generation forward replaced by ``expert_value_pass``.

    Generation's expert weights are built from parameters it owns, which training's are
    not, so the value pass runs the FlashInfer kernel on weights built from the training
    parameters.
    """
    dispatcher = layer.token_dispatcher
    hidden_states, _, probs = dispatcher.dispatch_postprocess(hidden_states, probs)
    output, _ = expert_value_pass(layer.experts, hidden_states, probs, dispatcher.routing_map)
    return dispatcher.combine_preprocess(output)


# The value pass's kernel weights per experts module, kept here rather than on the
# module. Weak keys, so a deleted model frees its entries.
_value_pass_weights: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_hooked_on_load: weakref.WeakSet = weakref.WeakSet()
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


def expert_value_pass(experts: InferenceGroupedMLP, hidden_states, probs, routing_map):
    """The experts' value pass: generation's FlashInfer kernel on the training parameters.

    The kernel weights are built from the parameters on the first value pass after an
    optimizer step, and reused until the next step, a checkpoint load, or
    ``release_inference_value_pass_weights``. A process-wide optimizer step post-hook and
    a load_state_dict post-hook on the experts, both registered here, drop them, so
    neither the training pipeline nor the experts module keeps any value-pass state.
    """
    global _optimizer_step_hook
    assert routing_map is not None, "routing_map is required for the training forward."
    weights = _value_pass_weights.get(experts)
    if weights is None:
        if _optimizer_step_hook is None:
            _optimizer_step_hook = register_optimizer_step_post_hook(
                lambda *_: _value_pass_weights.clear()
            )
        if experts not in _hooked_on_load:
            experts.register_load_state_dict_post_hook(_drop_value_pass_weights)
            _hooked_on_load.add(experts)
        weights = _value_pass_weights[experts] = _build_value_pass_weights(experts)
    mxfp8_weights, nvfp4_weights = weights
    kwargs = dict(
        num_experts=experts.num_local_experts * experts.ep_group.size(),
        local_expert_offset=experts.ep_group.rank() * experts.num_local_experts,
        activation_type=experts._flashinfer_activation_type.value,
        out=NVLSAllGatherVDispatcher._get_rsv_tensor() if experts._nvls_dispatcher else None,
    )
    if mxfp8_weights is not None:
        output = flashinfer_routed_mxfp8_moe(
            hidden_states, routing_map, probs, *mxfp8_weights, **kwargs
        )
    else:
        output = flashinfer_routed_nvfp4_moe(
            hidden_states, routing_map, probs, nvfp4_weights, **kwargs
        )
    return output, None


def _build_value_pass_weights(experts: InferenceGroupedMLP):
    """The value pass's (MXFP8, NVFP4) kernel weights, one of them None, from the parameters."""
    # The weights are read without running linear_fc1 or linear_fc2, so their pre-forward
    # hooks, which finish DDP's overlapped parameter all-gather, run here.
    for submodule in chain(experts.linear_fc1.modules(), experts.linear_fc2.modules()):
        for hook in submodule._forward_pre_hooks.values():
            hook(submodule, None)
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
