# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer callables for HybridModel's EP-overlap (combined 1F1B) schedule plan.

``build_hybrid_stack_callables`` splits a group of physical decoder layers, such as
``[M*E]`` or a single unbracketed layer, into the schedule plan's slots:

* pre-dispatch compute: the Mamba, attention, MLA or GDN layers that precede the
  terminal MLP/MoE layer, then the MoE layer's pre-MLP norm, shared experts, router
  and dispatch preprocessing;
* token dispatch;
* the dense MLP or the routed experts;
* token combine and the MoE residual add.

``_MoEBackwardDWWrapper`` gives each slot the delayed MoE weight-gradient work it
owns.
"""

from contextlib import nullcontext
from functools import partial

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.common.model_chunk_schedule_plan import (
    LayerScheduleGroup,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.common.utils import PostProcessNode, PreProcessNode
from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.pipeline_parallel.utils import (
    ScheduleNode,
    StageDispatchBwdGrad,
    get_comm_stream,
)
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.transformer_layer import make_viewless_tensor

_ATTENTION_CONFIG_TYPES = (
    layer_utils.AttentionLayerConfig,
    layer_utils.DSALayerConfig,
    layer_utils.MLALayerConfig,
    layer_utils.GDNLayerConfig,
    layer_utils.KDALayerConfig,
)


class _MoEBackwardDWWrapper:
    """Run delayed MoE wgrad and expose only the parameters owned by its slot.

    Shared experts and the latent down projection run in pre-dispatch; routed
    experts and the latent up projection run later. Their gradient-accumulation
    hooks must run after their respective wgrad, even if a previous microbatch
    has already populated ``param.grad``.
    """

    def __init__(self, mlp, routed_experts: bool):
        self.dw_callable = partial(
            mlp.backward_dw, routed_experts=routed_experts, shared_experts=not routed_experts
        )
        self.wgrad_on_comm_stream = False
        self.submodules = []
        if routed_experts:
            self.submodules.append(mlp.experts)
        elif mlp.use_shared_expert and not mlp.shared_expert_overlap:
            self.submodules.append(mlp.shared_experts)
        if mlp.config.moe_latent_size and mlp.config.overlap_moe_expert_parallel_comm:
            self.submodules.append(mlp.fc2_latent_proj if routed_experts else mlp.fc1_latent_proj)
            self.wgrad_on_comm_stream = routed_experts

    def backward_dw(self):
        """Run this slot's wgrad after its autograd backward."""
        self.dw_callable()
        if self.wgrad_on_comm_stream:
            # MoELayer runs the latent up-projection wgrad on the communication
            # stream. Its accumulation hooks run on this node's compute stream.
            torch.cuda.current_stream().wait_stream(get_comm_stream())
        self.dw_callable = None

    def parameters(self):
        """Keep parameter owners alive until the node collects post-wgrad hooks.

        ``TransformerLayerNode.backward_dw`` enumerates these parameters after
        calling ``backward_dw``, then releases the wrapper with the slot state.
        """
        for module in self.submodules:
            yield from module.parameters()


def _get_inner_quant_context(layer):
    config = layer.config
    if config.fp8 and config.fp8_recipe != Fp8Recipe.delayed:
        return get_fp8_context(config, layer.layer_number - 1)
    if config.fp4:
        return get_fp4_context(config, layer.layer_number - 1)
    return nullcontext()


def validate_overlap_schedule(model):
    """Reject configurations whose eager behavior the schedule cannot reproduce."""
    config = model.config
    if model._has_custom_layers:
        raise ValueError("Custom hybrid layers do not support overlap_moe_expert_parallel_comm.")
    if config.cuda_graph_impl != "none":
        raise ValueError(
            "overlap_moe_expert_parallel_comm with HybridModel does not support CUDA graphs "
            "yet. Set cuda_graph_impl='none'."
        )
    if config.enable_mhc_connections or config.moe_shortcut_connection:
        raise ValueError(
            "overlap_moe_expert_parallel_comm with HybridModel does not support "
            "enable_mhc_connections or moe_shortcut_connection."
        )
    # A first-class layer config can enable wide residuals independently of the root.
    # These callables bypass the stack's residual expansion/readout and the MoE residual write.
    if config.wide_residual is not None or any(
        layer_config.wide_residual is not None for layer_config in model.decoder.layer_config_list
    ):
        raise ValueError(
            "overlap_moe_expert_parallel_comm with HybridModel does not support wide_residual."
        )
    if config.recompute_granularity == "full":
        raise ValueError(
            "overlap_moe_expert_parallel_comm with HybridModel does not support "
            "recompute_granularity='full' yet."
        )
    # The schedule calls physical layers directly. HybridStack.forward owns CP
    # layout conversions and the packed metadata for chunkwise linear layers.
    decoder = model.decoder
    # A dynamic microbatch can use multiple CP ranks even when the static group has size 1.
    may_use_multi_rank_cp = config.dynamic_context_parallel or decoder.cp_group.size() > 1
    if may_use_multi_rank_cp and any(
        layout != decoder.boundary_layout for layout in decoder.layer_cp_layouts
    ):
        raise ValueError(
            "overlap_moe_expert_parallel_comm with HybridModel does not support mixed "
            "context-parallel layouts (linear_cp_layout != attention_cp_layout with "
            "context_parallel_size > 1)."
        )
    if may_use_multi_rank_cp and decoder._has_linear_layer_with_chunkwise_cp:
        raise ValueError(
            "overlap_moe_expert_parallel_comm with HybridModel does not support "
            "linear_cp_mode='chunkwise' with context_parallel_size > 1."
        )


def hybrid_layer_schedule_groups(decoder):
    """Build schedule slots over the flat, stage-local decoder layers."""
    groups = []
    start = 0
    sizes = decoder.schedule_group_sizes or (1,) * len(decoder.layers)
    for size in sizes:
        end = start + size
        layers = tuple(decoder.layers[i] for i in range(start, end))
        callables = build_hybrid_stack_callables(layers, decoder.layer_config_list[start:end])
        groups.append(LayerScheduleGroup(layers, *callables))
        start = end
    return groups


def _split_hybrid_layers_for_overlap(layers, configs):
    layer_items = []
    for config, item_layer in zip(configs, layers, strict=True):
        if layer_utils.get_custom_layer_type(config) is not None:
            raise ValueError(
                "Custom hybrid layers do not support overlap_moe_expert_parallel_comm."
            )
        config_type = type(config)
        if config_type is layer_utils.CSALayerConfig:
            symbol = layer_utils.get_layer_symbol_from_config(config)
            raise ValueError(f"HybridStack overlap does not support layer type '{symbol}'.")
        layer_items.append((config_type, item_layer))

    terminal_idx = None
    for idx, (item_type, _) in enumerate(layer_items):
        if item_type in (layer_utils.MLPLayerConfig, layer_utils.MoELayerConfig):
            terminal_idx = idx
            break

    if terminal_idx is not None and terminal_idx != len(layer_items) - 1:
        raise ValueError("HybridStack overlap requires MLP/MoE to be the last layer in a group.")

    terminal_type = layer_items[terminal_idx][0] if terminal_idx is not None else None
    terminal_layer = layer_items[terminal_idx][1] if terminal_idx is not None else None
    pre_layers = layer_items[:terminal_idx] if terminal_idx is not None else layer_items
    is_moe = terminal_type is layer_utils.MoELayerConfig
    num_local_experts = terminal_layer.mlp.num_local_experts if is_moe else None
    return pre_layers, terminal_type, terminal_layer, is_moe, num_local_experts


def _run_moe_preprocess(layer, node: ScheduleNode, hidden_states: Tensor):
    pre_mlp_layernorm_output = layer._forward_pre_mlp_layernorm(hidden_states)
    # A subsequent microbatch can reuse this layer before this node combines.
    node.layer_state.mlp_norm_manager = layer.mlp_norm_manager
    layer.mlp_norm_manager = None
    if isinstance(pre_mlp_layernorm_output, tuple):
        if len(pre_mlp_layernorm_output) != 2:
            raise ValueError(
                f"When the output of pre_mlp_layernorm is a tuple, it is expected to have "
                f"2 elements (output, residual), but got {len(pre_mlp_layernorm_output)}"
            )
        pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
    else:
        residual = hidden_states

    if layer.config.fp32_residual_connection:
        residual = residual.float()

    # Same routing inputs as the eager TransformerLayer._forward_mlp path: a packed batch
    # flattened to [mbs*s, 1, h] is restored to [s, mbs, h] so the router computes its
    # per-sample aux loss, and route and preprocess take the batch-first padding mask
    # (dropless HybridEP excludes padded tokens). Combine flattens the output back.
    pre_mlp_layernorm_output, padding_mask, _, moe_unflatten_mbs = layer._maybe_unflatten_for_moe(
        pre_mlp_layernorm_output,
        node.chunk_state.padding_mask,
        None,
        node.chunk_state.packed_seq_params,
    )
    node.layer_state.moe_unflatten_mbs = moe_unflatten_mbs
    shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
    probs, routing_map = layer.mlp.route(
        pre_mlp_layernorm_output, padding_mask, packed_seq_params=node.chunk_state.packed_seq_params
    )
    local_tokens, probs = layer.mlp.preprocess(
        pre_mlp_layernorm_output,
        probs,
        routing_map,
        padding_mask,
        packed_seq_params=node.chunk_state.packed_seq_params,
    )

    node.layer_state.residual = node.detach(residual)
    node.layer_state.shared_expert_output = None
    if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
        node.layer_state.shared_expert_output = node.detach(shared_expert_output)

    return local_tokens, probs


def _run_moe_experts(layer, node: ScheduleNode, dispatched_tokens: Tensor):
    dispatched_probs = node.layer_state.dispatched_probs
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend in ("deepep", "deepepv2")
    )
    enable_ncclep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "ncclep"
    )
    token_dispatcher = layer.mlp.token_dispatcher
    if enable_deepep or enable_hybridep or enable_ncclep:
        token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

    expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)

    if enable_hybridep or enable_ncclep:
        tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
        node.layer_state.tokens_per_expert = tokens_per_expert

    if layer.recompute_pre_mlp_layernorm:
        layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

    return expert_output


def _run_moe_combine(layer, node: ScheduleNode, output: Tensor):
    residual = node.layer_state.residual
    shared_expert_output = node.layer_state.shared_expert_output
    output = layer.mlp.combine(output)
    output = layer.mlp.postprocess(output, shared_expert_output)
    output = layer._maybe_reflatten_from_moe(
        output, node.chunk_state.packed_seq_params, node.layer_state.moe_unflatten_mbs
    )
    node.layer_state.moe_unflatten_mbs = None
    # Inline bda instead of calling ``layer._forward_post_mlp``, which would register
    # a second pre_mlp_layernorm recompute hook on the combine output. The hook is
    # already registered on ``expert_output`` in ``_run_moe_experts``. A second one
    # would run the layernorm recompute before attention's backward in the same
    # pre-dispatch slot of a bracketed group (``[*E]``), which corrupts the attention
    # gradients.
    mlp_output_with_bias = (output, None)
    with layer.bias_dropout_add_exec_handler():
        output = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual, layer.hidden_dropout
        )
    mlp_norm_manager = node.layer_state.mlp_norm_manager
    if mlp_norm_manager is not None:
        output = mlp_norm_manager.group_offload(output, forced_released_tensors=[residual])
        node.layer_state.mlp_norm_manager = None
    output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

    node.layer_state.residual.record_stream(torch.cuda.current_stream())
    if shared_expert_output is not None:
        shared_expert_output.record_stream(torch.cuda.current_stream())

    node.layer_state.residual = None
    node.layer_state.shared_expert_output = None

    return output


def build_hybrid_stack_callables(layers, configs):
    """Split a contiguous group of physical layers into compute and communication slots."""
    pre_layers, terminal_type, terminal_layer, is_moe, num_local_experts = (
        _split_hybrid_layers_for_overlap(layers, configs)
    )

    def pre_dispatch_computation(node: ScheduleNode, hidden_states: Tensor):
        for item_type, item_layer in pre_layers:
            with _get_inner_quant_context(item_layer):
                if item_type is layer_utils.MambaLayerConfig:
                    # The schedule releases each physical FSDP unit explicitly. Calling
                    # forward directly leaves the unit hooks to the schedule, as for attention.
                    hidden_states = item_layer.forward(
                        hidden_states,
                        attention_mask=node.chunk_state.attention_mask,
                        packed_seq_params=node.chunk_state.packed_seq_params,
                    )
                elif item_type in _ATTENTION_CONFIG_TYPES:
                    # Use _forward_attention rather than __call__: an attention half-layer has
                    # mlp=IdentityOp / mlp_bda=IdentityFuncOp by default, and TransformerLayer's
                    # __call__ would route through _forward_mlp + mlp_bda, double-applying the
                    # post-attention residual.
                    hidden_states, _ = item_layer._forward_attention(
                        hidden_states,
                        attention_mask=node.chunk_state.attention_mask,
                        rotary_pos_emb=node.chunk_state.rotary_pos_emb,
                        packed_seq_params=node.chunk_state.packed_seq_params,
                    )
                    # _forward_attention returns the bias_dropout_add output which can be a
                    # view tensor (the mlp_bda's add into the post-attention residual produces
                    # a view from a fused/JIT kernel). Downstream cuBLAS matmuls — including
                    # the terminal MLP/MoE's pre_mlp_layernorm and the next attention's QKV
                    # projection in a multi-pre-layer group — pick algorithms based on input
                    # strides, so a view can make the forward output differ slightly between
                    # processes. TransformerLayer makes its MLP output viewless for the same
                    # reason (``TransformerLayer._apply_mlp_bda_step``); the _forward_attention
                    # shortcut here does not, so do it explicitly.
                    hidden_states = make_viewless_tensor(
                        inp=hidden_states,
                        requires_grad=hidden_states.requires_grad,
                        keep_graph=True,
                    )
                else:
                    raise ValueError(
                        f"HybridStack overlap does not support layer type '{item_type.__name__}' "
                        "before the terminal MLP/MoE layer."
                    )

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        if terminal_type is layer_utils.MoELayerConfig:
            with _get_inner_quant_context(terminal_layer):
                return _run_moe_preprocess(terminal_layer, node, hidden_states)

        return hidden_states

    def dispatch(node: ScheduleNode, local_tokens: Tensor, probs: Tensor):
        enable_hybridep = (
            terminal_layer.config.moe_token_dispatcher_type == "flex"
            and terminal_layer.config.moe_flex_dispatcher_backend == "hybridep"
        )
        enable_deepep = (
            terminal_layer.config.moe_token_dispatcher_type == "flex"
            and terminal_layer.config.moe_flex_dispatcher_backend in ("deepep", "deepepv2")
        )
        enable_ncclep = (
            terminal_layer.config.moe_token_dispatcher_type == "flex"
            and terminal_layer.config.moe_flex_dispatcher_backend == "ncclep"
        )
        token_dispatcher = terminal_layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep or enable_ncclep:
            token_dispatcher._comm_manager.token_probs = probs
        with _get_inner_quant_context(terminal_layer):
            dispatched_tokens, dispatched_probs = terminal_layer.mlp.dispatch(local_tokens, probs)
        if enable_ncclep and terminal_layer.config.moe_ncclep_zero_copy:
            dispatched_tokens = StageDispatchBwdGrad.apply(dispatched_tokens, token_dispatcher)
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def mlp(node: ScheduleNode, hidden_states: Tensor):
        if terminal_type is layer_utils.MLPLayerConfig:
            with _get_inner_quant_context(terminal_layer):
                return terminal_layer._forward_mlp(
                    hidden_states, padding_mask=node.chunk_state.padding_mask
                )
        if terminal_type is layer_utils.MoELayerConfig:
            with _get_inner_quant_context(terminal_layer):
                return _run_moe_experts(terminal_layer, node, hidden_states)
        return hidden_states

    def combine(node: ScheduleNode, output: Tensor):
        with _get_inner_quant_context(terminal_layer):
            return _run_moe_combine(terminal_layer, node, output)

    def raise_not_implemented(*args):
        raise NotImplementedError("This callable is not implemented for non-MoE hybrid layers.")

    backward_dw = {}
    pre_bwd_dw = []
    for item_type, item_layer in pre_layers:
        if not item_layer.config.delay_wgrad_compute:
            continue
        if item_type in _ATTENTION_CONFIG_TYPES:
            # TransformerLayer-backed pre-layers go through the standard
            # _BackwardDWWrapper which coordinates attn / shared-expert wgrad
            # with cuda-graph replay scopes.
            item_layer.init_backward_dw_wrapper()
            pre_bwd_dw.append(item_layer.backward_dw_wrapper)
        elif item_type is layer_utils.MambaLayerConfig:
            # MambaLayer is not a TransformerLayer, so init_backward_dw_wrapper
            # would assert. MambaLayer.backward_dw delegates to its mixer, which
            # in turn calls backward_dw on the in_proj / out_proj linears. The
            # schedule node iterates this list and calls .backward_dw() on each;
            # registering the layer directly is sufficient.
            if not isinstance(item_layer.mixer, MambaMixer):
                raise ValueError(
                    "delay_wgrad_compute with overlap_moe_expert_parallel_comm does not support "
                    f"Mamba layers with a {type(item_layer.mixer).__name__} mixer."
                )
            pre_bwd_dw.append(item_layer)
    if is_moe and terminal_layer.config.delay_wgrad_compute:
        # Each slot owns the hooks for exactly the parameters whose delayed wgrad
        # it computes. Shared-expert dgrad has not run yet in the routed MLP slot.
        pre_dispatch_dw = _MoEBackwardDWWrapper(terminal_layer.mlp, routed_experts=False)
        if pre_dispatch_dw.submodules:
            pre_bwd_dw.append(pre_dispatch_dw)
        backward_dw["mlp"] = _MoEBackwardDWWrapper(terminal_layer.mlp, routed_experts=True)
    elif terminal_type is layer_utils.MLPLayerConfig and terminal_layer.config.delay_wgrad_compute:
        backward_dw["mlp"] = terminal_layer.mlp

    if pre_bwd_dw:
        backward_dw["pre_dispatch_computation"] = pre_bwd_dw

    forward_funcs = [
        pre_dispatch_computation,
        dispatch if is_moe else raise_not_implemented,
        mlp,
        combine if is_moe else raise_not_implemented,
        None,
    ]
    return forward_funcs, backward_dw, is_moe, num_local_experts


class HybridEmbeddingNode(PreProcessNode):
    """Prepare a training microbatch or receive the previous pipeline stage's activations."""

    def forward_impl(self):
        """Store the inputs used by the physical layer callables."""
        from .hybrid_model import _align_padding_mask, _embed_tokens, _rotary_embedding

        model, state = self.model, self.chunk_state
        if not model.pre_process:
            state.decoder_input = model.decoder.input_tensor
        elif state.decoder_input is None:
            state.decoder_input = _embed_tokens(model, state.input_ids, state.position_ids)
        state.padding_mask = _align_padding_mask(model, state.decoder_input, state.padding_mask)
        state.rotary_pos_emb = (
            None
            if model.hybrid_layer_config_list is not None and model._decoder_uses_mla
            else _rotary_embedding(model, state.decoder_input, state.packed_seq_params)
        )
        if not model.pre_process and not model.post_process and len(model.decoder.layers) == 0:
            # The pipeline can deallocate the sent output while backward still needs the input.
            state.decoder_input = state.decoder_input.clone()
        return state.decoder_input


class HybridOutputNode(PostProcessNode):
    """Produce the training output after the last logical layer of the chunk."""

    def forward_impl(self, hidden_states):
        """Apply the decoder's final norm and the same output head used by eager training."""
        from .hybrid_model import _project_logits, _run_output_processor

        model, state = self.model, self.chunk_state
        final_norm = getattr(model.decoder, "final_norm", None)
        if final_norm is not None:
            hidden_states = final_norm(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
        weight = (
            model.shared_embedding_or_output_weight()
            if model.share_embeddings_and_output_weights
            else None
        )
        if state.output_processor is not None:
            output = _run_output_processor(
                model,
                state.output_processor,
                hidden_states=hidden_states,
                output_weight=weight,
                labels=state.labels,
                loss_mask=state.loss_mask,
                input_ids=state.input_ids,
                position_ids=state.position_ids,
                attention_mask=state.attention_mask,
                decoder_input=state.decoder_input,
                inference_context=None,
                packed_seq_params=state.packed_seq_params,
                runtime_gather_output=state.runtime_gather_output,
                context=state.output_processor_context,
            )
        else:
            logits = _project_logits(model, hidden_states, weight, state.runtime_gather_output)
            if state.labels is None:
                output = logits.transpose(0, 1).contiguous()
            else:
                output = model.compute_language_model_loss(state.labels, logits)
        return float16_to_fp32(output)


class HybridSchedulePlan(TransformerModelChunkSchedulePlan):
    """Use the Hybrid embedding and output head around the shared 1F1B executor."""

    PRE_PROCESS_NODE_CLASS = HybridEmbeddingNode
    POST_PROCESS_NODE_CLASS = HybridOutputNode


def build_hybrid_schedule_plan(model, *args, **kwargs):
    """Attach the stage's groups of physical decoder layers to the overlap schedule."""
    if model.config.fine_grained_activation_offloading:
        model.preprocess_for_fine_grained_offloading()
    if model.config.moe_paged_stash:
        model.preprocess_for_paged_stash()
    return HybridSchedulePlan(
        model, *args, decoder_layer_groups=hybrid_layer_schedule_groups(model.decoder), **kwargs
    )
