# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Schedule-plan helpers shared by GPTModel and HybridModel.

These pieces used to live in ``core/models/gpt/fine_grained_callables.py`` and
were imported by ``core/models/common/model_chunk_schedule_plan.py`` and the
hybrid schedule plan via that path. They are model-agnostic in practice — the
``Pre/PostProcessNode`` classes call the model's ``_preprocess`` /
``_postprocess`` methods and don't otherwise care which model implements
them — so they live here and the GPT module re-exports them for backward
compatibility with existing imports.
"""

import weakref
from functools import partial
from typing import Callable

import torch

from megatron.core.pipeline_parallel.utils import ScheduleNode, make_viewless
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.module import GraphableMegatronModule, float16_to_fp32
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.utils import internal_api, nvtx_range_pop, nvtx_range_push


def weak_method(method):
    """Wrap ``method`` in a weakref-keyed dispatcher to break refcycles.

    ``ScheduleNode`` keeps a reference to the bound forward / backward functions
    of every node in the plan; using a strong reference would keep the layer
    plan (and the model chunk through it) alive after the iteration completes.
    The ``weakref.WeakMethod`` lets the schedule plan be torn down between
    iterations without manual ``del`` chains.
    """
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        return method_ref()(*args, **kwarg)

    return wrapped_func


@internal_api
def should_free_input(name, is_moe, config, num_local_experts):
    """Whether the schedule node named ``name`` can free its input after forward.

    The schedule decomposes a transformer layer into ``pre_dispatch_computation``,
    ``moe_dispatch``, ``mlp``, and ``moe_combine`` nodes; the inputs to some of
    those nodes are not needed in backward and can be released early to lower
    peak activation memory. Dense layers and the ``pre_dispatch_computation``
    node always need their input retained (the attention residual flows through
    the post-MLP BDA).

    Args:
        name: Schedule node name.
        is_moe: True for MoE layers; dense layers always retain inputs.
        config: ``TransformerConfig`` for the layer.
        num_local_experts: Local expert count on this rank (None for dense).

    Returns:
        True iff the named node may free its input after forward.
    """
    # For dense layers [pre_dispatch_computation, fake, mlp, fake], the input is needed
    # during backward pass
    if not is_moe:
        return False
    enable_deepep = (
        config.moe_token_dispatcher_type == "flex"
        and config.moe_flex_dispatcher_backend == "deepep"
    )
    enable_hybridep = (
        config.moe_token_dispatcher_type == "flex"
        and config.moe_flex_dispatcher_backend == "hybridep"
    )
    # Define which nodes should free input memory.
    # Since we split the computing graph into multiple nodes, we can manually control
    # when and how to free the input memory.
    # The input and output of A2A are not needed anymore after the forward pass,
    # so we can free the input memory after the forward pass.

    # When low precision fp8/4 is enabled, the casted tensors are saved and the
    # original bf16 tensors are safe to be freed.
    free_mlp = config.fp8 is not None or config.fp4 is not None
    if not free_mlp:
        # AlltoAll dispatcher with local_num_experts=1 and HybridEP both use identity
        # operation for `dispatch_postprocess`, hence the mlp inputs will be directly
        # passed to GroupedGemm and should be saved for backward pass.
        free_mlp = num_local_experts > 1 or config.moe_token_dispatcher_type != "alltoall"
        free_mlp = free_mlp and not enable_hybridep

    free_input_nodes = {
        "mlp": free_mlp,
        "moe_combine": True,
        # For non-DeepEP and non-HybridEP dispatcher mode, the input is the un-dispatched
        # tokens and probs before dispatch A2A and it's not needed anymore after the
        # forward pass. For DeepEP and HybridEP dispatcher mode, they are both needed in
        # backward pass and cannot be freed.
        # If moe_preprocess is in cuda graph scope, tokens and probs are fixed size
        # tensors, so they cannot be freed.
        "moe_dispatch": not (enable_deepep or enable_hybridep)
        and (CudaGraphScope.moe_preprocess not in config.cuda_graph_scope),
    }

    return free_input_nodes.get(name, False)


class LayerState:
    """State shared between the schedule nodes that come from one logical layer.

    Empty placeholder; nodes attach their own attributes (residual, dispatched
    probs, shared-expert outputs) for downstream nodes in the same layer to
    consume. Kept as a real class so weakrefs work uniformly.
    """

    pass


class PreProcessNode(ScheduleNode):
    """Run the model's ``_preprocess`` (embedding + rotary + padding mask).

    The schedule plan wraps a model that exposes a ``_preprocess`` method
    returning the canonical 6-tuple ``(decoder_input, rotary_pos_emb,
    rotary_pos_cos, rotary_pos_sin, sequence_len_offset, padding_mask)``
    (slots a given model doesn't use are returned as ``None``). The chunk
    state is mutated in-place so layer nodes can read the same fields by
    name.
    """

    def __init__(self, model, chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event, name="pre_process")
        self.model = model
        self.chunk_state = chunk_state

    def forward_impl(self):
        if not self.model.pre_process:
            self.chunk_state.decoder_input = self.model.decoder.input_tensor
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = self.model._preprocess(
            input_ids=self.chunk_state.input_ids,
            position_ids=self.chunk_state.position_ids,
            decoder_input=self.chunk_state.decoder_input,
            packed_seq_params=self.chunk_state.packed_seq_params,
            padding_mask=self.chunk_state.padding_mask,
        )

        self.chunk_state.decoder_input = decoder_input
        self.chunk_state.rotary_pos_emb = rotary_pos_emb
        self.chunk_state.rotary_pos_cos = rotary_pos_cos
        self.chunk_state.rotary_pos_sin = rotary_pos_sin
        self.chunk_state.sequence_len_offset = sequence_len_offset
        self.chunk_state.padding_mask = padding_mask
        return decoder_input


class PostProcessNode(ScheduleNode):
    """Run the model's ``_postprocess`` (final norm, output layer, loss).

    Calls ``_postprocess`` with ``mtp_in_postprocess=False`` because the
    schedule plan handles MTP layers as sibling layer nodes inside the same
    chunk; the model's MTP block is not invoked here. The optional final
    layernorm — applied only when this rank holds an empty decoder shard
    (early stage of pipeline parallel) — is handled here so the chunk plan
    does not need a separate node for it.
    """

    def __init__(self, model, chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event, name="post_process")
        self.model = model
        self.chunk_state = chunk_state

    def forward_impl(self, hidden_states):
        empty_decoder = len(self.model.decoder.layers) == 0
        layer_norm = self.model.decoder.final_layernorm
        if not self.model.config.mtp_num_layers and empty_decoder and layer_norm:
            hidden_states = layer_norm(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        loss = self.model._postprocess(
            hidden_states=hidden_states,
            input_ids=self.chunk_state.input_ids,
            position_ids=self.chunk_state.position_ids,
            labels=self.chunk_state.labels,
            decoder_input=self.chunk_state.decoder_input,
            rotary_pos_emb=self.chunk_state.rotary_pos_emb,
            rotary_pos_cos=self.chunk_state.rotary_pos_cos,
            rotary_pos_sin=self.chunk_state.rotary_pos_sin,
            mtp_in_postprocess=False,
            loss_mask=self.chunk_state.loss_mask,
            attention_mask=self.chunk_state.attention_mask,
            packed_seq_params=self.chunk_state.packed_seq_params,
            sequence_len_offset=self.chunk_state.sequence_len_offset,
            runtime_gather_output=self.chunk_state.runtime_gather_output,
            extra_block_kwargs=self.chunk_state.extra_block_kwargs,
        )

        # combined-1F1B currently expects fp32 loss output.
        return float16_to_fp32(loss)


class TransformerLayerNode(ScheduleNode):
    """Schedule node for one slot of a fine-grained transformer layer plan.

    Each transformer layer is decomposed into ``pre_dispatch_computation``,
    ``moe_dispatch``, ``mlp``, and ``moe_combine`` slots; this class is the scheduler-side
    handle for one slot. It owns the slot's stream / event, the per-slot
    ``free_input`` policy, and the optional delayed weight-gradient hook.
    Subclasses override ``_resolve_free_input`` to specialize the policy
    (HybridStackNode does this for grouped layers).
    """

    def __init__(
        self,
        stream,
        event,
        layer_state,
        chunk_state,
        submodule,
        name="default",
        bwd_dw_callables=None,
        extra_args={},
    ):
        config = extra_args.get("config", None)
        assert config is not None, "model config must be passed to TransformerLayerNode."
        is_moe = extra_args.get("is_moe", False)
        num_local_experts = extra_args.get("num_local_experts", None)
        free_input = self._resolve_free_input(name, is_moe, config, num_local_experts)
        self.delay_wgrad_compute = extra_args.get("delay_wgrad_compute", False)

        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            free_input=free_input,
            name=name,
        )
        self.layer_state = layer_state
        self.chunk_state = chunk_state
        self.submodule = submodule
        self.detached = tuple()
        self.before_detached = tuple()
        self.is_mtp = extra_args.get("is_mtp", False)

        self.is_first_layer = extra_args.get("is_first_layer", False)
        self.is_last_layer = extra_args.get("is_last_layer", False)

        self.bwd_dw_callables = []
        if bwd_dw_callables is not None:
            self.bwd_dw_callables = (
                bwd_dw_callables if isinstance(bwd_dw_callables, list) else [bwd_dw_callables]
            )

    @staticmethod
    def _resolve_free_input(name, is_moe, config, num_local_experts):
        """Free-input policy hook. Subclasses override to specialize."""
        return should_free_input(name, is_moe, config, num_local_experts)

    def detach(self, t):
        """Detach a tensor and remember it for backward through the schedule node."""
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        self.before_detached = self.before_detached + (t,)
        self.detached = self.detached + (detached,)
        return detached

    def forward_impl(self, *args):
        """Invoke the slot's submodule forward."""
        return self.submodule(self, *args)

    def backward_impl(self, outputs, output_grad):
        """Run the slot's backward, holding output_grads when wgrad is delayed."""
        detached_grad = tuple([e.grad for e in self.detached])
        grads = output_grad + detached_grad
        self.default_backward_func(outputs + self.before_detached, grads)
        # Release the output grad memory after backward finishes, except when
        # delay_wgrad_compute is enabled — then the grads are kept until every
        # registered ``backward_dw`` callable has run.
        if self.delay_wgrad_compute:
            self.output_grads = grads
            self.delay_grads_release = len(self.bwd_dw_callables) > 0

        return grads

    def backward_dw(self):
        """Run the slot's delayed weight-gradient callables on the slot's stream."""
        if not self.delay_wgrad_compute:
            return
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with torch.cuda.stream(self.stream):
            nvtx_msg = f"{self.name} wgrad"
            nvtx_range_push(nvtx_msg)
            for module in self.bwd_dw_callables:
                module.backward_dw()
            nvtx_range_pop(nvtx_msg)

        # The output grad memory is last used in wgrad compute; safe to release now.
        assert self.delay_grads_release, "output grad memory should be valid before wgrad."
        if self.manual_release_grads:
            for tensor in self.output_grads:
                tensor.untyped_storage().resize_(0)
        self.output_grads = None

        self.bwd_dw_callables = None

    def __del__(self):
        # Release references early to help avoid leaks across iterations.
        self.before_detached = None
        self.detached = None
        self.layer_state = None
        self.chunk_state = None
        self.submodule = None


class _BackwardDWWrapper:
    """Backward weight-gradient wrapper for the ``pre_dispatch_computation`` slot of a transformer layer.

    Runs the layer's ``self_attention.backward_dw`` plus, on MoE layers, the
    shared-expert ``backward_dw``; coordinates with the cuda-graph wgrad
    capture (``set_graphed_backward_dw_callable``) so that scopes covered by
    the graph are not re-run eagerly. Used when
    ``overlap_moe_expert_parallel_comm`` and ``delay_wgrad_compute`` are both
    enabled.
    """

    def __init__(self, layer):
        assert isinstance(
            layer, GraphableMegatronModule
        ), "cuda graphed ep overlap only supports GraphableMegatronModule."
        assert isinstance(
            layer, TransformerLayer
        ), "cuda graphed ep overlap only supports TransformerLayer for now."
        self.layer = layer
        self.graphed_backward_dw_callable = None
        self.attn_dw_callable = layer.self_attention.backward_dw
        if layer.is_moe_layer:
            self.shared_expert_dw_callable = partial(
                layer.mlp.backward_dw, routed_experts=False, shared_experts=True
            )
        else:
            self.shared_expert_dw_callable = None
        self.cuda_graph_scope = layer.config.cuda_graph_scope

    def backward_dw(self):
        is_replay = hasattr(self.layer, 'cuda_graphs') and self.layer.cuda_graphs
        if self.shared_expert_dw_callable is not None and (
            not is_replay or CudaGraphScope.moe_router not in self.cuda_graph_scope
        ):
            self.shared_expert_dw_callable()
        if not is_replay or CudaGraphScope.attn not in self.cuda_graph_scope:
            self.attn_dw_callable()
        if is_replay and self.graphed_backward_dw_callable is not None:
            self.graphed_backward_dw_callable()
        self.layer = None

    def set_graphed_backward_dw_callable(self, graphed_backward_dw_callable):
        """Plug the cuda-graph backward wgrad replay callable."""
        self.graphed_backward_dw_callable = graphed_backward_dw_callable
