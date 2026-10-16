# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Schedule-plan helpers shared by GPTModel and HybridModel.

These pieces used to live in ``core/models/gpt/fine_grained_callables.py`` and
were imported by ``core/models/common/model_chunk_schedule_plan.py`` and the
hybrid schedule plan via that path. They are model-agnostic in practice — the
``Pre/PostProcessNode`` classes call the model's ``_preprocess`` /
``_postprocess`` methods and don't otherwise care which model implements
them — so they live here now.
"""

import weakref
from functools import partial
from typing import Callable

import torch

from megatron.core.pipeline_parallel.utils import (
    ScheduleNode,
    ScheduleNodeInputWrapper,
    make_viewless,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import GraphableMegatronModule, float16_to_fp32
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


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
        """Run model preprocessing and store chunk-level inputs for layer nodes."""
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
        """Run model postprocessing for the chunk's final hidden states."""
        empty_decoder = len(self.model.decoder.layers) == 0
        layer_norm = getattr(self.model.decoder, "final_norm", None) or getattr(
            self.model.decoder, "final_layernorm", None
        )
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
            mtp_input_mask=self.chunk_state.mtp_input_mask,
            attention_mask=self.chunk_state.attention_mask,
            packed_seq_params=self.chunk_state.packed_seq_params,
            sequence_len_offset=self.chunk_state.sequence_len_offset,
            runtime_gather_output=self.chunk_state.runtime_gather_output,
            extra_block_kwargs=self.chunk_state.extra_block_kwargs,
            output_processor=self.chunk_state.output_processor,
            output_processor_context=self.chunk_state.output_processor_context,
        )

        # combined-1F1B currently expects fp32 loss output.
        return float16_to_fp32(loss)


class TransformerLayerNode(ScheduleNode):
    """Schedule node for one slot of a fine-grained transformer layer plan.

    Each transformer layer is decomposed into ``pre_dispatch_computation``,
    ``moe_dispatch``, ``mlp``, and ``moe_combine`` slots; this class is the scheduler-side
    handle for one slot. It owns the slot's stream / event and the optional
    delayed weight-gradient hook.
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
        self.delay_wgrad_compute = extra_args.get("delay_wgrad_compute", False)

        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            name=name,
        )
        self.layer_state = layer_state
        self.chunk_state = chunk_state
        self.submodule = submodule
        self.detached_wrappers = tuple()
        self.before_detached_edges = tuple()
        self.is_mtp = extra_args.get("is_mtp", False)
        self.post_wgrad_grad_acc_hooks = None

        self.is_first_layer = extra_args.get("is_first_layer", False)
        self.is_last_layer = extra_args.get("is_last_layer", False)

        # Whether this slot is the first/last node of its TransformerLayer in
        # forward / backward order. Set by ``set_post_*_hook``; used to decide
        # when to invoke the layer-level FSDP reshard hooks.
        self.is_layer_first_node = None
        self.is_layer_last_node = None

        self.bwd_dw_callables = []
        if bwd_dw_callables is not None:
            self.bwd_dw_callables = (
                bwd_dw_callables if isinstance(bwd_dw_callables, list) else [bwd_dw_callables]
            )

    def detach(self, t):
        """Create a storage-free autograd boundary for a saved side output."""
        t = make_viewless(t)
        before_detached_edge = (
            torch.autograd.graph.get_gradient_edge(t) if t.requires_grad else None
        )
        wrapper = ScheduleNodeInputWrapper(t)
        if before_detached_edge is not None:
            self.before_detached_edges = self.before_detached_edges + (before_detached_edge,)
            self.detached_wrappers = self.detached_wrappers + (wrapper,)
        detached = wrapper.unwrap_real_tensor()
        return detached

    def forward_impl(self, *args):
        """Invoke the slot's submodule forward."""
        return self.submodule(self, *args)

    def backward_impl(self, output_edges, output_grads):
        """Run the slot's backward and return the input grads."""
        detached_grads = tuple(wrapper.grad for wrapper in self.detached_wrappers)
        grads = output_grads + detached_grads
        self.default_backward_func(output_edges + self.before_detached_edges, grads)
        self.detached_wrappers = tuple()
        self.before_detached_edges = tuple()

        return grads

    def forward(self, *inputs):
        """Execute forward and fire the per-layer post-forward hook on the last slot."""
        output = super().forward(*inputs)
        if self.is_layer_last_node:
            self._post_forward_hook()
        return output

    def backward(self, *output_grad):
        """Execute backward and fire the per-layer post-backward hook on the first slot.

        When ``delay_wgrad_compute`` is set, the hook fires after ``backward_dw``
        instead, because the wgrad work has not yet run when ``backward`` returns.
        """
        grads = super().backward(*output_grad)
        if not self.delay_wgrad_compute and self.is_layer_first_node:
            self._post_backward_hook()
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

        # Collect ``post_wgrad_grad_acc_hook`` from params whose grads were
        # produced by *this* slot's wgrad callables. The hook must run on the
        # same stream right after the wgrad it depends on; collecting on the
        # first invocation makes the per-iteration hook order deterministic.
        if self.post_wgrad_grad_acc_hooks is None:
            self.post_wgrad_grad_acc_hooks = []
            for module in self.bwd_dw_callables:
                for param in module.parameters():
                    if (
                        getattr(param, "post_wgrad_grad_acc_hook", False)
                        and param.requires_grad
                        and param.grad is not None
                    ):
                        self.post_wgrad_grad_acc_hooks.append(param.post_wgrad_grad_acc_hook)

        if self.post_wgrad_grad_acc_hooks:
            with torch.cuda.stream(self.stream):
                for hook in self.post_wgrad_grad_acc_hooks:
                    hook()

        if self.is_layer_first_node:
            self._post_backward_hook()
        self.bwd_dw_callables = None

    def set_post_forward_hook(self, hook):
        """Mark this slot as the layer's last fwd node and register the hook."""
        self.is_layer_last_node = True
        self._post_forward_hook = hook

    def set_post_backward_hook(self, hook):
        """Mark this slot as the layer's first bwd node and register the hook."""
        self.is_layer_first_node = True
        self._post_backward_hook = hook

    def __del__(self):
        # Release references early to help avoid leaks across iterations.
        self.detached_wrappers = None
        self.before_detached_edges = None
        self.layer_state = None
        self.chunk_state = None
        self.submodule = None


class _BackwardDWWrapper:
    """Backward weight-gradient wrapper for a transformer pre-dispatch slot.

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
        self.submodules = [layer.self_attention]
        if layer.is_moe_layer:
            self.shared_expert_dw_callable = partial(
                layer.mlp.backward_dw, routed_experts=False, shared_experts=True
            )
            if layer.mlp.use_shared_expert:
                self.submodules.append(layer.mlp.shared_experts)
        else:
            self.shared_expert_dw_callable = None
        self.cuda_graph_modules = layer.config.cuda_graph_modules

    def backward_dw(self):
        """Run eager or graphed backward wgrad callables for the wrapped layer."""
        is_replay = hasattr(self.layer, 'cuda_graphs') and self.layer.cuda_graphs
        if self.shared_expert_dw_callable is not None and (
            not is_replay or CudaGraphModule.moe_router not in self.cuda_graph_modules
        ):
            self.shared_expert_dw_callable()
        if not is_replay or CudaGraphModule.attn not in self.cuda_graph_modules:
            self.attn_dw_callable()
        if is_replay and self.graphed_backward_dw_callable is not None:
            self.graphed_backward_dw_callable()
        self.layer = None

    def set_graphed_backward_dw_callable(self, graphed_backward_dw_callable):
        """Plug the cuda-graph backward wgrad replay callable."""
        self.graphed_backward_dw_callable = graphed_backward_dw_callable

    def parameters(self):
        """Yield parameters from the wrapped layer's wgrad submodules.

        Mirrors ``torch.nn.Module.parameters`` so callers (notably
        ``TransformerLayerNode.backward_dw``) can collect ``post_wgrad_grad_acc_hook``
        without knowing the concrete layer layout.
        """
        for module in self.submodules:
            for param in module.parameters():
                yield param
