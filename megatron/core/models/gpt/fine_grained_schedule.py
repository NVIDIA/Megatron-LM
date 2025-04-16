# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
import weakref
from typing import Optional

import torch
from torch import Tensor

from megatron.core.pipeline_parallel.combined_1f1b import (
    AbstractSchedulePlan,
    FakeScheduleNode,
    FreeInputsMemoryStrategy,
    NoOpMemoryStrategy,
    ScheduleNode,
    get_com_stream,
    get_comp_stream,
    make_viewless,
)
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer


def weak_method(method):
    """Creates a weak reference to a method to prevent circular references.

    This function creates a weak reference to a method and returns a wrapper function
    that calls the method when invoked. This helps prevent memory leaks from circular
    references.
    """
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        # nonlocal object_ref
        return method_ref()(*args, **kwarg)

    return wrapped_func


class MemoryStrategyRegistry:
    """Registry for memory management strategies based on node names.

    This class centralizes the definition of which memory strategy
    should be used for each type of node in the computation graph.
    """

    _strategies = {
        "default": NoOpMemoryStrategy(),
        "attn": NoOpMemoryStrategy(),  # Attention nodes keep their inputs
        "dispatch": FreeInputsMemoryStrategy(),  # Dispatch nodes free inputs after use
        "mlp": FreeInputsMemoryStrategy(),  # MLP nodes free inputs after use
        "combine": FreeInputsMemoryStrategy(),  # Combine nodes free inputs after use
    }

    @classmethod
    def get_strategy_by_name(cls, name, is_moe, is_deepep):
        """Gets the appropriate memory strategy for a node based on its name and MoE status.

        Args:
            name: The name of the node, which determines which strategy to use.
            is_moe: Whether the node is part of a Mixture of Experts model.

        Returns:
            The memory strategy to use for the node.
        """
        # TODO: add memory strategy for deepep
        if is_deepep:
            return NoOpMemoryStrategy()
        if is_moe:
            return cls._strategies.get(name, cls._strategies["default"])
        return NoOpMemoryStrategy()


class PreProcessNode(ScheduleNode):
    """Node responsible for preprocessing operations in the model.

    This node handles embedding and rotary positional embedding computations
    before the main transformer layers.
    """

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        """Initializes a preprocessing node.

        Args:
            gpt_model: The GPT model instance.
            model_chunk_state: State shared across the model chunk.
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="pre_process")
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self):
        """Implements the forward pass for preprocessing.

        This method handles:
        1. Decoder embedding computation
        2. Rotary positional embedding computation
        3. Sequence length offset computation for flash decoding

        Returns:
            The processed decoder input tensor.
        """
        gpt_model = self.gpt_model
        decoder_input = self.model_chunk_state.decoder_input
        input_ids = self.model_chunk_state.input_ids
        position_ids = self.model_chunk_state.position_ids
        inference_params = self.model_chunk_state.inference_params
        packed_seq_params = self.model_chunk_state.packed_seq_params

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif gpt_model.pre_process:
            decoder_input = gpt_model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = gpt_model.decoder.input_tensor

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if (
            gpt_model.position_embedding_type == 'rope'
            and not gpt_model.config.multi_latent_attention
        ):
            if not gpt_model.training and gpt_model.config.flash_decode and inference_params:
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = gpt_model.rotary_pos_emb_cache.setdefault(
                    inference_params.max_sequence_length,
                    gpt_model.rotary_pos_emb.get_cos_sin(inference_params.max_sequence_length),
                )
            else:
                rotary_seq_len = gpt_model.rotary_pos_emb.get_rotary_seq_len(
                    inference_params,
                    gpt_model.decoder,
                    decoder_input,
                    gpt_model.config,
                    packed_seq_params,
                )
                rotary_pos_emb = gpt_model.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
        if (
            (gpt_model.config.enable_cuda_graph or gpt_model.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_params
        ):
            sequence_len_offset = torch.tensor(
                [inference_params.sequence_len_offset] * inference_params.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # saved for later use
        self.model_chunk_state.rotary_pos_emb = rotary_pos_emb
        self.model_chunk_state.rotary_pos_cos = rotary_pos_cos
        self.model_chunk_state.rotary_pos_sin = rotary_pos_sin
        self.model_chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input


class PostProcessNode(ScheduleNode):
    """Node responsible for postprocessing operations in the model.

    This node handles final layer normalization and output layer computation
    after the main transformer layers.
    """

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        """Initializes a postprocessing node.

        Args:
            gpt_model: The GPT model instance.
            model_chunk_state: State shared across the model chunk.
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="post_process")
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self, hidden_states):
        """Implements the forward pass for postprocessing.

        This method handles:
        1. Final layer normalization
        2. Output layer computation
        3. Loss computation if labels are provided

        Args:
            hidden_states: The hidden states from the transformer layers.

        Returns:
            The logits or loss depending on whether labels are provided.
        """
        # Final layer norm.
        if self.gpt_model.decoder.final_layernorm is not None:
            hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = transformer_layer.make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        gpt_model = self.gpt_model
        runtime_gather_output = self.model_chunk_state.runtime_gather_output
        labels = self.model_chunk_state.labels
        output_weight = None
        if gpt_model.share_embeddings_and_output_weights:
            output_weight = gpt_model.shared_embedding_or_output_weight()
        logits, _ = gpt_model.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if labels is None:
            # [s b h] => [b s h]
            return float16_to_fp32(logits.transpose(0, 1).contiguous())
        loss = float16_to_fp32(gpt_model.compute_language_model_loss(labels, logits))
        return loss


class TransformerLayerNode(ScheduleNode):
    """Base class for transformer layer computation nodes.

    This class provides common functionality for different types of
    transformer layer nodes (attention, MLP, etc.)
    """

    def __init__(self, stream, event, state, callables, name="default"):
        """Initialize a transformer layer node.

        Args:
            stream (torch.cuda.Stream): CUDA stream for execution
            event (torch.cuda.Event): Synchronization event
            common_state (TransformerLayerState): State shared within a transformer layer
            callables (Callable): The callables contain forward and dw function
            it's the per_batch_state_context, o.w. nullcontext
            name (str): Node name, also used to determine memory strategy
        """
        # Get memory strategy based on node name
        memory_strategy = MemoryStrategyRegistry.get_strategy_by_name(name, callables.is_moe, callables.is_deepep)

        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            memory_strategy=memory_strategy,
            name=name,
        )
        self.common_state = state
        self.callables = callables
        self.detached = tuple()
        self.before_detached = tuple()

    def detach(self, t):
        """Detaches a tensor and stores it for backward computation."""
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        self.before_detached = self.before_detached + (t,)
        self.detached = self.detached + (detached,)
        return detached

    def forward_impl(self, *args):
        """Implements the forward pass for the transformer layer node."""
        return self.callables.forward(self, *args)

    def backward_impl(self, outputs, output_grad):
        """Implements the backward pass for the transformer layer node."""
        detached_grad = tuple([e.grad for e in self.detached])
        grads = output_grad + detached_grad
        self.default_backward_func(outputs + self.before_detached, grads)
        self.before_detached = None
        self.detached = None
        # return grads for record stream
        return grads

    def dw(self):
        """Computes the weight gradients for the transformer layer node."""
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.callables.dw()


class TransformerLayerState:
    """State shared within a transformer layer.

    This class holds state that is shared between different nodes
    within a transformer layer.
    """

    pass


class ModelChunkSate:
    """State shared across a model chunk.

    This class holds state that is shared between different components
    of a model chunk, such as input tensors, parameters, and configuration.
    """

    pass


class TransformerLayerSchedulePlan:
    """Schedule plan for a transformer layer.

    This class organizes the computation nodes for a transformer layer,
    including attention, MLP, dispatch, and combine nodes.
    """

    def __init__(self, layer, event, chunk_state, comp_stream, com_stream):
        """Initializes a transformer layer schedule plan.

        Args:
            layer (TransformerLayer): The transformer layer to schedule.
            event (torch.cuda.Event): CUDA event for synchronization.
            chunk_state (ModelChunkState): State shared across the model chunk.
            comp_stream (torch.cuda.Stream): CUDA stream for computation.
            com_stream (torch.cuda.Stream): CUDA stream for communication.
        """
        self.common_state = TransformerLayerState()
        # get callables for transformer layer
        attn_callable, dispatch_callable, mlp_callable, combine_callable = (
            layer.get_submodule_callables(chunk_state).as_array()
        )

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.attn = TransformerLayerNode(
            comp_stream, event, self.common_state, attn_callable, name="attn"
        )
        self.mlp = TransformerLayerNode(
            comp_stream, event, self.common_state, mlp_callable, name="mlp"
        )
        if attn_callable.is_moe:
            self.dispatch = TransformerLayerNode(
                com_stream, event, self.common_state, dispatch_callable, name="dispatch"
            )
            self.combine = TransformerLayerNode(
                com_stream, event, self.common_state, combine_callable, name="combine"
            )
        else:
            self.dispatch = FakeScheduleNode()
            self.combine = FakeScheduleNode()


class ModelChunkSchedulePlan(AbstractSchedulePlan):
    """Schedule plan for a model chunk.

    This class organizes the computation nodes for a model chunk,
    including preprocessing, transformer layers, and postprocessing.
    """

    def __init__(self):
        """Initializes a model chunk schedule plan."""
        super().__init__()
        self._pre_process = None
        self._post_process = None
        self._model_chunk_state = ModelChunkSate()
        self._transformer_layers = []
        self._event = torch.cuda.Event()

    @classmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Schedules forward and backward passes for model chunks.

        Args:
            f_schedule_plan (ModelChunkSchedulePlan): Forward schedule plan.
            b_schedule_plan (ModelChunkSchedulePlan): Backward schedule plan.
            grad (Tensor): Gradient for backward computation.
            f_context (VppContextManager or None): The VppContextManager for the forward pass.
            b_context (VppContextManager or None): The VppContextManager for the backward pass
            pre_forward (Callable): Callback for preprocessing in forward pass.
            pre_backward (Callable): Callback for preprocessing in backward pass.
            post_forward (Callable): Callback for postprocessing in forward pass.
            post_backward (Callable): Callback for postprocessing in backward pass.

        Returns:
            The output of the forward pass.
        """
        return schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            grad=grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

    @property
    def event(self):
        """Gets the CUDA event for synchronization."""
        return self._event

    def record_current_stream(self):
        """Records the current CUDA stream in the event."""
        stream = torch.cuda.current_stream()
        self.event.record(stream)

    def wait_current_stream(self):
        """Waits for the event to complete on the current CUDA stream."""
        stream = torch.cuda.current_stream()
        self.event.wait(stream)

    @property
    def pre_process(self):
        """Gets the preprocessing node."""
        return self._pre_process

    @pre_process.setter
    def pre_process(self, value):
        """Sets the preprocessing node."""
        self._pre_process = value

    @property
    def post_process(self):
        """Gets the postprocessing node."""
        return self._post_process

    @post_process.setter
    def post_process(self, value):
        """Sets the postprocessing node."""
        self._post_process = value

    def get_layer(self, i):
        """Gets the transformer layer at the specified index."""
        assert i < self.num_layers()
        return self._transformer_layers[i]

    def num_layers(self):
        """Gets the number of transformer layers."""
        return len(self._transformer_layers)

    def add_layer(self, layer):
        """Adds a transformer layer to the schedule plan."""
        self._transformer_layers.append(layer)

    @property
    def state(self):
        """Gets the model chunk state."""
        return self._model_chunk_state


def schedule_layer_1f1b(
    f_layer,
    b_layer,
    f_input=None,
    b_grad=None,
    pre_forward=None,
    pre_backward=None,
    pre_backward_dw=None,
    f_context=None,
    b_context=None,
):
    """Schedule one-forward-one-backward operations for a single layer.

    This function interleaves forward and backward operations to maximize
    parallelism and efficiency.

    Args:
        f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
        b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
        f_input (Tensor): Input for forward computation
        b_grad (Tensor): Gradient for backward computation
        pre_forward (Callable): Callback to get forward input if not provided
        pre_backward (Callable): Callback to get backward gradient if not provided
        pre_backward_dw (Callable): Callback for weight gradient computation
        f_context (VppContextManager or None): The VppContextManager for the forward pass.
        b_context (VppContextManager or None): The VppContextManager for the backward pass

    Returns:
        Functions or values for next iteration's computation
    """
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    if pre_forward is not None:
        assert f_input is None
        # combine from last iter
        f_input = pre_forward()
        del pre_forward

    if pre_backward is not None:
        # attn backward from last iter
        assert b_grad is None
        b_grad = pre_backward()
        del pre_backward

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.combine.backward(b_grad)

    if pre_backward_dw is not None:
        pre_backward_dw()
        del pre_backward_dw

    if f_layer is not None:
        with f_context:
            f_input = f_layer.attn.forward(f_input)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.dispatch.forward(f_input)

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.mlp.backward(b_grad)
            b_grad = b_layer.dispatch.backward(b_grad)
            b_layer.mlp.dw()

    if f_layer is not None:
        with f_context:
            f_input = f_layer.mlp.forward(f_input)

    def next_iter_pre_forward():
        if f_layer is not None:
            with f_context:
                output = f_layer.combine.forward(f_input)
                return output

    def next_iter_pre_backward():
        if b_layer is not None:
            with b_context:
                grad = b_layer.attn.backward(b_grad)
                return grad

    def next_iter_pre_backward_dw():
        if b_layer is not None:
            with b_context:
                b_layer.attn.dw()

    if f_layer and b_layer:
        return next_iter_pre_forward, next_iter_pre_backward, next_iter_pre_backward_dw
    else:
        return next_iter_pre_forward(), next_iter_pre_backward(), next_iter_pre_backward_dw()


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    grad=None,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
    """Schedules one-forward-one-backward operations for a model chunk.

    This function interleaves forward and backward operations across multiple layers
    to maximize parallelism and efficiency.

    Args:
        f_schedule_plan: Forward schedule plan.
        b_schedule_plan: Backward schedule plan.
        grad: Gradient for backward computation.
        f_context: Context for forward computation.
        b_context: Context for backward computation.
        pre_forward: Callback for preprocessing in forward pass.
        pre_backward: Callback for preprocessing in backward pass.
        post_forward: Callback for postprocessing in forward pass.
        post_backward: Callback for postprocessing in backward pass.

    Returns:
        The output of the forward pass.
    """
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    if f_schedule_plan:
        # pp output send/receive sync
        if pre_forward is not None:
            with f_context:  # virtual pipeline parallel context
                pre_forward()
        f_schedule_plan.record_current_stream()

    if b_schedule_plan:
        b_schedule_plan.record_current_stream()

    f_input = None

    def layer_pre_forward():
        tmp = f_input
        if f_schedule_plan is not None:
            tmp = f_schedule_plan.pre_process.forward()
        return tmp

    def layer_pre_backward():
        tmp = grad
        if b_schedule_plan is not None:
            assert grad is not None
            if b_schedule_plan.post_process is not None:
                with b_context:  # virtual pipeline parallel context
                    tmp = b_schedule_plan.post_process.backward(grad)

            if pre_backward is not None:
                # pp grad send receive sync here, safe for now, maybe not safe in the future
                with torch.cuda.stream(get_com_stream()):
                    b_schedule_plan.wait_current_stream()
                    with b_context:  # virtual pipeline parallel context
                        pre_backward()
                    b_schedule_plan.record_current_stream()

        return tmp

    def layer_pre_backward_dw():
        pass

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlaped_layers = min(f_num_layers, b_num_layers)

    for i in range(overlaped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        layer_pre_forward, layer_pre_backward, layer_pre_backward_dw = schedule_layer_1f1b(
            f_layer,
            b_layer,
            pre_forward=layer_pre_forward,
            pre_backward=layer_pre_backward,
            pre_backward_dw=layer_pre_backward_dw,
            f_context=f_context,
            b_context=b_context,
        )
        torch.cuda.nvtx.range_pop()

    # tail backward
    grad = layer_pre_backward()
    del layer_pre_backward
    with b_context:
        for i in range(overlaped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            tmp, grad, _ = schedule_layer_1f1b(None, b_layer, b_grad=grad)
            torch.cuda.nvtx.range_pop()

        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(grad)

    # tail forward
    f_input = layer_pre_forward()
    del layer_pre_forward
    with f_context:
        for i in range(overlaped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, tmp, _ = schedule_layer_1f1b(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)

    # output pp send receive, overlapped with attn backward
    if f_schedule_plan is not None and post_forward is not None:
        with f_context:
            f_schedule_plan.wait_current_stream()
            post_forward(f_input)

    # pp grad send / receive, overlapped with attn dw of cur micro-batch
    # and forward attn of next micro-batch
    if b_schedule_plan is not None and post_backward is not None:
        with b_context:
            b_schedule_plan.wait_current_stream()
            post_backward(grad)

    # The last wgrad of attention
    layer_pre_backward_dw()
    del layer_pre_backward_dw

    if f_schedule_plan:
        f_schedule_plan.wait_current_stream()
    if b_schedule_plan:
        b_schedule_plan.wait_current_stream()

    return f_input


def build_model_chunk_schedule_plan(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_params=None,
    packed_seq_params=None,
    extra_block_kwargs=None,
    runtime_gather_output: Optional[bool] = None,
):
    """Builds a schedule plan for a model chunk.

    This function creates a schedule plan for a model chunk, including
    preprocessing, transformer layers, and postprocessing.

    Args:
        model: The model to build a schedule plan for.
        input_ids: Input token IDs.
        position_ids: Position IDs.
        attention_mask: Attention mask.
        decoder_input: Decoder input tensor.
        labels: Labels for loss computation.
        inference_params: Parameters for inference.
        packed_seq_params: Parameters for packed sequences.
        extra_block_kwargs: Additional keyword arguments for blocks.
        runtime_gather_output: Whether to gather output at runtime.

    Returns:
        The model chunk schedule plan.
    """
    comp_stream = get_comp_stream()
    com_stream = get_com_stream()
    model_chunk_schedule_plan = ModelChunkSchedulePlan()
    event = model_chunk_schedule_plan.event
    state = model_chunk_schedule_plan.state
    # save for later use
    state.input_ids = input_ids
    state.position_ids = position_ids
    state.attention_mask = attention_mask
    state.decoder_input = decoder_input
    state.labels = labels
    state.inference_params = inference_params
    state.packed_seq_params = packed_seq_params
    state.extra_block_kwargs = extra_block_kwargs
    state.runtime_gather_output = runtime_gather_output
    state.context = None
    state.context_mask = None
    state.attention_bias = None

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(model, state, event, comp_stream)
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = TransformerLayerSchedulePlan(layer, event, state, comp_stream, com_stream)
        model_chunk_schedule_plan.add_layer(layer_plan)
    # build post process
    if model.post_process:
        model_chunk_schedule_plan.post_process = PostProcessNode(model, state, event, comp_stream)

    return model_chunk_schedule_plan
