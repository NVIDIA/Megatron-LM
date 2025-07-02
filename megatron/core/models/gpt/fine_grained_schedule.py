# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.fine_grained_callables import (
    PostProcessNode,
    PreProcessNode,
    TransformerLayerNode,
    TransformerLayerState,
    build_layer_callables,
)
from megatron.core.pipeline_parallel.utils import (
    AbstractSchedulePlan,
    FakeScheduleNode,
    get_com_stream,
    get_comp_stream,
)


class ModelChunkState:
    """State shared across a model chunk.

    This class holds state that is shared between different components
    of a model chunk, such as input tensors, parameters, and configuration.
    """

    pass


class LayerSchedulePlan:
    """Schedule plan for a transformer layer.

    This class organizes the computation nodes for a transformer layer,
    including attention, post attention, MLP, dispatch, and combine nodes.

    layer (LayerSchedulePlan)
    ├── attn (TransformerLayerNode): attention module
    ├── post_attn (TransformerLayerNode): layernorm -> router -> dispatch preprocess
    ├── moe_dispatch (TransformerLayerNode): dispatch All2All
    ├── mlp (TransformerLayerNode): mlp module
    ├── moe_combine (TransformerLayerNode): combine All2All
    └── post_process (PostProcessNode): post process
    """

    attn = None
    post_attn = None
    moe_dispatch = None
    mlp = None
    moe_combine = None
    post_process = None

    def __init__(self, layer, event, chunk_state, comp_stream, com_stream, extra_args={}):
        """Initializes a transformer layer schedule plan.

        Args:
            layer (TransformerLayer):
                split a transformer layer into multiple nodes for fine-grained scheduling.
            event (torch.cuda.Event):
                record CUDA event across multiple nodes on different streams for synchronization.
            chunk_state (ModelChunkState): model state shared in the model chunk.
            comp_stream (torch.cuda.Stream): CUDA stream for computation.
            com_stream (torch.cuda.Stream): CUDA stream for communication.
            extra_args (dict): extra arguments for the layer.

        The event and chunk_state are binded to the ModelChunkSchedulePlan
        and shared across all layers in the model chunk.
        """
        self.layer_state = TransformerLayerState()
        self.chunk_state = chunk_state
        self.layer = layer
        self.event = event
        self.comp_stream = comp_stream
        self.com_stream = com_stream

        # get callable nodes for transformer/mtp layer
        self._build_callable_nodes(event, comp_stream, com_stream, extra_args)

    def _build_callable_nodes(self, event, comp_stream, com_stream, extra_args):
        """
        Builds the callable nodes for the transformer/mtp layer:
            attn, post_attn, mlp, dispatch, combine, and post_process.
        """
        from megatron.core.transformer.moe.moe_layer import MoELayer

        # build the forward and backward callables for the transformer/mtp layer
        fwd_callables, bwd_dw_callable_map = build_layer_callables(self.layer)

        # get flags for latter use
        is_moe = isinstance(self.layer.mlp, MoELayer)
        enable_deepep = self.layer.config.moe_enable_deepep
        extra_args["enable_deepep"] = enable_deepep
        extra_args["is_moe"] = is_moe

        # wrapper to help create TransformerLayerNode
        def create_node(stream, module, name):
            bwd_dw_callables = bwd_dw_callable_map.get(name, None)
            return TransformerLayerNode(
                stream,
                event,
                self.layer_state,
                self.chunk_state,
                module,
                name=name,
                bwd_dw_callables=bwd_dw_callables,
                extra_args=extra_args,
            )

        (
            attn_module,
            post_attn_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            post_process_module,
        ) = fwd_callables

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.attn = create_node(comp_stream, attn_module, "attn")
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.post_attn = create_node(comp_stream, post_attn_module, "post_attn")
            self.moe_dispatch = create_node(com_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(com_stream, moe_combine_module, "moe_combine")
        else:
            self.post_attn = FakeScheduleNode()
            self.moe_dispatch = FakeScheduleNode()
            self.moe_combine = FakeScheduleNode()

        self.post_process = FakeScheduleNode()

    def get_fp8_context(self):
        """
        Get the fp8 context for the transformer layer.
        """
        use_inner_fp8_context = (
            self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        )
        return (
            get_fp8_context(self.layer.config, self.layer.layer_number - 1)
            if use_inner_fp8_context
            else nullcontext()
        )
    
    @classmethod
    def run(
        cls,
        f_layer,
        b_layer,
        f_input=None,
        b_grad=None,
        f_context=None,
        b_context=None,
        is_last_layer_in_bwd=False,
    ):
        """Schedule one-forward-one-backward operations for a single transformer layer.

        This function interleaves forward and backward operations to maximize
        parallelism and efficiency.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            f_context (VppContextManager or None): The VppContextManager for the forward pass.
            b_context (VppContextManager or None): The VppContextManager for the backward pass
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """
        f_context = f_context if f_context is not None else contextlib.nullcontext()
        b_context = b_context if b_context is not None else contextlib.nullcontext()

        if b_layer is not None:
            with b_context:
                b_grad = b_layer.post_process.backward(b_grad)
                b_grad = b_layer.moe_combine.backward(b_grad)

        if f_layer is not None:
            with f_context and f_layer.get_fp8_context():
                f_input = f_layer.attn.forward(f_input)
                f_input = f_layer.post_attn.forward(f_input)

        if b_layer is not None:
            with b_context:
                b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with f_context and f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input)

        if b_layer is not None:
            with b_context:
                b_layer.mlp.backward_dw()
                b_grad = b_layer.moe_dispatch.backward(b_grad)

        if f_layer is not None:
            with f_context and f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if f_layer is not None:
            with f_context and f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(f_input)
                f_input = f_layer.post_process.forward(f_input)

        if b_layer is not None:
            with b_context:
                b_grad = b_layer.post_attn.backward(b_grad)
                b_grad = b_layer.attn.backward(b_grad)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_layer is not None and not is_last_layer_in_bwd:
            with b_context:
                b_layer.attn.backward_dw()

        return f_input, b_grad


class ModelChunkSchedulePlan(AbstractSchedulePlan):
    """Schedule plan for a model chunk.

    This class organizes the computation nodes for a model chunk,
    including preprocessing, transformer layers, and postprocessing.

    ModelChunkSchedulePlan
    ├── pre_process: PreProcessNode
    ├── layers: List[LayerSchedulePlan]
    │   ├── layer[0]: LayerSchedulePlan
    │   ├── layer[1]: LayerSchedulePlan
    │   └── ...
    └── post_process: PostProcessNode
    """

    def __init__(
        self,
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
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
            packed_seq_params: Parameters for packed sequences.
            extra_block_kwargs: Additional keyword arguments for blocks.
            runtime_gather_output: Whether to gather output at runtime.
            loss_mask (torch.Tensor): Used to mask out some portions of the loss

        Returns:
            The model chunk schedule plan.
        """
        self._model_chunk_state = ModelChunkState()
        self._transformer_layers = []
        self._event = torch.cuda.Event()
        self._pre_process = None
        self._post_process = None

        comp_stream = get_comp_stream()
        com_stream = get_com_stream()

        # save the inputs of model.forward() to ModelChunkState
        self._model_chunk_state.input_ids = input_ids
        self._model_chunk_state.position_ids = position_ids
        self._model_chunk_state.attention_mask = attention_mask
        self._model_chunk_state.decoder_input = decoder_input
        self._model_chunk_state.labels = labels
        self._model_chunk_state.loss_mask = loss_mask
        self._model_chunk_state.packed_seq_params = packed_seq_params
        self._model_chunk_state.extra_block_kwargs = extra_block_kwargs
        self._model_chunk_state.runtime_gather_output = runtime_gather_output

        transformer_num_layers = model.decoder.num_layers_per_pipeline_rank
        # build preprocess
        self._pre_process = PreProcessNode(model, self._model_chunk_state, self._event, comp_stream)
        # build layer schedule plan for each layer
        for layer_idx in range(transformer_num_layers):
            layer = model.decoder._get_layer(layer_idx)
            layer_plan = LayerSchedulePlan(layer, self._event, self._model_chunk_state, comp_stream, com_stream)
            self._transformer_layers.append(layer_plan)
        # build post process
        if model.post_process:
            self._post_process = PostProcessNode(model, self._model_chunk_state, self._event, comp_stream)

    @classmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
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
            b_grad=b_grad,
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

    @property
    def post_process(self):
        """Gets the postprocessing node."""
        return self._post_process

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

    def release_state(self):
        """Release reference, this helps avoid memory leak."""
        self._pre_process.model_chunk_state = None
        self._pre_process = None

        if self._post_process is not None:
            self._post_process.model_chunk_state = None
            self._post_process = None

    @classmethod
    def run(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Model Chunk level 1f1b fine-grained scheduler.

        This function schedules the forward and backward passes for a model chunk,
        which interleaves forward and backward operations across multiple layers
        to maximize parallelism and efficiency.

        Assume there are 4 layers in the given model chunk:
        Phase 0: p2p_comm_sync -> forward_preprocess -> p2p_comm_sync -> backward_postprocess
        Phase 1: forward_layer[0] + backward_layer[3], overlapped execution by schedule_layer_1f1b
        Phase 2: forward_layer[1] + backward_layer[2], overlapped execution by schedule_layer_1f1b
        Phase 3: forward_layer[2] + backward_layer[1], overlapped execution by schedule_layer_1f1b
        Phase 4: forward_layer[3] + backward_layer[0], overlapped execution by schedule_layer_1f1b
        Phase 5: send_forward_recv_backward -> send_backward_recv_forward
        Phase 6: backward_dw of the first layer -> forward_postprocess -> backward_preprocess

        Args:
            f_schedule_plan (ModelChunkSchedulePlan): The forward schedule plan
            b_schedule_plan (ModelChunkSchedulePlan): The backward schedule plan
            b_grad (Tensor or None): The gradient of the loss function
            f_context (VppContextManager or None): The VppContextManager for the forward pass
            b_context (VppContextManager or None): The VppContextManager for the backward pass
            pre_forward (callable or None): The function to call before the forward pass
            pre_backward (callable or None): The function to call before the backward pass
            post_forward (callable or None): The function to call after the forward pass
            post_backward (callable or None): The function to call after the backward pass
        Returns:
            The output of the forward pass.
        """
        f_context = f_context if f_context is not None else contextlib.nullcontext()
        b_context = b_context if b_context is not None else contextlib.nullcontext()

        f_input = None
        if f_schedule_plan:
            # pp output send/receive sync
            if pre_forward is not None:
                with f_context as ctx:  # virtual pipeline parallel context
                    pre_forward(ctx.vpp_rank)
            f_schedule_plan.record_current_stream()
            f_input = f_schedule_plan.pre_process.forward()

        if b_schedule_plan:
            b_schedule_plan.record_current_stream()
            assert b_grad is not None

            if pre_backward is not None:
                # If the post_process node is FakeScheduleNode, it means the last node of
                # the backward pass is combine node, which is running in the communication stream.
                if isinstance(b_schedule_plan.post_process, FakeScheduleNode):
                    stream = get_com_stream()
                else:
                    stream = get_comp_stream()
                with torch.cuda.stream(stream):
                    b_schedule_plan.wait_current_stream()
                    with b_context as ctx:  # virtual pipeline parallel context
                        pre_backward(ctx.vpp_rank)
                    b_schedule_plan.record_current_stream()

            if b_schedule_plan.post_process is not None:
                with b_context:  # virtual pipeline parallel context
                    b_grad = b_schedule_plan.post_process.backward(b_grad)
                    # b_schedule_plan.post_process.backward_dw()

        f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
        b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
        overlaped_layers = min(f_num_layers, b_num_layers)

        for i in range(overlaped_layers):
            f_layer = f_schedule_plan.get_layer(i)
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
            f_input, b_grad = LayerSchedulePlan.run(
                f_layer,
                b_layer,
                f_input=f_input,
                b_grad=b_grad,
                f_context=f_context,
                b_context=b_context,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
            )
            torch.cuda.nvtx.range_pop()

        # backward pass for the remaining layers
        with b_context:
            for i in range(overlaped_layers, b_num_layers):
                b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
                torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
                _, b_grad = LayerSchedulePlan.run(
                    None, b_layer, b_grad=b_grad, is_last_layer_in_bwd=(i == b_num_layers - 1)
                )
                torch.cuda.nvtx.range_pop()

        # forward pass for the remaining layers
        with f_context:
            for i in range(overlaped_layers, f_num_layers):
                f_layer = f_schedule_plan.get_layer(i)
                torch.cuda.nvtx.range_push(f"layer_{i}f")
                f_input, _ = LayerSchedulePlan.run(f_layer, None, f_input=f_input)
                torch.cuda.nvtx.range_pop()

        if f_schedule_plan is not None and post_forward is not None:
            with f_context as ctx:
                # The last submodule is running in the communication stream,
                # so the p2p comm could be overlapped with the attn backward
                with torch.cuda.stream(get_com_stream()):
                    f_schedule_plan.wait_current_stream()
                    post_forward(f_input, ctx.vpp_rank)

        # pp grad send / receive, overlapped with attn dw of cur micro-batch
        # and forward attn of next micro-batch
        if b_schedule_plan is not None and post_backward is not None:
            with b_context as ctx:
                b_schedule_plan.wait_current_stream()
                post_backward(b_grad, ctx.vpp_rank)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_num_layers > 0:
            with b_context:
                b_schedule_plan.get_layer(0).attn.backward_dw()

        with f_context:
            if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
                f_input = f_schedule_plan.post_process.forward(f_input)
        with b_context:
            if b_schedule_plan is not None:
                b_schedule_plan.pre_process.backward(b_grad)

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()

        # Release reference as early as possible, this helps avoid memory leak.
        if b_schedule_plan is not None:
            b_schedule_plan.release_state()

        return f_input

