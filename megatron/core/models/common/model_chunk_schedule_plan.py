# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.pipeline_parallel.utils import (
    AbstractSchedulePlan,
    NoopScheduleNode,
    get_comm_stream,
    get_comp_stream,
)
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


class ModelChunkState:
    """State shared across a model chunk.

    This class holds state that is shared between different components
    of a model chunk, such as input tensors, parameters, and configuration.
    """

    pass


class TransformerLayerSchedulePlan:
    """Schedule the executing plan of the nodes in a transformer/mtp layer.

    This class organizes the sub-modules of a transformer/mtp layer,
    including attention, post attention, MLP, dispatch, combine and
    mtp post process nodes.

    layer (TransformerLayerSchedulePlan)
    ├── pre_dispatch_computation (TransformerLayerNode):
    │     attention -> layernorm -> router -> dispatch preprocess
    ├── moe_dispatch (TransformerLayerNode): dispatch All2All
    ├── mlp (TransformerLayerNode): mlp module
    ├── moe_combine (TransformerLayerNode): combine All2All
    └── mtp_post_process (PostProcessNode): mtp post process

    Note that MTP layer has the same operation and execution order with TransformerLayer regarding
    moe_dispatch, mlp, moe_combine, but contains extra operations in
    pre_dispatch_computation and mtp_post_process:
    * mtp.pre_dispatch_computation wraps around transformer_layer.pre_dispatch_computation with
      extra norm, proj and embedding operations.
    * mtp.mtp_post_process contains output_layer, mtp loss operations, whereas
      transformer_layer.mtp_post_process is empty.
    """

    pre_dispatch_computation = None
    moe_dispatch = None
    mlp = None
    moe_combine = None
    mtp_post_process = None

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args={}):
        """Initializes a transformer layer schedule plan.

        Args:
            layer (TransformerLayer):
                split a transformer layer into multiple nodes for fine-grained scheduling.
            event (torch.cuda.Event):
                record CUDA event across multiple nodes on different streams for synchronization.
            chunk_state (ModelChunkState): model state shared in the model chunk.
            comp_stream (Callable): Func that returns CUDA stream for computation.
            comm_stream (Callable): Func that returns CUDA stream for communication.
            extra_args (dict): extra arguments for the layer.

        The event and chunk_state are binded to the TransformerModelChunkSchedulePlan
        and shared across all layers in the model chunk.
        """
        from megatron.core.models.common.utils import LayerState

        self.config = layer.config
        self.layer_state = LayerState()
        self.chunk_state = chunk_state
        self.layer = layer
        self.event = event
        self.comp_stream = comp_stream
        self.comm_stream = comm_stream

        # get callable nodes for transformer/mtp layer
        self._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

    def release_state(self):
        """Release reference, this helps avoid memory leak."""
        if hasattr(self, 'pre_dispatch_computation') and self.pre_dispatch_computation is not None:
            del self.pre_dispatch_computation
            self.pre_dispatch_computation = None
        if hasattr(self, 'moe_dispatch') and self.moe_dispatch is not None:
            del self.moe_dispatch
            self.moe_dispatch = None
        if hasattr(self, 'mlp') and self.mlp is not None:
            del self.mlp
            self.mlp = None
        if hasattr(self, 'moe_combine') and self.moe_combine is not None:
            del self.moe_combine
            self.moe_combine = None
        if hasattr(self, 'mtp_post_process') and self.mtp_post_process is not None:
            del self.mtp_post_process
            self.mtp_post_process = None
        if hasattr(self, 'layer_state') and self.layer_state is not None:
            del self.layer_state
            self.layer_state = None
        if hasattr(self, 'layer'):
            del self.layer

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        """
        Builds the callable nodes for the transformer/mtp layer:
            pre_dispatch_computation, moe_dispatch, mlp, moe_combine,
            and mtp_post_process.
        """
        from megatron.core.models.common.fine_grained_callables import (
            build_layer_callables,
            get_layer_moe_metadata,
        )
        from megatron.core.models.common.utils import TransformerLayerNode
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        fwd_callables, bwd_dw_callable_map = build_layer_callables(self.layer)
        is_moe, num_local_experts = get_layer_moe_metadata(self.layer)

        is_mtp = isinstance(self.layer, MultiTokenPredictionLayer)

        extra_args["config"] = self.layer.config
        extra_args["is_moe"] = is_moe
        extra_args["num_local_experts"] = num_local_experts
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = is_mtp

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
            pre_dispatch_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
        ) = fwd_callables

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.pre_dispatch_computation = create_node(
            comp_stream, pre_dispatch_module, "pre_dispatch_computation"
        )
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        if is_mtp:
            self.mtp_post_process = create_node(
                comp_stream, mtp_post_process_module, "mtp_post_process"
            )
        else:
            self.mtp_post_process = NoopScheduleNode()

    def set_fsdp_reshard_hooks(self, post_forward_hook, post_backward_hook):
        """Wire FSDP parameter release callbacks for the fine-grained overlap schedule.

        The EP overlap schedule bypasses the normal FSDP forward/backward hooks
        (registered on the FSDP unit module) because it calls sub-modules directly
        instead of going through TransformerLayer.forward(). This method attaches
        explicit release hooks to individual schedule nodes so that all-gathered
        parameters are freed at the right time.

        Args:
            post_forward_hook: Callable(module) that releases forward-pass params
                (bwd=False). Typically ``fsdp_wrapper.post_forward_release_module``.
            post_backward_hook: Callable(module) that releases backward-pass params
                (bwd=True). Typically ``fsdp_wrapper.post_backward_release_module``.
        """
        from megatron.core.models.hybrid.hybrid_block import HybridStack
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
        from megatron.core.transformer.transformer_layer import TransformerLayer

        assert isinstance(self.layer, (TransformerLayer, HybridStack, MultiTokenPredictionLayer)), (
            f"Megatron FSDP with EP Overlap only supports TransformerLayer, "
            f"HybridStack and MultiTokenPredictionLayer, "
            f"but got {type(self.layer).__name__}."
        )

        if isinstance(self.layer, (TransformerLayer, HybridStack)):
            hook_module = self.layer
        else:
            hook_module = self.layer.mtp_model_layer

        # After the last backward op (pre_dispatch_computation), release backward-pass params.
        self.pre_dispatch_computation.set_post_backward_hook(
            lambda: post_backward_hook(hook_module)
        )

        # Determine the last node in forward order.
        if isinstance(self.moe_combine, NoopScheduleNode):
            last_fwd_node = self.mlp
        else:
            last_fwd_node = self.moe_combine

        # After the last forward op, release forward-pass params.
        last_fwd_node.set_post_forward_hook(lambda: post_forward_hook(hook_module))

    def _iter_schedule_nodes(self):
        """Yield the real ScheduleNode instances of this layer (skips NoopScheduleNode)."""
        from megatron.core.pipeline_parallel.utils import ScheduleNode

        for node in (
            self.pre_dispatch_computation,
            self.moe_dispatch,
            self.mlp,
            self.moe_combine,
            self.mtp_post_process,
        ):
            if isinstance(node, ScheduleNode):
                yield node

    def set_forward_no_grad(self, no_grad: bool):
        """Toggle no-grad forward for all layer nodes (VPP-stage full recompute).

        The initial forward runs with ``no_grad=True`` (no autograd graph retained);
        the backward-time recompute runs with ``no_grad=False`` to rebuild the graph.
        """
        for node in self._iter_schedule_nodes():
            node.forward_no_grad = no_grad

    def reset_for_recompute(self):
        """Free the retained forward activations of this layer, keeping the nodes
        reusable for a recompute forward. Also clears the per-layer shared state."""
        for node in self._iter_schedule_nodes():
            node.reset_for_recompute()
        if getattr(self, 'layer_state', None) is not None:
            # Nodes hold a reference to this same layer_state object, so clear it
            # in place rather than replacing it.
            self.layer_state.__dict__.clear()

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

    @staticmethod
    def run(f_layer, b_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False):
        """Schedule one-forward-one-backward operations for a single transformer layer.

        This function interleaves forward and backward operations, overlapping the communications
        (dispatch or combine) of one with the computations (pre_dispatch or mlp) of the other
        to maximize parallelism and efficiency.

        When f_layer and b_layer are not None, forward and backward pass are overlapped as follows:
        comm_stream: combine_bwd       | dispatch_fwd->dispatch_bwd  | combine_fwd
        comp_stream: pre_dispatch_fwd  | mlp_bwd->mlp_bwd_dw->mlp_fwd| pre_dispatch_bwd
        For MTP, mtp_post_process_fwd is executed after the combine_fwd in the comp_stream,
        and mtp_post_process_bwd is executed before the combine_bwd in the comp_stream.

        Args:
            f_layer (TransformerLayerSchedulePlan): Forward layer (for current microbatch)
            b_layer (TransformerLayerSchedulePlan): Backward layer (for previous microbatch)
            f_input (Tensor): Input for forward computation
            b_grad (Tensor): Gradient for backward computation
            is_last_layer_in_bwd (bool):
                Whether the current layer is the last layer in the backward pass.

        Returns:
            Functions or values for next iteration's computation
        """

        if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.pre_dispatch_computation.forward(f_input)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input)

        if b_layer is not None:
            b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if b_layer is not None and b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.pre_dispatch_computation.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(f_input)

        if b_layer is not None and not b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.pre_dispatch_computation.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mtp_post_process.forward(f_input)

        # Delay the last pre_dispatch_computation wgrad in backward pass (wgrad
        # of the first layer) for overlapping with the p2p comm.
        if b_layer is not None and not is_last_layer_in_bwd:
            b_layer.pre_dispatch_computation.backward_dw()

        return f_input, b_grad


class TransformerModelChunkSchedulePlan(AbstractSchedulePlan):
    """Schedule the executing plan of the sub-modules in a model chunk sub-modules.

    This class organizes the computation nodes for a model chunk,
    including preprocessing, transformer layers, and postprocessing.

    TransformerModelChunkSchedulePlan
    ├── pre_process: PreProcessNode
    ├── layers: List[TransformerLayerSchedulePlan]
    │   ├── layer[0]: TransformerLayerSchedulePlan
    │   ├── layer[1]: TransformerLayerSchedulePlan
    │   └── ...
    └── post_process: PostProcessNode

    Subclasses can swap the per-layer schedule plan by overriding the
    ``LAYER_SCHEDULE_PLAN_CLASS`` class attribute (e.g. HybridStack uses a
    layer plan that understands grouped/inferred layer types). They can also
    swap the pre/post-process node classes via ``PRE_PROCESS_NODE_CLASS`` /
    ``POST_PROCESS_NODE_CLASS`` so each model owns its own embedding / output
    layer node implementations.
    """

    #: The TransformerLayerSchedulePlan-compatible class used to build per-layer
    #: schedule plans. Subclasses override this to inject a layer-plan variant.
    LAYER_SCHEDULE_PLAN_CLASS = None

    #: Pre/post-process node classes. Defaults below pull in the GPT-side
    #: ``PreProcessNode`` / ``PostProcessNode`` (which call ``GPTModel._preprocess`` /
    #: ``GPTModel._postprocess``). Subclasses set these to model-specific node
    #: classes so the node calls the right model's ``_preprocess`` /
    #: ``_postprocess`` methods.
    PRE_PROCESS_NODE_CLASS = None
    POST_PROCESS_NODE_CLASS = None

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
        padding_mask=None,
        *,
        output_processor: Optional[Callable[..., Tensor]] = None,
        output_processor_context: Optional[Any] = None,
    ):
        """Initialize the schedule plan of all Transformer layers' sub-modules.

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
            output_processor (Callable): Custom postprocess hook to run instead of the
                default logits/loss path.
            output_processor_context (Any): User-defined context object forwarded to
                `output_processor`.

        Returns:
            The model chunk schedule plan.
        """
        from megatron.core.models.common.utils import PostProcessNode, PreProcessNode

        pre_process_cls = self.PRE_PROCESS_NODE_CLASS or PreProcessNode
        post_process_cls = self.POST_PROCESS_NODE_CLASS or PostProcessNode

        self._model_chunk_state = ModelChunkState()
        self._transformer_layers = []
        self._event = torch.cuda.Event()
        self.pre_process = None
        self.post_process = None
        self.vp_stage = model.vp_stage

        # VPP-stage full recompute (EP A2A overlap): retain only the stage input
        # tensor across the forward->backward gap and recompute the whole stage
        # forward at backward time. See recompute_model_chunk_schedule_plan().
        self.recompute_vpp_stage = (
            model.config.recompute_granularity == 'full'
            and model.config.overlap_moe_expert_parallel_comm
        )
        # RNG states captured at the start of the stage forward, replayed by the
        # recompute so dropout / rng-forked ops reproduce the forward pass.
        self._rng_states = None
        # Snapshot of the mutable chunk-state fields (input_ids/position_ids/
        # padding_mask) captured before the forward and restored before recompute.
        self._recompute_state_snapshot = None

        # save the inputs of model.forward() to ModelChunkState
        self._model_chunk_state.input_ids = input_ids
        self._model_chunk_state.position_ids = position_ids
        self._model_chunk_state.attention_mask = attention_mask
        self._model_chunk_state.decoder_input = decoder_input
        self._model_chunk_state.labels = labels
        self._model_chunk_state.mtp_hidden_states = None
        self._model_chunk_state.loss_mask = loss_mask
        self._model_chunk_state.packed_seq_params = packed_seq_params
        self._model_chunk_state.padding_mask = padding_mask
        self._model_chunk_state.extra_block_kwargs = extra_block_kwargs
        self._model_chunk_state.runtime_gather_output = runtime_gather_output
        self._model_chunk_state.output_processor = output_processor
        self._model_chunk_state.output_processor_context = output_processor_context
        self._model_chunk_state.model = model
        self._model_chunk_state.context = None
        self._model_chunk_state.context_mask = None
        self._model_chunk_state.attention_bias = None

        # build preprocess
        self.pre_process = pre_process_cls(
            model, self._model_chunk_state, self._event, get_comp_stream
        )

        # build layer schedule plan for each layer.
        # The methods to obtain layers are different for MTP so we need the other build plan for
        # MTP. Also, this can help annotate MTP layer so that it can know where MTP is.
        self._build_layer_schedule_plan(model.decoder, get_comp_stream, get_comm_stream)
        self._build_layer_schedule_plan(
            getattr(model, "mtp", None), get_comp_stream, get_comm_stream
        )

        # build post process
        if model.post_process:
            self.post_process = post_process_cls(
                model, self._model_chunk_state, self._event, get_comp_stream
            )

        # For VPP-stage full recompute, the initial forward of every transformer
        # layer runs under no_grad (no activation retained); pre_process and
        # post_process keep their graphs (they are not recomputed).
        if self.recompute_vpp_stage:
            for layer_plan in self._transformer_layers:
                layer_plan.set_forward_no_grad(True)

    def _build_layer_schedule_plan(self, module, comp_stream, comm_stream):
        if module is None:
            return
        plan_cls = self.LAYER_SCHEDULE_PLAN_CLASS or TransformerLayerSchedulePlan
        num_layers = len(module.layers)
        for layer_idx in range(num_layers):
            extra_args = self._extra_args_for_layer(module, layer_idx, num_layers)
            layer_plan = plan_cls(
                module.layers[layer_idx],
                self.event,
                self.state,
                comp_stream,
                comm_stream,
                extra_args,
            )
            self._transformer_layers.append(layer_plan)

    def _extra_args_for_layer(self, module, layer_idx, num_layers):
        """Per-layer ``extra_args`` dict passed to the layer plan constructor.

        Subclasses extend this hook to thread additional metadata (e.g. hybrid
        layer-type symbols) without overriding ``_build_layer_schedule_plan``.
        """
        return {"is_first_layer": layer_idx == 0, "is_last_layer": layer_idx == num_layers - 1}

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

    def get_layer(self, i):
        """Gets the transformer layer at the specified index."""
        assert i < self.num_layers()
        return self._transformer_layers[i]

    def pop_layer(self):
        """Pops the transformer layer in FILO order."""
        return self._transformer_layers.pop()

    def num_layers(self):
        """Gets the number of transformer layers."""
        return len(self._transformer_layers)

    @property
    def state(self):
        """Gets the model chunk state."""
        return self._model_chunk_state

    def release_state(self):
        """Release reference, this helps avoid memory leak."""
        self._model_chunk_state.model = None
        self.pre_process.model_chunk_state = None
        self.pre_process = None

        if self.post_process is not None:
            self.post_process.model_chunk_state = None
            self.post_process = None

    def snapshot_rng_for_recompute(self):
        """Capture the RNG states and the mutable chunk-state fields at the start of
        the stage forward so the backward-time recompute can replay an identical
        forward.

        The MTP path mutates ``chunk_state.input_ids`` / ``position_ids`` /
        ``padding_mask`` in place (it rolls the tokens by one and writes them back),
        so these must be restored to their pre-forward values before recompute,
        otherwise the recompute would roll them a second time.
        """
        if not self.recompute_vpp_stage:
            return
        from megatron.core.tensor_parallel.random import _get_all_rng_states

        self._rng_states = _get_all_rng_states()
        cs = self._model_chunk_state
        self._recompute_state_snapshot = {
            'input_ids': cs.input_ids,
            'position_ids': cs.position_ids,
            'padding_mask': cs.padding_mask,
        }

    def release_layer_activations(self):
        """Free every transformer layer's retained forward activations after the
        initial forward, keeping only the stage input tensor (pre_process output).

        The layer nodes remain reusable; recompute_model_chunk_schedule_plan() rebuilds
        their forward state before the backward pass.
        """
        if not self.recompute_vpp_stage:
            return
        for layer_plan in self._transformer_layers:
            layer_plan.reset_for_recompute()

    def recompute_model_chunk_schedule_plan(self):
        """Recompute the whole VPP-stage (model chunk) forward from the retained
        stage input tensor, rebuilding every layer ScheduleNode's forward state so
        the subsequent backward can run.

        This is called at the start of the stage backward. The A2A communications
        issued here are exposed (not overlapped) by design; the normal forward and
        normal backward A2A overlap is preserved elsewhere in the schedule.

        The mapping from the recomputed activations back to the per-submodule
        ScheduleNodes is implicit: the same node objects are re-run in the same
        forward order, so each node repopulates its own ``inputs`` / ``output`` /
        ``detached`` / ``layer_state`` slots exactly as in the initial forward.
        """
        if not self.recompute_vpp_stage:
            return
        from megatron.core.tensor_parallel.random import _fork_rng, _set_all_rng_states

        assert self.pre_process is not None and self.pre_process.output is not None, (
            "VPP-stage full recompute requires the retained stage input tensor "
            "(pre_process output), but it is missing."
        )
        stage_input = self.pre_process.output

        # Restore the mutable chunk-state fields (MTP rolls input_ids/position_ids/
        # padding_mask in place during the forward) so the recompute starts from the
        # same inputs as the initial forward.
        if self._recompute_state_snapshot is not None:
            cs = self._model_chunk_state
            cs.input_ids = self._recompute_state_snapshot['input_ids']
            cs.position_ids = self._recompute_state_snapshot['position_ids']
            cs.padding_mask = self._recompute_state_snapshot['padding_mask']
            cs.mtp_hidden_states = None
            self._recompute_state_snapshot = None

        # Enable grad on the layer nodes' forward so the recompute builds the graph.
        for layer_plan in self._transformer_layers:
            layer_plan.set_forward_no_grad(False)

        # Replay the forward RNG stream so dropout / rng-forked ops reproduce the
        # initial forward. _fork_rng() restores the ambient RNG state on exit.
        with _fork_rng():
            if self._rng_states is not None:
                _set_all_rng_states(*self._rng_states)
            f_input = stage_input
            for i, layer_plan in enumerate(self._transformer_layers):
                nvtx_msg = f"recompute_layer_{i}"
                nvtx_range_push(nvtx_msg)
                f_input, _ = TransformerLayerSchedulePlan.run(layer_plan, None, f_input=f_input)
                nvtx_range_pop(nvtx_msg)
        self._rng_states = None

    @staticmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Model Chunk level 1f1b fine-grained scheduler.

        This function schedules the forward and backward passes for a model chunk,
        which interleaves forward and backward function of multiple Transformer layers
        within a model chunk, and this is needed to overlap the submodules between the individual
        forward and backward functions.

        Assume there are 4 layers in the given model chunk:
        Phase 0: p2p_comm_sync -> forward_preprocess -> p2p_comm_sync -> backward_postprocess
        Phase 1: forward_layer[0] + backward_layer[3], overlapped execution by schedule_layer_1f1b
        Phase 2: forward_layer[1] + backward_layer[2], overlapped execution by schedule_layer_1f1b
        Phase 3: forward_layer[2] + backward_layer[1], overlapped execution by schedule_layer_1f1b
        Phase 4: forward_layer[3] + backward_layer[0], overlapped execution by schedule_layer_1f1b
        Phase 5: send_forward_recv_backward -> send_backward_recv_forward
        Phase 6: backward_dw of the first layer -> forward_postprocess -> backward_preprocess

        Args:
            f_schedule_plan (TransformerModelChunkSchedulePlan): The forward schedule plan
            b_schedule_plan (TransformerModelChunkSchedulePlan): The backward schedule plan
            b_grad (Tensor or None): The gradient of the loss function
            pre_forward (callable or None): The function to call before the forward pass
            pre_backward (callable or None): The function to call before the backward pass
            post_forward (callable or None): The function to call after the forward pass
            post_backward (callable or None): The function to call after the backward pass
        Returns:
            The output of the forward pass.
        """
        f_input = None
        if f_schedule_plan:
            # pp output send/receive sync
            if pre_forward is not None:
                pre_forward(f_schedule_plan.vp_stage)
            f_schedule_plan.record_current_stream()
            f_input = f_schedule_plan.pre_process.forward()
            # Capture RNG right after pre_process so the stage recompute (at backward
            # time) replays the same forward random stream. No-op unless recompute is on.
            f_schedule_plan.snapshot_rng_for_recompute()

        if b_schedule_plan:
            b_schedule_plan.record_current_stream()
            assert b_grad is not None
            if pre_backward is not None:
                pre_backward(b_schedule_plan.vp_stage)
                b_schedule_plan.record_current_stream()

            # VPP-stage full recompute: rebuild the whole stage's layer forward
            # graphs from the retained stage input before any layer backward runs.
            # The A2A issued here is exposed (not overlapped) by design. No-op unless
            # recompute is on.
            b_schedule_plan.recompute_model_chunk_schedule_plan()

            if b_schedule_plan.post_process is not None:
                b_grad = b_schedule_plan.post_process.backward(b_grad)

        f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
        b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
        overlapped_layers = min(f_num_layers, b_num_layers)

        f_layer = b_layer = None
        # combined forward and backward pass for overlapped layers
        for i in range(overlapped_layers):
            f_layer = f_schedule_plan.get_layer(i)
            b_layer = b_schedule_plan.pop_layer()
            nvtx_msg = f"layer_{i}f-layer_{b_schedule_plan.num_layers()}b"
            nvtx_range_push(nvtx_msg)
            f_input, b_grad = TransformerLayerSchedulePlan.run(
                f_layer,
                b_layer,
                f_input=f_input,
                b_grad=b_grad,
                is_last_layer_in_bwd=(i == b_num_layers - 1),
            )
            if i < b_num_layers - 1:
                b_layer.release_state()
            nvtx_range_pop(nvtx_msg)

        # backward pass for the remaining layers
        for i in range(overlapped_layers, b_num_layers):
            b_layer = b_schedule_plan.pop_layer()
            nvtx_msg = f"layer_{b_schedule_plan.num_layers()}b"
            nvtx_range_push(nvtx_msg)
            _, b_grad = TransformerLayerSchedulePlan.run(
                None, b_layer, b_grad=b_grad, is_last_layer_in_bwd=(i == b_num_layers - 1)
            )
            if i < b_num_layers - 1:
                b_layer.release_state()
            nvtx_range_pop(nvtx_msg)

        # forward pass for the remaining layers
        for i in range(overlapped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            nvtx_msg = f"layer_{i}f"
            nvtx_range_push(nvtx_msg)
            f_input, _ = TransformerLayerSchedulePlan.run(f_layer, None, f_input=f_input)
            nvtx_range_pop(nvtx_msg)

        if f_schedule_plan is not None and post_forward is not None:
            # post_forward()/send_forward_recv_forward() is running in the communication stream,
            # so the p2p comm could be overlapped with the pre_dispatch backward
            with torch.cuda.stream(get_comm_stream()):
                f_schedule_plan.wait_current_stream()
                post_forward(f_input, f_schedule_plan.vp_stage)

        # post_backward()/send_backward_recv_backward() is running in the computation stream,
        # so the p2p comm could be overlapped with the wgrad of pre_dispatch backward
        if b_schedule_plan is not None and post_backward is not None:
            b_schedule_plan.wait_current_stream()
            post_backward(b_grad, b_schedule_plan.vp_stage)

        # Delay the last pre_dispatch_computation wgrad in backward pass (wgrad
        # of the first layer) for overlapping with the p2p comm.
        if b_num_layers > 0:
            assert b_layer is not None
            b_layer.pre_dispatch_computation.backward_dw()
            b_layer.release_state()

        # post process forward
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            if (
                f_schedule_plan.recompute_vpp_stage
                and f_input is not None
                and not f_input.requires_grad
            ):
                # The stage forward ran under no_grad, so the stage output is a plain
                # value tensor. post_process needs a grad-requiring leaf so its backward
                # produces the grad seed that feeds the recomputed last-layer backward.
                f_input.requires_grad_(True)
            f_input = f_schedule_plan.post_process.forward(f_input)
        # pre process backward
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

        # Free the forward stage's layer activations now that its forward output has
        # been consumed (PP send / post_process). Only the stage input tensor is kept
        # for the backward-time recompute. No-op unless recompute is on.
        if f_schedule_plan is not None:
            f_schedule_plan.release_layer_activations()

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()
            # Release reference as early as possible, this helps avoid memory leak.
            b_schedule_plan.release_state()

        return f_input
