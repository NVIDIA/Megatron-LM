# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, List, Optional

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.pipeline_parallel.utils import (
    AbstractSchedulePlan,
    NoopScheduleNode,
    ScheduleNode,
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


# ModelChunkState fields the layer forwards mutate in place; each segment snapshots them
# so its backward-time replay sees the pre-segment values instead of end-of-chunk ones.
_MUTABLE_CHUNK_STATE_FIELDS = (
    "input_ids",
    "position_ids",
    "padding_mask",
    "mtp_hidden_states",
    "mhc_multistream",
)


def _copy_chunk_state_value(value: Any) -> Any:
    """Copy list-valued chunk state so a later in-place append cannot corrupt a snapshot."""
    return list(value) if isinstance(value, list) else value


class RecomputeSegment:
    """A contiguous group of layers recomputed as one unit.

    The initial forward runs under ``no_grad`` and is replayed with grad enabled at the
    start of the segment's own backward. What survives the forward->backward gap is the
    segment's input tensor plus the ``_MUTABLE_CHUNK_STATE_FIELDS`` snapshot - which for an
    MTP segment pins the pre-contraction mHC bridge tensor, since the carrier hand-off
    needs it. Layers left eager under ``block`` keep their activations throughout. The A2A
    issued by a replay is exposed by design; normal fwd/bwd A2A overlap is preserved.

    ``start_index`` is the segment's first layer index within the chunk, for NVTX labels.
    """

    def __init__(
        self,
        layers: List["TransformerLayerSchedulePlan"],
        chunk_state: ModelChunkState,
        start_index: int,
    ):
        assert len(layers) > 0, "a recompute segment must contain at least one layer"
        self.layers = layers
        self.chunk_state = chunk_state
        self.start_index = start_index
        # Captured at the start of the segment's initial forward.
        self.input_tensor = None
        self.state_snapshot = None
        self.rng_states = None
        for layer in layers:
            layer.recompute_segment = self
            # The initial forward retains no autograd graph for these layers.
            layer.set_forward_no_grad(True)

    def capture(self, layer: "TransformerLayerSchedulePlan", f_input: Tensor) -> None:
        """When the head layer starts its forward, snapshot what the replay needs:
        input tensor, mutable chunk state, RNG."""
        if layer is not self.layers[0]:
            return
        from megatron.core.tensor_parallel.random import _get_all_rng_states

        self.input_tensor = f_input
        cs = self.chunk_state
        self.state_snapshot = {
            name: _copy_chunk_state_value(getattr(cs, name)) for name in _MUTABLE_CHUNK_STATE_FIELDS
        }
        self.rng_states = _get_all_rng_states()

    def release_input(self, layer: "TransformerLayerSchedulePlan") -> None:
        """Drop the retained input once the head layer has finished its backward."""
        if layer is not self.layers[0]:
            return
        self.input_tensor = None
        self.state_snapshot = None

    def recompute(self, layer: "TransformerLayerSchedulePlan") -> None:
        """Replay the segment with grad enabled when its tail layer enters the backward.

        Must run before any of that layer's nodes backwards: ``mtp_post_process`` is in the
        recompute scope, so its graph does not exist until the replay has run. Re-running
        the same nodes in the same order repopulates each node's ``inputs`` / ``output`` /
        ``detached`` / ``layer_state`` as in the initial forward.
        """
        if layer is not self.layers[-1]:
            return
        from megatron.core.tensor_parallel.random import _fork_rng, _set_all_rng_states

        assert self.input_tensor is not None, (
            "layer-level full recompute requires the retained segment input tensor, "
            "but it is missing."
        )
        cs = self.chunk_state

        for name, value in self.state_snapshot.items():
            setattr(cs, name, _copy_chunk_state_value(value))

        # mhc_multistream is a detach bridge from the last decoder layer to the MTP layer.
        # Backward runs in reverse, so MTP is replayed and backwarded before the producer
        # replay installs a fresh leaf; park the old leaf so that replay can carry its
        # gradient over (mhc_multistream itself is cleared by the replayed MTP postprocess).
        if cs.mhc_multistream is not None:
            if not cs.mhc_multistream.requires_grad:
                cs.mhc_multistream.requires_grad_(True)
            cs.mhc_grad_carrier = cs.mhc_multistream

        # The replayed nodes read inputs[i].grad to reach the previous segment, so the
        # retained input has to be a grad-tracking leaf.
        segment_input = self.input_tensor
        if not segment_input.requires_grad:
            segment_input.requires_grad_(True)

        for seg_layer in self.layers:
            seg_layer.set_forward_no_grad(False)

        # Replay the forward RNG stream so rng-forked ops reproduce the initial forward.
        with _fork_rng():
            _set_all_rng_states(*self.rng_states)
            f_input = segment_input
            for i, seg_layer in enumerate(self.layers):
                nvtx_msg = f"recompute_layer_{self.start_index + i}"
                nvtx_range_push(nvtx_msg)
                f_input = seg_layer.recompute_forward(f_input)
                nvtx_range_pop(nvtx_msg)
        self.rng_states = None

        # This replay re-produced the bridge leaf: hand it the parked gradient.
        new_mhc_multistream = cs.mhc_multistream
        carrier = cs.mhc_grad_carrier
        if (
            new_mhc_multistream is not None
            and carrier is not None
            and new_mhc_multistream is not carrier
        ):
            new_mhc_multistream.grad = carrier.grad
            # Only cleared here: the carrier has to survive from the consumer's replay
            # until the producer's, so there is no earlier point at which it is dead.
            cs.mhc_grad_carrier = None


class TransformerLayerSchedulePlan:
    """Schedule the execution plan for nodes in a transformer or MTP layer.

    This class organizes the submodules of a transformer or MTP layer, including attention,
    MLP, MoE dispatch and combine, optional mHC recomputation, and MTP post-processing nodes.

    layer (TransformerLayerSchedulePlan)
    ├── attn (TransformerLayerNode): attention -> layernorm -> router -> dispatch preprocess
    ├── moe_dispatch (TransformerLayerNode): dispatch All2All
    ├── mlp (TransformerLayerNode): mlp module
    ├── moe_combine (TransformerLayerNode): combine All2All
    ├── mhc_post (TransformerLayerNode): MLP-side mHC post-processing, on the
    │   compute stream. It belongs here rather than folded into moe_combine:
    │   running it in that communication-stream node let a recompute subgraph be
    │   allocated on one stream and read from another, which the caching
    │   allocator cannot track.
    ├── mhc_recompute (ScheduleNode): optional explicit replay before mHC backward
    └── mtp_post_process (PostProcessNode): mtp post process

    Note that MTP layer has the same operation and execution order with TransformerLayer regarding
    moe_dispatch, mlp, moe_combine, but contains extra operations in attn and mtp_post_process:
    * mtp.attn wraps around transformer_layer.attn with extra norm, proj and embedding operations.
    * mtp.mtp_post_process contains output_layer, mtp loss operations, whereas
      transformer_layer.mtp_post_process is empty.
    """

    attn = None
    moe_dispatch = None
    mlp = None
    moe_combine = None
    mhc_post = None
    mhc_recompute = None
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
        from megatron.core.models.gpt.fine_grained_callables import TransformerLayerState

        self.config = layer.config
        self.layer_state = TransformerLayerState()
        self.chunk_state = chunk_state
        self.layer = layer
        self.event = event
        self.comp_stream = comp_stream
        self.comm_stream = comm_stream

        # Set by _build_recompute_segments(); None means this layer keeps its graph.
        self.recompute_segment = None

        # get callable nodes for transformer/mtp layer
        self._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

    def release_state(self):
        """Release reference, this helps avoid memory leak."""
        self.recompute_segment = None
        if hasattr(self, 'attn') and self.attn is not None:
            del self.attn
            self.attn = None
        if hasattr(self, 'moe_dispatch') and self.moe_dispatch is not None:
            del self.moe_dispatch
            self.moe_dispatch = None
        if hasattr(self, 'mlp') and self.mlp is not None:
            del self.mlp
            self.mlp = None
        if hasattr(self, 'moe_combine') and self.moe_combine is not None:
            del self.moe_combine
            self.moe_combine = None
        if hasattr(self, 'mhc_post') and self.mhc_post is not None:
            del self.mhc_post
            self.mhc_post = None
        if hasattr(self, 'mhc_recompute') and self.mhc_recompute is not None:
            del self.mhc_recompute
            self.mhc_recompute = None
        if hasattr(self, 'mtp_post_process') and self.mtp_post_process is not None:
            del self.mtp_post_process
            self.mtp_post_process = None
        if hasattr(self, 'layer_state') and self.layer_state is not None:
            del self.layer_state
            self.layer_state = None
        if hasattr(self, 'layer'):
            # The schedule installs _mhc_recompute_manager on the layer directly,
            # bypassing TransformerLayer.__call__, which is what would otherwise
            # refresh or clear it each forward. Left in place it pins this
            # iteration's MHCCheckpointManager -- and every saved tensor it still
            # holds -- on the module, and a later replay that arrives without a
            # fresh assignment would bind arena slots against an already
            # recomputed checkpoint set.
            #
            # Clear it on the object it was installed on: build_mtp_layer_callables
            # builds over layer.mtp_model_layer, so for an MTP plan the manager
            # lives on the inner transformer layer, not on the wrapper.
            from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

            if isinstance(self.layer, MultiTokenPredictionLayer):
                self.layer.mtp_model_layer._mhc_recompute_manager = None
            else:
                self.layer._mhc_recompute_manager = None
            del self.layer

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        """
        Builds the callable nodes for the transformer/mtp layer:
            attn, mlp, moe_dispatch, moe_combine, and mtp_post_process.
        """
        from megatron.core.models.gpt.fine_grained_callables import (
            TransformerLayerNode,
            build_layer_callables,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        # build the forward and backward callables for the transformer/mtp layer
        fwd_callables, bwd_dw_callable_map = build_layer_callables(self.layer)

        # get flags for latter use
        is_mtp = isinstance(self.layer, MultiTokenPredictionLayer)
        transformer_layer = self.layer.mtp_model_layer if is_mtp else self.layer
        is_moe = isinstance(transformer_layer.mlp, MoELayer)
        num_local_experts = transformer_layer.mlp.num_local_experts if is_moe else None

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
            attn_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
            mhc_post_module,
        ) = fwd_callables

        # Create nodes for different operations in the layer
        # Each node type has a predefined name that determines its memory strategy
        self.attn = create_node(comp_stream, attn_module, "attn")
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        # mHC post-processing is compute and belongs on the compute stream: keeping
        # it inside the communication-stream combine node made the recompute's
        # tensors cross-stream (allocated on compute, read on comm, then freed),
        # which the caching allocator cannot track.
        if mhc_post_module is not None:
            self.mhc_post = create_node(comp_stream, mhc_post_module, "mhc_post")
        else:
            self.mhc_post = NoopScheduleNode()

        mhc_recompute_manager = extra_args.get("mhc_recompute_manager")
        if mhc_recompute_manager is not None and extra_args.get(
            "is_last_layer_in_mhc_recompute_group", False
        ):
            from megatron.core.transformer.mhc_recompute import MHCRecomputePhase

            group_index = extra_args["mhc_recompute_group_index"]
            # The group counter restarts per module (decoder / mtp), so fold the
            # module tag into the NVTX label to keep profiles unambiguous.
            module_tag = extra_args.get("mhc_recompute_module_tag", "decoder")
            self.mhc_recompute = ScheduleNode(
                partial(
                    mhc_recompute_manager.recompute_until, MHCRecomputePhase.BEFORE_COMBINE_BWD
                ),
                comp_stream,
                event,
                name="mhc_recompute",
                forward_nvtx_name=f"mhc/recompute/{module_tag}/group_{group_index}/B",
            )
        else:
            self.mhc_recompute = None

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
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
        from megatron.core.transformer.transformer_layer import TransformerLayer

        assert isinstance(self.layer, (TransformerLayer, MultiTokenPredictionLayer)), (
            f"Megatron FSDP with EP Overlap only supports TransformerLayer, "
            f"but got {type(self.layer).__name__}."
        )

        if isinstance(self.layer, TransformerLayer):
            hook_module = self.layer
        else:
            hook_module = self.layer.mtp_model_layer

        # After the last backward op (attn), release backward-pass params.
        self.attn.set_post_backward_hook(lambda: post_backward_hook(hook_module))

        # Determine the last node in forward order. mHC post-processing runs after
        # the combine, so releasing forward-pass params at the combine would pull
        # them out from under it.
        if not isinstance(self.mhc_post, NoopScheduleNode):
            last_fwd_node = self.mhc_post
        elif isinstance(self.moe_combine, NoopScheduleNode):
            last_fwd_node = self.mlp
        else:
            last_fwd_node = self.moe_combine

        # After the last forward op, release forward-pass params.
        last_fwd_node.set_post_forward_hook(lambda: post_forward_hook(hook_module))

    def _iter_layer_nodes(self):
        """Yield this layer's nodes in forward order, NoopScheduleNodes included.

        Single definition of the sequence for recompute_forward/set_forward_no_grad/
        reset_for_recompute; run() issues the same order inline (it interleaves each node
        with the backward layer's), so a new node must be added in both places.

        This is also the full-recompute scope. mtp_post_process must stay in it: it builds
        a torch.cat over the MTP attn node's detached input, so leaving it outside would
        anchor that cat on a no_grad leaf and drop the decoder's gradient via MTP.
        """
        yield self.attn
        yield self.moe_dispatch
        yield self.mlp
        yield self.moe_combine
        yield self.mhc_post
        yield self.mtp_post_process

    def _iter_recomputed_nodes(self):
        """Yield the real ScheduleNodes (skips NoopScheduleNode, which holds no state).

        Only for state toggling; the forward helpers walk _iter_layer_nodes, since a Noop
        still passes the tensor through and still consumes an fp8 context in run().
        """
        for node in self._iter_layer_nodes():
            if isinstance(node, ScheduleNode):
                yield node

    def set_forward_no_grad(self, no_grad: bool):
        """Toggle no-grad forward for this layer's recomputed nodes.

        The initial forward runs with ``no_grad=True`` (no autograd graph retained);
        the backward-time recompute runs with ``no_grad=False`` to rebuild the graph.
        """
        for node in self._iter_recomputed_nodes():
            node.forward_no_grad = no_grad

    def recompute_forward(self, f_input: Tensor) -> Tensor:
        """Re-run this layer with grad enabled and return its output.

        One fp8 context per node, matching run()'s forward half node for node.
        """
        for node in self._iter_layer_nodes():
            with self.get_fp8_context():
                f_input = node.forward(f_input)
        return f_input

    def reset_for_recompute(self):
        """Free the retained forward activations of this layer, keeping the nodes
        reusable for a recompute forward. Also clears the per-layer shared state."""
        for node in self._iter_recomputed_nodes():
            node.reset_for_recompute()
        if getattr(self, 'layer_state', None) is not None:
            # Nodes hold a reference to this same layer_state object, so clear it
            # in place rather than replacing it.
            self.layer_state.__dict__.clear()
        # Free the MoE token dispatcher's transient forward metadata (probs /
        # routing map / permutation mappings). The backward-time recompute re-runs
        # dispatch_preprocess, which repopulates all of it before combine, so this
        # is correctness-neutral and only frees the across-gap per-layer metadata.
        self._reset_moe_dispatcher_state()

    def _reset_moe_dispatcher_state(self):
        """Reset the MoE token dispatcher of this layer's MLP, if any."""
        inner_layer = getattr(self.layer, 'mtp_model_layer', self.layer)
        mlp = getattr(inner_layer, 'mlp', None)
        dispatcher = getattr(mlp, 'token_dispatcher', None)
        if dispatcher is not None:
            dispatcher.reset_transient_forward_state()

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
        (dispatch or combine) of one with the computations (att or mlp) of the other
        to maximize parallelism and efficiency.

        When f_layer and b_layer are not None, forward and backward pass are overlapped as follows:
        comm_stream: combine_bwd | dispatch_fwd->dispatch_bwd  | combine_fwd
        comp_stream: attn_fwd    | mlp_bwd->mlp_bwd_dw->mlp_fwd| attn_bwd
        MLP-side mHC post-processing runs in its own compute-stream node right after
        combine, so the communication stream carries only communication and the
        recompute's tensors are produced and consumed on the same stream.
        Group recompute runs on the normal compute stream immediately before the node containing
        mHC post-processing backward.
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
            # Full recompute: rebuild this layer's segment graph before its backward.
            if b_layer.recompute_segment is not None:
                b_layer.recompute_segment.recompute(b_layer)
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            if b_layer.mhc_recompute is not None:
                b_layer.mhc_recompute.forward()
            b_grad = b_layer.mhc_post.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

        if f_layer is not None:
            # Full recompute: retain this segment's input for the backward-time replay.
            if f_layer.recompute_segment is not None:
                f_layer.recompute_segment.capture(f_layer, f_input)
            with f_layer.get_fp8_context():
                f_input = f_layer.attn.forward(f_input)

        if b_layer is not None:
            b_grad = b_layer.mlp.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_dispatch.forward(f_input)

        if b_layer is not None:
            b_layer.mlp.backward_dw()
            b_grad = b_layer.moe_dispatch.backward(b_grad)

        if b_layer is not None and b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mlp.forward(f_input)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.moe_combine.forward(f_input)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mhc_post.forward(f_input)

        if b_layer is not None and not b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if f_layer is not None:
            with f_layer.get_fp8_context():
                f_input = f_layer.mtp_post_process.forward(f_input)
            segment = f_layer.recompute_segment
            if segment is not None and f_layer is segment.layers[-1]:
                # The segment ran under no_grad, so its output can be a plain leaf.
                # Whatever consumes it - the next segment, an eager layer under
                # recompute_method='block', or post_process - reads a gradient off it to
                # reach back into this segment, which needs a grad-tracking leaf.
                if not f_input.requires_grad:
                    f_input.requires_grad_(True)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_layer is not None and not is_last_layer_in_bwd:
            b_layer.attn.backward_dw()

        if b_layer is not None and b_layer.recompute_segment is not None:
            # The replay has run and its gradient is past this layer; drop the input.
            b_layer.recompute_segment.release_input(b_layer)

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
        padding_mask=None,
        *,
        output_processor: Optional[Callable[..., Any]] = None,
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
        from megatron.core.models.gpt.fine_grained_callables import PostProcessNode, PreProcessNode

        self._model_chunk_state = ModelChunkState()
        self._transformer_layers = []
        self._event = torch.cuda.Event()
        self.pre_process = None
        self.post_process = None
        self.vp_stage = model.vp_stage

        # Full activation recompute; see RecomputeSegment. This plan is only built for
        # the EP A2A overlap scheduler, so recompute_granularity alone decides (not
        # re-checking overlap_moe_expert_parallel_comm keeps the schedule unit-testable).
        self.recompute_full = model.config.recompute_granularity == 'full'
        self._recompute_segments = []
        self._num_decoder_layers = 0

        # save the inputs of model.forward() to ModelChunkState
        self._model_chunk_state.input_ids = input_ids
        self._model_chunk_state.position_ids = position_ids
        self._model_chunk_state.attention_mask = attention_mask
        self._model_chunk_state.decoder_input = decoder_input
        self._model_chunk_state.labels = labels
        self._model_chunk_state.mtp_hidden_states = None
        self._model_chunk_state.mhc_multistream = None
        # Holds the mHC bridge leaf across a segment replay; see RecomputeSegment.
        self._model_chunk_state.mhc_grad_carrier = None
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
        self.pre_process = PreProcessNode(
            model, self._model_chunk_state, self._event, get_comp_stream
        )

        # build layer schedule plan for each layer.
        # The methods to obtain layers are different for MTP so we need the other build plan for
        # MTP. Also, this can help annotate MTP layer so that it can know where MTP is.
        self._build_layer_schedule_plan(
            model.decoder, get_comp_stream, get_comm_stream, module_tag="decoder"
        )
        # Segmentation applies different rules to the decoder and MTP; mark the split.
        self._num_decoder_layers = len(self._transformer_layers)
        self._build_layer_schedule_plan(
            getattr(model, "mtp", None), get_comp_stream, get_comm_stream, module_tag="mtp"
        )

        # build post process
        if model.post_process:
            self.post_process = PostProcessNode(
                model, self._model_chunk_state, self._event, get_comp_stream
            )

        # Split into segments; pre_process and post_process keep their graphs.
        self._build_recompute_segments(model.config)

    def _build_layer_schedule_plan(self, module, comp_stream, comm_stream, module_tag):
        if module is None:
            return

        from megatron.core.tensor_parallel.random import MHCCheckpointManager

        num_layers = len(module.layers)
        config = module.config
        use_mhc_recompute = (
            module.training
            and torch.is_grad_enabled()
            and config.enable_hyper_connections
            and config.recompute_granularity == "selective"
            and "mhc" in config.recompute_modules
        )
        group_size = config.mhc_recompute_layer_num or num_layers
        mhc_recompute_manager = (
            MHCCheckpointManager() if use_mhc_recompute and num_layers > 0 else None
        )
        group_index = 0

        for layer_idx in range(num_layers):
            is_group_end = bool(
                mhc_recompute_manager is not None
                and (layer_idx == num_layers - 1 or (layer_idx + 1) % group_size == 0)
            )
            extra_args = {
                "is_first_layer": layer_idx == 0,
                "is_last_layer": layer_idx == num_layers - 1,
                "mhc_recompute_manager": mhc_recompute_manager,
                "is_last_layer_in_mhc_recompute_group": is_group_end,
                "mhc_recompute_group_index": group_index,
                "mhc_recompute_module_tag": module_tag,
            }
            layer_plan = TransformerLayerSchedulePlan(
                module.layers[layer_idx],
                self.event,
                self.state,
                comp_stream,
                comm_stream,
                extra_args,
            )
            self._transformer_layers.append(layer_plan)

            if is_group_end and layer_idx != num_layers - 1:
                group_index += 1
                mhc_recompute_manager = MHCCheckpointManager()

    def _build_recompute_segments(self, config):
        """Split this chunk's layers into full-recompute segments.

        Decoder segmentation follows the same high-level grouping as
        megatron.core.recompute.checkpointed_forward: 'uniform' groups every decoder layer
        by recompute_num_layers, while 'block' recomputes the first recompute_num_layers
        decoder layers one per segment. The MTP and quantized-block exceptions below
        intentionally differ from the non-overlap checkpoint path.

        Unlike the non-overlap 'block' branch there is no recompute_skip_num_layers
        window under fp8/fp4: no checkpoint primitive is involved, since the replay marks
        the retained segment input grad-tracking itself.
        """
        if not self.recompute_full:
            return

        method = config.recompute_method
        num_layers = config.recompute_num_layers
        # TransformerConfig owns the user-facing validation; re-check here because
        # build_schedule_plan() is reachable directly and 'block' with num_layers=None
        # would slice [:None] and silently recompute every decoder layer.
        if method not in ("uniform", "block"):
            raise ValueError(f"Invalid activation recompute method: {method}.")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError(f"recompute_num_layers must be a positive integer, got {num_layers}.")

        def add_segment(layers, start_index):
            self._recompute_segments.append(
                RecomputeSegment(layers, self._model_chunk_state, start_index)
            )

        decoder_layers = self._transformer_layers[: self._num_decoder_layers]
        mtp_layers = self._transformer_layers[self._num_decoder_layers :]

        if method == "uniform":
            for start in range(0, len(decoder_layers), num_layers):
                add_segment(decoder_layers[start : start + num_layers], start)
        else:
            for start, layer in enumerate(decoder_layers[:num_layers]):
                add_segment([layer], start)
        # MTP is one segment per depth under both methods. overlap_moe_expert_parallel_comm
        # only supports mtp_num_layers == 1, so the MTP side of the config is identical
        # either way. MultiTokenPredictionLayer._checkpointed_forward leaves MTP eager
        # under 'block' (its own TODO), which is safe there because the decoder block goes
        # through tensor_parallel.checkpoint and keeps an autograd edge into MTP. This
        # path's replay has no such edge, so an eager MTP would strand the hand-off.
        for i, layer in enumerate(mtp_layers):
            add_segment([layer], self._num_decoder_layers + i)

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
        self._recompute_segments = []
        self._model_chunk_state.model = None
        self.pre_process.model_chunk_state = None
        self.pre_process = None

        if self.post_process is not None:
            self.post_process.model_chunk_state = None
            self.post_process = None

    def release_layer_activations(self):
        """Free the segments' forward activations, keeping only their input tensors."""
        for segment in self._recompute_segments:
            for layer in segment.layers:
                layer.reset_for_recompute()

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

        if b_schedule_plan:
            b_schedule_plan.record_current_stream()
            assert b_grad is not None
            if pre_backward is not None:
                pre_backward(b_schedule_plan.vp_stage)
                b_schedule_plan.record_current_stream()

            # Before any layer backward: post_process keeps its own graph and detaches
            # its input, and releasing it first avoids co-materializing a large vocab
            # graph with a recomputed segment on the last PP stage.
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
            # so the p2p comm could be overlapped with the attn backward
            with torch.cuda.stream(get_comm_stream()):
                f_schedule_plan.wait_current_stream()
                post_forward(f_input, f_schedule_plan.vp_stage)

        # post_backward()/send_backward_recv_backward() is running in the computation stream,
        # so the p2p comm could be overlapped with the wgrad of attn backward
        if b_schedule_plan is not None and post_backward is not None:
            b_schedule_plan.wait_current_stream()
            post_backward(b_grad, b_schedule_plan.vp_stage)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_num_layers > 0:
            assert b_layer is not None
            b_layer.attn.backward_dw()
            b_layer.release_state()

        # post process forward
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            if f_schedule_plan.recompute_full and f_input is not None and not f_input.requires_grad:
                # The last layer ran under no_grad, so post_process needs a grad-tracking
                # leaf to seed the replayed last segment's backward.
                f_input.requires_grad_(True)
            f_input = f_schedule_plan.post_process.forward(f_input)
        # pre process backward
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

        # The forward output has been consumed (PP send / post_process), so the
        # recomputed layers' activations can go; only segment inputs are kept.
        if f_schedule_plan is not None:
            f_schedule_plan.release_layer_activations()

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()
            # Release reference as early as possible, this helps avoid memory leak.
            b_schedule_plan.release_state()

        return f_input
