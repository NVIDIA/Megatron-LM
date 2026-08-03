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


# Fields of ModelChunkState that layer forwards mutate in place, and that a
# backward-time recompute must therefore see at their pre-segment values:
#   * input_ids / position_ids / padding_mask: MultiTokenPredictionLayer._get_embeddings
#     rolls the tokens by one and writes them back (see
#     fine_grained_callables.submodule_mtp_attn_forward). Decoder attention reads
#     padding_mask too, so without a restore the decoder segments (replayed after the
#     MTP ones, since backward runs in reverse order) would see the MTP-rolled mask.
#   * mtp_hidden_states: rebuilt by the first MTP layer, appended to by each MTP
#     post-process, cleared by the last one.
#   * mhc_multistream: produced by the last decoder layer, consumed by the first MTP
#     layer, cleared by the last MTP post-process.
_MUTABLE_CHUNK_STATE_FIELDS = (
    'input_ids',
    'position_ids',
    'padding_mask',
    'mtp_hidden_states',
    'mhc_multistream',
)


def _copy_chunk_state_value(value):
    """Copy list-valued chunk state so later in-place mutation cannot corrupt a
    snapshot (``mtp_hidden_states`` is appended to during the forward)."""
    return list(value) if isinstance(value, list) else value


class RecomputeSegment:
    """A contiguous group of transformer layers recomputed as a single unit.

    Full activation recompute under ``--overlap-moe-expert-parallel-comm`` keeps only
    the *input* of each segment alive across the forward->backward gap: the initial
    forward of every segment layer runs under ``no_grad`` and the layers' activations
    are released once the chunk forward completes. At the start of the segment's
    backward the forward is replayed with grad enabled, so only one segment's
    activations are materialized at a time.

    Segmentation mirrors the non-overlap recompute semantics -
    ``megatron.core.recompute.checkpointed_forward`` for the decoder block and
    ``MultiTokenPredictionLayer._checkpointed_forward`` for MTP:

    * ``recompute_method='uniform'``: decoder layers in groups of
      ``recompute_num_layers``; one segment per MTP layer (``recompute_num_layers``
      is asserted to be 1 when MTP is present, as in the non-overlap path).
    * ``recompute_method='block'``: the first ``recompute_num_layers`` decoder layers,
      one segment each; MTP layers are not recomputed.

    The A2A communication issued by a segment replay is exposed (not overlapped) by
    design; the normal forward and normal backward A2A overlap is preserved.
    """

    def __init__(self, layers, chunk_state, start_index):
        assert len(layers) > 0, "a recompute segment must contain at least one layer"
        self.layers = layers
        self.chunk_state = chunk_state
        self.start_index = start_index
        # Captured at the start of the segment's initial forward.
        self.input_tensor = None
        self.state_snapshot = None
        self.rng_states = None
        for i, layer in enumerate(layers):
            layer.recompute_segment = self
            layer.is_segment_head = i == 0
            layer.is_segment_tail = i == len(layers) - 1
            # The initial forward retains no autograd graph for these layers.
            layer.set_forward_no_grad(True)

    def capture(self, f_input):
        """Snapshot what the backward-time replay needs: the segment input tensor,
        the mutable chunk-state fields, and the RNG states."""
        from megatron.core.tensor_parallel.random import _get_all_rng_states

        self.input_tensor = f_input
        cs = self.chunk_state
        self.state_snapshot = {
            name: _copy_chunk_state_value(getattr(cs, name, None))
            for name in _MUTABLE_CHUNK_STATE_FIELDS
        }
        self.rng_states = _get_all_rng_states()

    def release_activations(self):
        """Free the layers' retained forward activations after the initial forward."""
        for layer in self.layers:
            layer.reset_for_recompute()

    def release_input(self):
        """Drop the retained input once the segment backward has consumed it."""
        self.input_tensor = None
        self.state_snapshot = None

    def recompute(self):
        """Replay the segment forward with grad enabled, right before its backward.

        The mapping from the replayed activations back to the per-submodule
        ScheduleNodes is implicit: the same node objects are re-run in the same
        forward order, so each node repopulates its own ``inputs`` / ``output`` /
        ``detached`` / ``layer_state`` slots exactly as in the initial forward.
        """
        from megatron.core.tensor_parallel.random import _fork_rng, _set_all_rng_states

        assert self.input_tensor is not None, (
            "layer-level full recompute requires the retained segment input tensor, "
            "but it is missing."
        )
        cs = self.chunk_state

        # ``mhc_multistream`` is the only cross-layer detach bridge
        # (fine_grained_callables.finalize_decoder_layer_output): the last decoder layer
        # detaches it into the chunk state and the first MTP layer consumes it, so the
        # MTP backward accumulates into that leaf *before* the decoder segment holding
        # its producer is replayed. The replay installs a freshly detached tensor in
        # ``node.detached``, so the already-accumulated gradient is carried over below.
        prev_mhc_multistream = getattr(cs, 'mhc_multistream', None)

        for name, value in self.state_snapshot.items():
            setattr(cs, name, _copy_chunk_state_value(value))

        if cs.mhc_multistream is not None and not cs.mhc_multistream.requires_grad:
            # Produced by the last decoder layer's no_grad forward, so the detached
            # leaf carries requires_grad=False. The first MTP layer consumes it inside
            # its replayed (grad-enabled) forward, and its producer's backward reads
            # this leaf's .grad, so it must be grad-tracking before the replay.
            cs.mhc_multistream.requires_grad_(True)

        stage_input = self.input_tensor
        if not stage_input.requires_grad:
            # The initial forward ran under no_grad, so the segment input is a plain
            # leaf. The replayed nodes detach their inputs and read ``inputs[i].grad``
            # to propagate to the previous segment, which needs a grad-tracking leaf.
            stage_input.requires_grad_(True)

        # Enable grad on the segment nodes' forward so the replay builds the graph.
        for layer in self.layers:
            layer.set_forward_no_grad(False)

        # Replay the forward RNG stream so dropout / rng-forked ops reproduce the
        # initial forward. _fork_rng() restores the ambient RNG state on exit.
        with _fork_rng():
            _set_all_rng_states(*self.rng_states)
            f_input = stage_input
            for i, layer in enumerate(self.layers):
                nvtx_msg = f"recompute_layer_{self.start_index + i}"
                nvtx_range_push(nvtx_msg)
                f_input = layer.recompute_forward(f_input)
                nvtx_range_pop(nvtx_msg)
        self.rng_states = None

        new_mhc_multistream = getattr(cs, 'mhc_multistream', None)
        if (
            prev_mhc_multistream is not None
            and new_mhc_multistream is not None
            and new_mhc_multistream is not prev_mhc_multistream
        ):
            new_mhc_multistream.grad = prev_mhc_multistream.grad


class TransformerLayerSchedulePlan:
    """Schedule the execution plan for nodes in a transformer or MTP layer.

    This class organizes the submodules of a transformer or MTP layer, including attention,
    MLP, MoE dispatch and combine, optional mHC recomputation, and MTP post-processing nodes.

    layer (TransformerLayerSchedulePlan)
    ├── attn (TransformerLayerNode): attention -> layernorm -> router -> dispatch preprocess
    ├── moe_dispatch (TransformerLayerNode): dispatch All2All
    ├── mlp (TransformerLayerNode): mlp module
    ├── moe_combine (TransformerLayerNode): combine All2All (incl. MLP-side mHC post-processing)
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

        # Layer-level full recompute bookkeeping, filled in by
        # TransformerModelChunkSchedulePlan._build_recompute_segments(). Layers that
        # are not recomputed keep recompute_segment == None and retain their graph.
        self.recompute_segment = None
        self.is_segment_head = False
        self.is_segment_tail = False

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

        mhc_recompute_manager = extra_args.get("mhc_recompute_manager")
        if mhc_recompute_manager is not None and extra_args.get(
            "is_last_layer_in_mhc_recompute_group", False
        ):
            group_index = extra_args["mhc_recompute_group_index"]
            # The group counter restarts per module (decoder / mtp), so fold the
            # module tag into the NVTX label to keep profiles unambiguous.
            module_tag = extra_args.get("mhc_recompute_module_tag", "decoder")
            self.mhc_recompute = ScheduleNode(
                mhc_recompute_manager.recompute_now,
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

        # Determine the last node in forward order.
        if isinstance(self.moe_combine, NoopScheduleNode):
            last_fwd_node = self.mlp
        else:
            last_fwd_node = self.moe_combine

        # After the last forward op, release forward-pass params.
        last_fwd_node.set_post_forward_hook(lambda: post_forward_hook(hook_module))

    def _iter_recomputed_nodes(self):
        """Yield the ScheduleNodes of this layer that take part in full recompute.

        The recompute scope is [attn, moe_dispatch, mlp, moe_combine].
        ``mtp_post_process`` is deliberately excluded so that this path mirrors the
        non-overlap checkpoint scope: ``MultiTokenPredictionLayer._checkpointed_forward``
        wraps ``_proj_and_transformer_layer`` only, leaving ``_postprocess`` outside the
        checkpoint. Keeping the MTP post-process node's autograd graph alive also keeps
        intact the un-detached ``chunk_state.mtp_hidden_states`` -> ``torch.cat`` graph
        that spans MTP depths, which a replayed post-process would break. For decoder
        layers ``mtp_post_process`` is a NoopScheduleNode, so excluding it costs nothing.
        """
        for node in (self.attn, self.moe_dispatch, self.mlp, self.moe_combine):
            if isinstance(node, ScheduleNode):
                yield node

    def set_forward_no_grad(self, no_grad: bool):
        """Toggle no-grad forward for this layer's recomputed nodes.

        The initial forward runs with ``no_grad=True`` (no autograd graph retained);
        the backward-time recompute runs with ``no_grad=False`` to rebuild the graph.
        """
        for node in self._iter_recomputed_nodes():
            node.forward_no_grad = no_grad

    def recompute_forward(self, f_input):
        """Re-run this layer's recompute scope with grad enabled.

        Mirrors the forward half of ``TransformerLayerSchedulePlan.run`` for the
        [attn, moe_dispatch, mlp, moe_combine] nodes, including the per-node fp8
        context. For a decoder layer the returned tensor is the layer output, so a
        multi-layer segment chains through it; MTP segments are always single-layer
        (``recompute_num_layers == 1``) and their chunk output comes from the
        non-recomputed ``mtp_post_process`` node instead.
        """
        with self.get_fp8_context():
            f_input = self.attn.forward(f_input)
        with self.get_fp8_context():
            f_input = self.moe_dispatch.forward(f_input)
        with self.get_fp8_context():
            f_input = self.mlp.forward(f_input)
        with self.get_fp8_context():
            f_input = self.moe_combine.forward(f_input)
        return f_input

    def maybe_capture_recompute_input(self, f_input):
        """Capture the segment input at the start of the segment's initial forward."""
        if self.recompute_segment is not None and self.is_segment_head:
            self.recompute_segment.capture(f_input)

    def maybe_recompute_segment(self):
        """Replay this layer's segment if this layer is its backward entry point.

        Called at the start of the layer backward, after ``mtp_post_process.backward``
        (whose graph is not recomputed) and before ``moe_combine.backward``.
        """
        if self.recompute_segment is not None and self.is_segment_tail:
            self.recompute_segment.recompute()

    def maybe_release_recompute_input(self):
        """Drop the retained segment input once the segment backward has consumed it."""
        if self.recompute_segment is not None and self.is_segment_head:
            self.recompute_segment.release_input()

    def prepare_post_process_input(self, f_input):
        """Make the input of a recomputed layer's MTP post-process node grad-tracking.

        ``mtp_post_process`` sits outside the recompute scope and keeps its own graph,
        but its input is the recomputed part's output, whose initial forward ran under
        no_grad. ScheduleNode._forward copies ``requires_grad`` from the incoming tensor
        onto the detached input it saves, so without this the node's backward would
        leave ``inputs[0].grad`` at None and the grad seed for the replayed combine
        backward would be lost. Same fix as the chunk-level post_process input.
        """
        if (
            self.recompute_segment is not None
            and isinstance(self.mtp_post_process, ScheduleNode)
            and f_input is not None
            and not f_input.requires_grad
        ):
            f_input.requires_grad_(True)
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
        MLP-side mHC post-processing runs inside the combine node on the communication stream.
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
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            if b_layer.mhc_recompute is not None:
                b_layer.mhc_recompute.forward()
            # Layer-level full recompute: rebuild this layer's segment forward graph
            # from the retained segment input before any of its layer backward runs.
            # Runs after mtp_post_process.backward, which keeps its own graph and is
            # therefore not replayed, and whose backward produces the grad seed that
            # feeds the replayed combine backward. No-op unless recompute is on.
            b_layer.maybe_recompute_segment()
            b_grad = b_layer.moe_combine.backward(b_grad)

        if f_layer is not None:
            # Capture the segment input right before its first layer's forward.
            # No-op unless this layer starts a recompute segment.
            f_layer.maybe_capture_recompute_input(f_input)
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

        if b_layer is not None and not b_layer.config.ep_overlap_early_attn_memory_release:
            b_grad = b_layer.attn.backward(b_grad)

        if f_layer is not None:
            f_input = f_layer.prepare_post_process_input(f_input)
            with f_layer.get_fp8_context():
                f_input = f_layer.mtp_post_process.forward(f_input)

        # Delay the last attn_dw in backward pass (attn_dw of the first layer)
        # for overlapping with the p2p comm
        if b_layer is not None and not is_last_layer_in_bwd:
            b_layer.attn.backward_dw()

        if b_layer is not None:
            # The segment input is only needed by the replay, which has already run and
            # whose gradient has now been propagated past this layer. attn.backward_dw()
            # does not read it. No-op unless this layer starts a recompute segment.
            b_layer.maybe_release_recompute_input()

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
        from megatron.core.models.gpt.fine_grained_callables import PostProcessNode, PreProcessNode

        self._model_chunk_state = ModelChunkState()
        self._transformer_layers = []
        self._event = torch.cuda.Event()
        self.pre_process = None
        self.post_process = None
        self.vp_stage = model.vp_stage

        # Full activation recompute: the chunk's layers are split into
        # RecomputeSegments; only each segment's input tensor is retained across the
        # forward->backward gap and the segment forward is replayed at the start of its
        # own backward. See _build_recompute_segments() and RecomputeSegment.
        # This schedule plan is only ever built for the EP A2A overlap scheduler, so
        # recompute_granularity alone decides; overlap_moe_expert_parallel_comm is not
        # re-checked here (that lets the schedule be unit-tested standalone).
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
        # Recompute segmentation follows the non-overlap path, which applies different
        # rules to the decoder block and to MTP, so record where the decoder ends.
        self._num_decoder_layers = len(self._transformer_layers)
        self._build_layer_schedule_plan(
            getattr(model, "mtp", None), get_comp_stream, get_comm_stream, module_tag="mtp"
        )

        # build post process
        if model.post_process:
            self.post_process = PostProcessNode(
                model, self._model_chunk_state, self._event, get_comp_stream
            )

        # For full recompute, split the layers into segments. The initial forward of
        # every segment layer runs under no_grad (no activation retained); pre_process,
        # post_process and mtp_post_process keep their graphs (they are not recomputed).
        self._build_recompute_segments(model.config)

    def _build_layer_schedule_plan(self, module, comp_stream, comm_stream, module_tag):
        if module is None:
            return

        from megatron.core.tensor_parallel.random import CheckpointManager

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
            CheckpointManager() if use_mhc_recompute and num_layers > 0 else None
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
                mhc_recompute_manager = CheckpointManager()

    def _build_recompute_segments(self, config):
        """Split this chunk's layers into full-recompute segments.

        The segmentation mirrors the non-overlap recompute path so that
        ``recompute_method`` / ``recompute_num_layers`` mean the same thing with and
        without ``--overlap-moe-expert-parallel-comm``:

        * ``uniform`` (``megatron.core.recompute.checkpointed_forward``): the decoder
          layers are uniformly divided into groups of ``recompute_num_layers``, and
          every group is recomputed. MTP recomputes one depth at a time
          (``MultiTokenPredictionLayer._checkpointed_forward`` asserts
          ``recompute_num_layers == 1``), so each MTP layer becomes its own segment.
        * ``block``: only the first ``recompute_num_layers`` decoder layers are
          recomputed, one layer per segment; the remaining decoder layers and all MTP
          layers keep their forward graph, matching the non-overlap path where MTP
          warns and skips recompute under ``block``.

        One difference from the non-overlap ``block`` branch is intentional: it grows a
        ``recompute_skip_num_layers`` window under fp8/fp4 while ``hidden_states`` does
        not require grad, because ``te_checkpoint``'s reentrant autograd needs a
        grad-requiring input. This path does not use a checkpoint primitive - the replay
        marks the retained segment input grad-tracking itself - so no such window is
        needed and the recomputed layer set is exactly the first N.
        """
        if not self.recompute_full:
            return

        method = config.recompute_method
        num_layers = config.recompute_num_layers
        # transformer_config validates these; assert here so a mis-wired config fails
        # at plan-build time rather than silently recomputing nothing.
        assert method in ("uniform", "block"), (
            "recompute_method must be 'uniform' or 'block' with "
            f"overlap_moe_expert_parallel_comm full recompute, got {method}"
        )
        assert (
            isinstance(num_layers, int) and num_layers >= 1
        ), f"recompute_num_layers must be a positive integer, got {num_layers}"

        decoder_layers = self._transformer_layers[: self._num_decoder_layers]
        mtp_layers = self._transformer_layers[self._num_decoder_layers :]

        def add_segment(layers, start_index):
            if layers:
                self._recompute_segments.append(
                    RecomputeSegment(layers, self._model_chunk_state, start_index)
                )

        if method == "uniform":
            for start in range(0, len(decoder_layers), num_layers):
                add_segment(decoder_layers[start : start + num_layers], start)
            for i, layer in enumerate(mtp_layers):
                add_segment([layer], self._num_decoder_layers + i)
        else:
            for start, layer in enumerate(decoder_layers[:num_layers]):
                add_segment([layer], start)

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
        # Segments hold the layer plans and the retained segment inputs.
        self._recompute_segments = []
        self._model_chunk_state.model = None
        self.pre_process.model_chunk_state = None
        self.pre_process = None

        if self.post_process is not None:
            self.post_process.model_chunk_state = None
            self.post_process = None

    def release_layer_activations(self):
        """Free every recompute segment's retained forward activations after the
        initial forward, keeping only each segment's input tensor.

        The layer nodes remain reusable; RecomputeSegment.recompute() rebuilds their
        forward state right before the segment's backward.
        """
        for segment in self._recompute_segments:
            segment.release_activations()

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

            # Run post_process backward before any layer backward. post_process keeps
            # its own graph (it is not recomputed) and detaches its input, so its
            # backward produces b_grad independently of the segment replays. Releasing
            # the post_process graph (output projection / logits / saved tensors) first
            # avoids co-materializing it with a recomputed segment on the last PP stage,
            # where a large vocab would otherwise spike peak memory.
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
            if (
                f_schedule_plan.recompute_full
                and f_input is not None
                and not f_input.requires_grad
            ):
                # The last layer's forward ran under no_grad, so the chunk output is a
                # plain value tensor. post_process needs a grad-requiring leaf so its
                # backward produces the grad seed that feeds the replayed last segment.
                # Under recompute_method='block' the trailing layers keep their graph
                # and f_input already requires grad, so this is skipped.
                f_input.requires_grad_(True)
            f_input = f_schedule_plan.post_process.forward(f_input)
        # pre process backward
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

        # Free the forward chunk's recomputed-layer activations now that its forward
        # output has been consumed (PP send / post_process). Only each segment's input
        # tensor is kept for the backward-time replay. No-op unless recompute is on.
        if f_schedule_plan is not None:
            f_schedule_plan.release_layer_activations()

        if f_schedule_plan:
            f_schedule_plan.wait_current_stream()
        if b_schedule_plan:
            b_schedule_plan.wait_current_stream()
            # Release reference as early as possible, this helps avoid memory leak.
            b_schedule_plan.release_state()

        return f_input
