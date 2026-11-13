# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transformer Engine CUDA graph execution attached to Megatron modules."""

from __future__ import annotations

from functools import lru_cache, partial
from importlib import import_module
from weakref import WeakKeyDictionary, proxy

import torch

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.utils import get_attr_wrapped_model


class _TECudaGraphBackwardDWWrapperMixin:
    """Skip captured eager weight gradients and replay the TE backward graph."""

    def __init__(self, layer) -> None:
        if layer.config.cuda_graph_impl == "transformer_engine":
            assert isinstance(
                layer, TECudaGraphMixin
            ), "TE cuda graphed ep overlap requires TECudaGraphMixin."
        super().__init__(layer)
        self.graphed_backward_dw_callable = None
        self.cuda_graph_modules = layer.config.cuda_graph_modules

    def backward_dw(self):
        """Run eager or graphed backward wgrad callables for the wrapped layer."""
        is_replay = (
            self.layer.config.cuda_graph_impl == "transformer_engine"
            and hasattr(self.layer, 'te_cuda_graphs')
            and self.layer.te_cuda_graphs
        )
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


@lru_cache(maxsize=None)
def get_te_cuda_graph_module_class(module):
    """Select a TE layer class after the caller checks the configured backend.

    Preserve custom TransformerLayer and MoELayer subclasses and their overrides.
    Other modules, including those already opting into the mixin, pass through.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer, TECudaGraphMoELayer
    from megatron.core.transformer.moe.moe_layer_mixins import TEPartialCaptureMoEMixin
    from megatron.core.transformer.transformer_layer import (
        TECudaGraphTransformerLayer,
        TransformerLayer,
    )

    if not isinstance(module, type):
        return module
    if issubclass(module, (TECudaGraphMixin, TEPartialCaptureMoEMixin)):
        return module
    if module is TransformerLayer:
        return TECudaGraphTransformerLayer
    if module is MoELayer:
        return TECudaGraphMoELayer

    if issubclass(module, MoELayer):
        bases = (module, TECudaGraphMoELayer)
    elif issubclass(module, TransformerLayer):
        from megatron.core.transformer.hyper_connection_transformer_layer import (
            HyperConnectionTransformerLayer,
        )
        from megatron.core.transformer.transformer_layer_mixins import (
            TECudaGraphHyperConnectionTransformerLayerMixin,
        )

        bases = (module, TECudaGraphTransformerLayer)
        if issubclass(module, HyperConnectionTransformerLayer):
            bases = (
                module,
                TECudaGraphHyperConnectionTransformerLayerMixin,
                TECudaGraphTransformerLayer,
            )
    else:
        return module

    # Keep the original subclass before the TE adapter so its __call__, input
    # preparation, and forward overrides retain precedence in the MRO.
    # Encode the original import path for reconstruction by pickle in another
    # process; dots in __qualname__ would otherwise mean nested attributes.
    qualname = (
        f"_te_cuda_graph_class_{module.__module__.encode().hex()}_"
        f"{module.__qualname__.encode().hex()}"
    )
    return type(
        f"TECudaGraph{module.__name__}", bases, {"__module__": __name__, "__qualname__": qualname}
    )


def __getattr__(name):
    """Resolve generated TE layer classes when unpickling an importable subclass."""
    prefix = "_te_cuda_graph_class_"
    if not name.startswith(prefix):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, class_path = name[len(prefix) :].split("_", 1)
    module = import_module(bytes.fromhex(module_path).decode())
    for component in bytes.fromhex(class_path).decode().split("."):
        module = getattr(module, component)
    return get_te_cuda_graph_module_class(module)


class TECudaGraphOffloadingInterface:
    """Coordinate TE graph execution with the current activation offload manager."""

    _instances = WeakKeyDictionary()

    @classmethod
    def get_instance(cls):
        """Share graph resources across layers for the lifetime of the offload manager."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        manager = PipelineOffloadManager.get_instance()
        if manager not in cls._instances:
            cls._instances[manager] = cls(manager)
        return cls._instances[manager]

    def __init__(self, manager) -> None:
        # Do not keep a replaced manager (and its offloaded tensors) alive through the cache.
        self._manager = proxy(manager)
        # TE CUDA graph offload paths need a stream/event pair that lives outside
        # individual layer objects so capture, replay, and backward hooks order
        # the same D2H/H2D work with the same synchronization primitives.
        self._cuda_graph_stream = torch.cuda.Stream()
        self._cuda_graph_event = torch.cuda.Event(external=True)

    def cuda_graph_stream(self):
        """Get the CUDA graph stream."""
        return self._cuda_graph_stream

    def cuda_graph_event(self):
        """Get the CUDA graph event."""
        return self._cuda_graph_event

    def forward_record(self) -> None:
        """Record the forward event for cuda graph capture."""
        mgr = self._manager
        torch.cuda.current_stream().record_event(self._cuda_graph_event)
        torch.cuda.current_stream().wait_stream(mgr.d2h_stream)

    @staticmethod
    def backward_record(tensor) -> torch.Tensor:
        """Insert TE graph synchronization with activation reloads in backward."""
        return TECudaGraphBackwardReloadSyncFunction.apply(tensor)

    def flush_delayed_groups(self):
        """Flush the delayed groups."""
        self._manager.flush_delayed_groups()

    def enter_replay(self):
        """Enter CUDA graph replay mode to enable delayed offloading."""
        self._manager.defer_group_commits = True

    def exit_replay(self):
        """Exit CUDA graph replay mode."""
        self._manager.defer_group_commits = False


class TECudaGraphBackwardReloadSyncFunction(torch.autograd.Function):
    """Synchronize TE CUDA graph backward with activation reloads.

    Forward is an identity. Backward records the graph synchronization event
    and waits for the offload manager's H2D stream before passing gradients through.
    """

    @staticmethod
    def forward(ctx, tensor) -> torch.Tensor:
        """Forward pass for cuda graph capture."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import debug_rank

        debug_rank("TECudaGraphBackwardReloadSyncFunction forward")
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Record the backward event and wait for the h2d stream on cuda graph stream."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import debug_rank

        debug_rank("TECudaGraphBackwardReloadSyncFunction backward")
        off_interface = TECudaGraphOffloadingInterface.get_instance()
        mgr = off_interface._manager
        # This event connects TE's graph stream with the reload stream so
        # backward consumers do not race H2D reloads launched outside the graph.
        torch.cuda.current_stream().record_event(off_interface.cuda_graph_event())
        torch.cuda.current_stream().wait_stream(mgr.h2d_stream)
        return (grad_output,)


class TECudaGraphMixin:
    """Run a Megatron module through Transformer Engine CUDA graphs.

    When config.cuda_graph_impl == "transformer_engine", a training-mode call to the
    module is redirected: while graphs are being captured it records a graph, and once
    te_cuda_graphs is populated it replays the graph for the current microbatch.
    Eval-mode calls, and calls with any other cuda_graph_impl, run forward() normally.

    How to use:
        List this mixin before the module's base class, so its __call__ runs first:

            class MambaLayer(TECudaGraphMixin, GraphableMegatronModule, ...):

        Every __init__ in the class chain must call super().__init__(...). This
        mixin's __init__ passes its arguments through unchanged, then sets up the
        graph state once the module is built.

    Default behavior:
        The whole forward() is one graph with a single tensor input, hidden_states.
        Modules can supply additional tensor inputs, such as masks or rotary embeddings,
        through their static-input and replay methods. These inputs pass through the shared
        methods as **kwargs so this mixin does not need each module's argument list.

    Graphing only part of the module:
        Subclasses override these methods (see TECudaGraphTransformerLayerMixin):

        _te_cuda_graph_capture_forward: runs the selected computation during capture.
        _te_cuda_graph_prepare_replay: runs any eager code that comes before the graph, then
            returns two callables: one that replays the graph, and one that takes
            the graph outputs and finishes the call.
        _te_cuda_graph_finish_replay: the default for that second callable. Runs any eager
            code after the graph and returns what forward() would have returned.
        is_graphable_for_scope / _get_submodules_under_cudagraphs: which
            cuda_graph_modules scopes this module covers and which submodules
            are inside the graph.

    Rules for subclasses:
        Graph inputs must be tensors or None; drop anything else (e.g.
        packed_seq_params, inference_context) before replay.
        Bind state that the after-graph code needs (masks, context, etc.) into
        the callable returned by _te_cuda_graph_prepare_replay, not onto self, so one
        call's state cannot leak into another.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.cuda_graph_impl == "transformer_engine":
            self.register_te_cuda_graph()

    def register_te_cuda_graph(self):
        """Initialize TE graph state without changing this module's parameters."""
        if self.config.cuda_graph_impl != "transformer_engine":
            raise ValueError("TE graph registration requires cuda_graph_impl='transformer_engine'")

        # List to store CUDA graphs. A list of `N` CUDA graphs for this layer where N is
        # the number of microbatches. Multiple CUDA graphs per layer is required to support
        # pipelining which requires running FWD graph of multiple microbatches before BWD
        # graph. To enable CUDA graph, this list should be populated in the model training
        # script with the graphs returned by make_graphed_callables API before the first
        # training step.
        self.te_cuda_graphs = []
        # List to store forward pre-hooks. Forward pre-hooks are not captured into CUDA
        # graphs. Those hooks and args are collected in this list and should be manually
        # triggered before CUDA Graph running. This is required to ensure the correct param
        # all-gather overlap with forward compute.
        self.te_cuda_graph_manual_hooks = []
        self.te_cuda_graph_current_microbatch = 0
        if self.config.fine_grained_activation_offloading:
            # Allocate the shared graph stream/event before capture starts.
            self.get_offload_interface()
        return self

    def __call__(self, *args, **kwargs):
        """Capture or replay the TE graph when this module owns the scope."""
        if self._should_call_te_cudagraph():
            if not self.te_cuda_graphs:
                # Do CUDA Graphs capture.
                cuda_graph_func = self._te_cuda_graph_capture
            else:
                # Do CUDA Graphs replay.
                cuda_graph_func = self._te_cuda_graph_replay
            return cuda_graph_func(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def is_graphable_for_scope(self, modules):
        """Whether this module covers one of the requested graph scopes."""
        return not modules

    def set_te_cuda_graph_backward_dw_wrapper(self):
        """Replace the backward_dw callable with dw cuda graph."""
        assert (
            self.backward_dw_wrapper is not None
        ), "`backward_dw_wrapper` must be set when cuda graphs are enabled for ep overlap."
        self.backward_dw_wrapper.set_graphed_backward_dw_callable(
            partial(self._te_cuda_graph_backward_dw_graph, self.te_cuda_graph_current_microbatch)
        )

    def _te_cuda_graph_index(self, microbatch_idx):
        """Select the replay slot shared by forward and deferred weight gradients."""
        if (
            self.config.sequence_packing_scheduler == "dp_balanced"
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.config.pipeline_model_parallel_size > 1
            and microbatch_idx >= len(self.te_cuda_graphs)
        ):
            raise RuntimeError(
                "Packed microbatch count exceeds the captured source-batch bound; "
                "recapture CUDA graphs before increasing the source batch size."
            )
        # PP1 captures one slot. Preserve modulo reuse for non-packed TE callers too;
        # only scheduler-produced packed PP batches guarantee a fixed source-batch bound.
        return microbatch_idx % len(self.te_cuda_graphs)

    def _te_cuda_graph_backward_dw_graph(self, microbatch_idx):
        """
        CUDA Graph backward weight gradient computation for current layer.
        """
        cg_index = self._te_cuda_graph_index(microbatch_idx)
        if not hasattr(self.te_cuda_graphs[cg_index], 'backward_dw'):
            return
        self.te_cuda_graphs[cg_index].backward_dw()

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """
        Get the static inputs for the layer.
        We assume that the module has one hidden_states input, whose shape is inferred
        from the seq_length, micro_batch_size, and parallel config.
        Override this method if the module has other inputs.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        # Calculate data shape related values.
        context_parallel_size = self.config.context_parallel_size
        slen_per_cp = seq_length // context_parallel_size
        sequence_parallel = self.config.sequence_parallel
        tensor_model_parallel_size = self.config.tensor_model_parallel_size
        slen_per_cptp = (
            slen_per_cp // tensor_model_parallel_size if sequence_parallel else slen_per_cp
        )

        static_inputs = {}
        static_inputs["hidden_states"] = torch.ones(
            (slen_per_cptp, micro_batch_size, self.config.hidden_size),
            dtype=torch.bfloat16,
            requires_grad=True,
            device=torch.cuda.current_device(),
        )
        return static_inputs

    def setup_manual_hooks(self, make_hook_func):
        """
        Set CUDA Graph manual hooks for the submodules that contain direct parameters and are
        covered by cudagraphs.
        """
        self.te_cuda_graph_manual_hooks = []

        # Select the modules who contain direct parameters and are covered by cudagraphs.
        # Add these modules to the `te_cuda_graph_manual_hooks` because their hooks will not
        # be automatically triggered when they go through the CUDA Graph path.
        param_modules = {}
        for submodule in self._get_submodules_under_cudagraphs():
            for module in submodule.modules():
                if next(module.parameters(recurse=False), None) is not None:
                    # Module contains direct parameters.
                    param_modules[id(module)] = module
        for module in param_modules.values():
            self.te_cuda_graph_manual_hooks.append((make_hook_func(), (module,)))

    def _get_submodules_under_cudagraphs(self):
        """
        Get the submodules that are covered by cudagraphs. Return a list that only contains the
        module itself if the whole layer is covered by cudagraphs.
        """
        return [self]

    def _te_cuda_graph_capture(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Capture the selected computation, including its offload dependencies."""
        off_interface = self.get_offload_interface()
        record_offload = off_interface is not None and getattr(
            self, 'offload_module_in_cuda_graph', False
        )
        # Record the backward event on cuda graph stream in backward pass.
        # This is to ensure the main stream waits for computing on cuda graph stream to complete,
        # and overlaps with the H2D transfer on reload stream.
        if record_offload:
            hidden_states = off_interface.backward_record(hidden_states)
        cuda_graph_outputs = self._te_cuda_graph_capture_forward(hidden_states, *args, **kwargs)
        # Record the forward event on cuda graph stream for cuda graph capture.
        # This is to ensure the main stream waits for computing on cuda graph stream to complete,
        # and overlaps with the D2H transfer on offloading stream.
        if record_offload:
            off_interface.forward_record()
        return cuda_graph_outputs

    def _te_cuda_graph_capture_forward(self, *args, **kwargs):
        """Capture the entire module by forwarding its inputs to forward()."""
        return self.forward(*args, **kwargs)

    def _te_cuda_graph_replay(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Replay the graph and run its continuation with shared offload ordering."""
        replay_graph, after_replay = self._te_cuda_graph_prepare_replay(
            hidden_states, *args, **kwargs
        )
        off_interface = self.get_offload_interface()
        delay_offload_until_cuda_graph = (
            off_interface is not None and self.config.delay_offload_until_cuda_graph
        )
        if delay_offload_until_cuda_graph:
            off_interface.enter_replay()

        try:
            cuda_graph_output = replay_graph()
            # Flush delayed offload groups from previous layers after graph replay.
            # The CPU is idle during the sync between graph replay and a2a comm,
            # so we use that time to execute the delayed offload operations.
            if delay_offload_until_cuda_graph:
                off_interface.flush_delayed_groups()
            return after_replay(cuda_graph_output)
        finally:
            if delay_offload_until_cuda_graph:
                off_interface.exit_replay()

    def _te_cuda_graph_prepare_replay(self, hidden_states: torch.Tensor, **kwargs):
        """Bind tensor inputs for graph replay and its eager continuation."""
        return (
            partial(self._te_cuda_graph_replay_graph, hidden_states, **kwargs),
            self._te_cuda_graph_finish_replay,
        )

    def _te_cuda_graph_finish_replay(self, outputs):
        """Restore the public forward result; whole-module graphs need no adaptation."""
        return outputs

    def get_offload_interface(self):
        """Return the shared TE graph offload interface when activation offloading is enabled."""
        if self.config.fine_grained_activation_offloading:
            return TECudaGraphOffloadingInterface.get_instance()
        return None

    def _te_cuda_graph_replay_graph(self, hidden_states: torch.Tensor, **kwargs):
        """Prepare tensor inputs and replay this microbatch's graph."""
        microbatch_idx = getattr(self, 'te_cuda_graph_current_microbatch', 0)
        cg_index = self._te_cuda_graph_index(microbatch_idx)
        cudagraph_args, cudagraph_kwargs = self._get_te_cuda_graph_replay_args(
            hidden_states, **kwargs
        )

        for hook, hook_args in self.te_cuda_graph_manual_hooks:
            hook(*hook_args)
        return self.te_cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)

    def _get_te_cuda_graph_replay_args(self, hidden_states: torch.Tensor, **kwargs):
        """Validate the module's tensor inputs and add TE execution metadata."""
        assert isinstance(hidden_states, torch.Tensor), "CUDA graph accepts only Tensor inputs."
        for value in kwargs.values():
            assert value is None or isinstance(
                value, torch.Tensor
            ), "CUDA graph accepts only Tensor inputs."
        cudagraph_args = (hidden_states,)
        cudagraph_kwargs = kwargs.copy()
        cudagraph_kwargs['is_first_microbatch'] = (
            getattr(self, 'te_cuda_graph_current_microbatch', 0) == 0
        )
        if self.config.fine_grained_activation_offloading and getattr(
            self, 'offload_module_in_cuda_graph', False
        ):
            off_interface = self.get_offload_interface()

            # TE captures/replays the module on its own graph stream. Passing the
            # offload stream/event in lets TE order graph compute with D2H/H2D
            # transfers managed by the fine-grained offload manager.
            cudagraph_kwargs['cuda_graph_stream'] = off_interface.cuda_graph_stream()
            cudagraph_kwargs['cuda_graph_event'] = off_interface.cuda_graph_event()
        return cudagraph_args, cudagraph_kwargs

    def _should_call_te_cudagraph(self):
        """
        Check if we should call the TE cudagraph path.
        """
        if self.config.cuda_graph_impl != "transformer_engine":
            return False
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        return self.training and (is_graph_capturing() or self.te_cuda_graphs)


# ---------------------------------------------------------------------------
# set_te_cuda_graph_current_microbatch: sets per-layer microbatch index for TE graph replay
# ---------------------------------------------------------------------------


def set_te_cuda_graph_current_microbatch(model, microbatch_id):
    """Set the current microbatch on all layers that use TE CUDA graph replay.

    current_microbatch is read by _te_cuda_graph_replay to select the
    correct graph index.  This helper is called from the pipeline-parallel
    schedule before each forward step.

    Check each layer's backend: a multimodal model can use different backends
    for its language decoder and vision encoder.
    """
    decoder_exists = True
    model_with_decoder = None
    try:
        model_with_decoder = get_attr_wrapped_model(
            model, "decoder", allow_none=False, return_model_obj=True
        )
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and model_with_decoder is not None:
        for layer in model_with_decoder.decoder.layers:
            if layer.config.cuda_graph_impl == "transformer_engine":
                layer.te_cuda_graph_current_microbatch = microbatch_id
        if hasattr(model_with_decoder, 'mtp'):
            for layer in model_with_decoder.mtp.layers:
                assert hasattr(
                    layer, 'mtp_model_layer'
                ), f"MTP layer {layer} must have 'mtp_model_layer' attribute"
                if layer.mtp_model_layer.config.cuda_graph_impl == "transformer_engine":
                    layer.mtp_model_layer.te_cuda_graph_current_microbatch = microbatch_id

    # Also set current_microbatch on vision encoder layers so that
    # _te_cuda_graph_replay selects the correct graph index. Without this,
    # vision layers always use graph 0 (since current_microbatch defaults to 0),
    # causing all microbatch forwards to overwrite the same static buffers.
    # When backward runs for earlier microbatches, the buffers contain stale
    # data from later forwards, producing NaN gradients.
    try:
        model_with_vision = get_attr_wrapped_model(
            model, "vision_model", allow_none=True, return_model_obj=True
        )
    except RuntimeError:
        model_with_vision = None
    if model_with_vision is not None and hasattr(model_with_vision, 'vision_model'):
        vision_model = model_with_vision.vision_model
        if hasattr(vision_model, 'decoder') and hasattr(vision_model.decoder, 'layers'):
            for layer in vision_model.decoder.layers:
                if layer.config.cuda_graph_impl == "transformer_engine":
                    layer.te_cuda_graph_current_microbatch = microbatch_id
