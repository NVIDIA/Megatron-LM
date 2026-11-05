# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transformer Engine CUDA graph execution attached to Megatron modules."""

from __future__ import annotations

from functools import lru_cache, partial
from importlib import import_module

import torch

from megatron.core.utils import get_attr_wrapped_model


@lru_cache(maxsize=None)
def get_te_cuda_graph_module_class(module):
    """Select a TE layer class after the caller checks the configured backend.

    Preserve custom TransformerLayer subclasses and their method overrides.
    Other modules, including those already opting into the mixin, pass through.
    """
    from megatron.core.transformer.transformer_layer import (
        TECudaGraphTransformerLayer,
        TransformerLayer,
    )

    if not isinstance(module, type) or not issubclass(module, TransformerLayer):
        return module
    if issubclass(module, TECudaGraphMixin):
        return module
    if module is TransformerLayer:
        return TECudaGraphTransformerLayer

    # Keep the original subclass before the TE adapter so its __call__, input
    # preparation, and forward overrides retain precedence in the MRO.
    # Encode the original import path for reconstruction by pickle in another
    # process; dots in __qualname__ would otherwise mean nested attributes.
    qualname = (
        f"_te_cuda_graph_class_{module.__module__.encode().hex()}_"
        f"{module.__qualname__.encode().hex()}"
    )
    return type(
        f"TECudaGraph{module.__name__}",
        (module, TECudaGraphTransformerLayer),
        {"__module__": __name__, "__qualname__": qualname},
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


class TECudaGraphMixin:
    """Provide TE CUDA graph capture and replay for an opt-in Megatron module.

    Place this before the module base in the inheritance list and use cooperative
    initialization. The constructor forwards arguments unchanged, then registers
    TE graphs when cuda_graph_impl='transformer_engine'. Shared dispatch, capture,
    and replay forward module inputs to the adapter hooks. Adapters declare their
    input contracts on those hooks and can use copy_signature on thin wrappers.
    Graph-input preparation validates tensors and adds TE execution metadata.
    By default, one graph covers forward(). Adapters can override static-input and
    replay-argument preparation, capture_forward_hook, before_replay_hook, and
    after_replay_hook for a smaller scope and run the surrounding eager operations.
    before_replay_hook returns a graph callback and an output callback. Bind any
    continuation state into the output callback so it remains local to the call.
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
        record_offload = off_interface is not None and self.offload_module_in_cuda_graph
        # Record the backward event on cuda graph stream in backward pass.
        # This is to ensure the main stream waits for computing on cuda graph stream to complete,
        # and overlaps with the H2D transfer on reload stream.
        if record_offload:
            hidden_states = off_interface.backward_record(hidden_states)
        cuda_graph_outputs = self.capture_forward_hook(hidden_states, *args, **kwargs)
        # Record the forward event on cuda graph stream for cuda graph capture.
        # This is to ensure the main stream waits for computing on cuda graph stream to complete,
        # and overlaps with the D2H transfer on offloading stream.
        if record_offload:
            off_interface.forward_record()
        return cuda_graph_outputs

    def capture_forward_hook(self, *args, **kwargs):
        """Capture the entire module by forwarding its inputs to forward()."""
        return self.forward(*args, **kwargs)

    def _te_cuda_graph_replay(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Replay the graph and run its continuation with shared offload ordering."""
        replay_graph, after_replay = self.before_replay_hook(hidden_states, *args, **kwargs)
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

    def before_replay_hook(self, hidden_states: torch.Tensor, **kwargs):
        """Bind tensor inputs for graph replay and its eager continuation."""
        return (
            partial(self._te_cuda_graph_replay_graph, hidden_states, **kwargs),
            self.after_replay_hook,
        )

    def after_replay_hook(self, outputs):
        """Restore the public forward result; whole-module graphs need no adaptation."""
        return outputs

    def get_offload_interface(self):
        """Return the offload interface for modules that coordinate it with TE graphs."""
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
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )

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
