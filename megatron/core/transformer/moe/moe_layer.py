# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch

from megatron.core import tensor_parallel, utils
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.transformer.moe.shortcut_cudagraph import (
    AsyncCombineToPersistentBuffer as _AsyncCombineToPersistentBuffer,
    AsyncDispatchToPersistentGradBuffers as _AsyncDispatchToPersistentGradBuffers,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    maybe_skip_or_early_return_by_cudagraph,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.shared_experts import (
    SharedExpertMLP,
    set_tensor_grad_fn_sequence_sr,
)
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.moe.token_dispatcher_inference import (
    NCCLAllGatherDispatcher,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import internal_api, nvtx_range_pop, nvtx_range_push

try:
    import flashinfer  # pylint: disable=unused-import

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

if HAVE_FLASHINFER:
    try:
        import flashinfer_cubin  # pylint: disable=unused-import
        import flashinfer_jit_cache  # pylint: disable=unused-import

        HAVE_FLASHINFER_CUBIN_AND_JIT_CACHE = True
    except ImportError:
        HAVE_FLASHINFER_CUBIN_AND_JIT_CACHE = False

try:
    import triton  # pylint: disable=unused-import

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, te_checkpoint
else:
    TELinear, te_checkpoint = None, None


class ExpertsInterface(Protocol):
    """Interface for the experts used in an MoELayer."""

    def forward(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the experts layer."""
        ...

    def backward_dw(self) -> None:
        """Backward pass to compute weight gradients for the experts."""
        ...


class ExpertsBuilder(Protocol):
    """Protocol for building the experts used in an MoELayer."""

    def __call__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        /,
        *,
        pg_collection: ProcessGroupCollection | None,
        name: str | None = None,
    ) -> ExpertsInterface: ...


class SharedExpertsInterface(Protocol):
    """Interface for the shared experts used in an MoELayer."""

    def forward(self, hidden_states: torch.Tensor, /) -> torch.Tensor:
        """Forward pass of the shared experts."""
        ...

    def backward_dw(self) -> None:
        """Backward pass to compute weight gradients for the shared experts."""
        ...


class SharedExpertsBuilder(Protocol):
    """Protocol for building the shared experts used in an MoELayer."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None,
        gate: bool,
        name: str | None = None,
    ) -> SharedExpertsInterface: ...


class RouterInterface(Protocol):
    """Interface for the router used in an MoELayer."""

    def forward(self, input: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the router.

        Returns:
            A tuple of (probabilities, routing_map).
        """
        ...

    def set_layer_number(self, layer_number: int) -> None:
        """Set the layer number for the router.

        Called from transformer_layer during initialization.
        """
        ...


class RouterBuilder(Protocol):
    """Protocol for building a Router."""

    def __call__(
        self, /, *, config: TransformerConfig, pg_collection: ProcessGroupCollection | None
    ) -> RouterInterface: ...


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: ExpertsBuilder
    shared_experts: SharedExpertsBuilder | None = None
    router: RouterBuilder = TopKRouter


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: RouterInterface = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    # Class-level CUDA stream for parallel ScMoE execution (shared across layers)
    _parallel_stream = None

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        self.submodules = not_none(submodules)
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        # If using mcore cudagraphs, recompute is handled by transformer_layer.MoETransformerLayer
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective'
            and "moe" in config.recompute_modules
            and config.cuda_graph_impl != 'local'
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        self.tp_group = pg_collection.tp

        # Initialize router.
        self.router = self.submodules.router(
            config=self.config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer
        )
        self.tp_group = pg_collection.tp

        # Initialize latent projections.
        if self.config.moe_latent_size:
            assert HAVE_TE, "TransformerEngine is required for MoE latent projections."
            if self.config.transformer_impl == "inference_optimized":
                from megatron.core.tensor_parallel.inference_layers import InferenceLinear

                linear_cls = InferenceLinear
            else:
                linear_cls = TELinear
            gtp_remat_group = (
                resolve_gtp_remat_group(pg_collection, is_expert=False)
                if "moe_latent_proj" in self.config.gtp_remat_opt_in_modules
                else None
            )
            self.fc1_latent_proj = linear_cls(
                self.config.hidden_size,
                self.config.moe_latent_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
                name=(name + ".fc1_latent_proj") if name is not None else None,
                gtp_remat_group=gtp_remat_group,
            )
            self.fc2_latent_proj = linear_cls(
                self.config.moe_latent_size,
                self.config.hidden_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
                name=(name + ".fc2_latent_proj") if name is not None else None,
                gtp_remat_group=gtp_remat_group,
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = self.submodules.experts(
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
            name=(name + ".experts") if name is not None else None,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            assert (
                self.submodules.shared_experts is not None
            ), "Shared experts builder is not provided in the module spec."
            self.shared_experts = self.submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        # Shortcut gating and normalization modules
        self._shortcut_scalar_gate = None
        self._shortcut_gate_vector = None
        self._shortcut_output_norm = None
        self._shortcut_tied_norm = None
        self._shortcut_untied_routed_norm = None
        self._shortcut_untied_shared_norm = None
        self._shortcut_post_norm = None

        if self.config.moe_shortcut_scalar_gate:
            self._shortcut_scalar_gate = torch.nn.Parameter(torch.zeros(1))

        if self.config.moe_shortcut_vector_gate:
            self._shortcut_gate_vector = torch.nn.Parameter(
                torch.zeros(self.config.hidden_size)
            )

        if (
            self.config.moe_shortcut_output_norm
            or self.config.moe_shortcut_tied_norm
            or self.config.moe_shortcut_untied_norm
            or self.config.moe_shortcut_post_norm
        ):
            # Use plain torch RMSNorm since the forward normalization is local to each token.
            eps = self.config.layernorm_epsilon
            if self.config.moe_shortcut_output_norm:
                self._shortcut_output_norm = torch.nn.RMSNorm(
                    self.config.hidden_size, eps=eps
                )
            if self.config.moe_shortcut_tied_norm:
                # One weight-tied norm reused for both parallel paths: LN(routed) + LN(shared).
                self._shortcut_tied_norm = torch.nn.RMSNorm(
                    self.config.hidden_size, eps=eps
                )
            if self.config.moe_shortcut_untied_norm:
                # Two independent norms, one per parallel path: LN_r(routed) + LN_s(shared).
                self._shortcut_untied_routed_norm = torch.nn.RMSNorm(
                    self.config.hidden_size, eps=eps
                )
                self._shortcut_untied_shared_norm = torch.nn.RMSNorm(
                    self.config.hidden_size, eps=eps
                )
            if self.config.moe_shortcut_post_norm:
                self._shortcut_post_norm = torch.nn.RMSNorm(
                    self.config.hidden_size, eps=eps
                )

        # These parameters are replicated across TP ranks, but under sequence parallel each
        # rank sees a different sequence shard. Mark them so finalize_model_grads performs the
        # required TP gradient all-reduce, just as it does for router and normalization weights.
        replicated_parameters = [self._shortcut_scalar_gate, self._shortcut_gate_vector]
        replicated_modules = [
            self._shortcut_output_norm,
            self._shortcut_tied_norm,
            self._shortcut_untied_routed_norm,
            self._shortcut_untied_shared_norm,
            self._shortcut_post_norm,
        ]
        for parameter in replicated_parameters:
            if parameter is not None:
                setattr(parameter, 'sequence_parallel', self.config.sequence_parallel)
        for module in replicated_modules:
            if module is not None:
                for parameter in module.parameters():
                    setattr(parameter, 'sequence_parallel', self.config.sequence_parallel)

        # Inference-optimized mode setup
        if config.transformer_impl == "inference_optimized":
            if config.inference_grouped_gemm_backend == 'auto':
                assert HAVE_FLASHINFER, (
                    "inference_grouped_gemm_backend='auto'"
                    "requires flashinfer-python. "
                    "Install flashinfer-python or set "
                    "inference_grouped_gemm_backend to 'torch' or 'te'."
                )

                # Verify that pre-compiled FlashInfer CUTLASS kernels are available
                # when using the FlashInfer backend. The flashinfer-jit-cache package
                # must be installed ahead of time to avoid a multi-minute JIT
                # compilation step at runtime.
                from megatron.core.inference.utils import check_flashinfer_jit_cache_installed

                check_flashinfer_jit_cache_installed()
            elif config.inference_grouped_gemm_backend == 'torch':
                assert hasattr(torch.nn.functional, 'grouped_mm') or hasattr(
                    torch, '_grouped_mm'
                ), (
                    "inference_grouped_gemm_backend='torch' requires "
                    "torch.nn.functional.grouped_mm (> torch 2.10) or torch._grouped_mm (<= 2.10)."
                )
            elif config.inference_grouped_gemm_backend == 'vllm':
                assert HAVE_TRITON, (
                    "inference_grouped_gemm_backend='vllm' requires Triton. "
                    "Install triton (pip install triton)."
                )
            self._setup_inference_mode(pg_collection)

        # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()
        self.fwd_execution_map = ["route", "expert_compute", "postprocess"]

        # Cached outputs for parallel ScMoE A2A-only async execution
        self._cached_dispatch_output = None
        self._cached_combine_output = None

        # Setup events and streams for delayed wgrad computation.
        self.setup_delayed_wgrad_for_dispatch_backward_overlap()

    def _setup_inference_mode(self, pg_collection):
        """Set up inference-optimized token dispatcher.

        Called from __init__ when config.transformer_impl == "inference_optimized".
        Stores the training dispatcher and creates the inference dispatcher selected
        by config.inference_moe_token_dispatcher_type ('nccl' or 'nvls').
        The active dispatcher is selected at the start of `forward` based on
        `InferenceMode.is_active()`.
        """
        dispatcher_type = self.config.inference_moe_token_dispatcher_type
        dispatcher_cls = (
            NVLSAllGatherVDispatcher if dispatcher_type == 'nvls' else NCCLAllGatherDispatcher
        )

        self._training_token_dispatcher = self.token_dispatcher
        self._inference_token_dispatcher = dispatcher_cls(
            self.num_local_experts,
            self.local_expert_indices,
            config=self.config,
            pg_collection=pg_collection,
        )

        # Wire shared-expert overlap into the inference dispatcher (NVLS only).
        # The dispatcher launches the shared-expert forward on SharedExpertMLP.stream
        # concurrently with AGV+experts+RSV and adds it back in combine_postprocess.
        if (
            dispatcher_type == 'nvls'
            and self.use_shared_expert
            and self.config.moe_shared_expert_overlap
        ):
            self._inference_token_dispatcher.set_shared_experts(self.shared_experts)
            # With MoE latent projections, the shared expert must run on the full
            # hidden_states (pre-latent) and its output added post-fc2_latent_proj.
            # The dispatcher only sees latent-dim tensors, so we move the launch+add
            # into preprocess/postprocess on the layer and tell the dispatcher to
            # skip its own internal launch+add.
            if self.config.moe_latent_size:
                self._inference_token_dispatcher._external_shared_expert_launch = True
        # Inference only: side-stream shared-expert output for latent-MoE + NVLS overlap
        # (preprocess launches on SharedExpertMLP.stream; postprocess joins+adds).
        self._latent_shared_expert_output: Optional[torch.Tensor] = None

    def setup_delayed_wgrad_for_dispatch_backward_overlap(self):
        """Initializes CUDA events and streams for overlapping expert
        weight gradient computation with dispatch backward.
        """
        self._delayed_wgrad_event: Optional[torch.cuda.Event] = None
        self._delayed_wgrad_stream: Optional[torch.cuda.Stream] = None
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            self._delayed_wgrad_event = torch.cuda.Event()
            self._delayed_wgrad_stream = torch.cuda.Stream(device="cuda")

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """Compute token routing for preprocessing.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping.
        """
        probs, routing_map = apply_module(self.router)(hidden_states, padding_mask)
        return probs, routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        shared_expert_input: Optional[torch.Tensor] = None,
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher.

        Args:
            shared_expert_input: When ScMoE is active and shared_expert_overlap is enabled,
                this provides the current layer's hidden states for the shared expert (since
                hidden_states here is the shortcut/routed input).
        """
        # Latent-MoE + NVLS-inference shared-expert overlap: launch the shared
        # expert on its side stream BEFORE fc1_latent_proj so it sees the full
        # hidden_states. The corresponding join+add runs in postprocess after
        # fc2_latent_proj. Skipped on the training / NCCL paths.
        if (
            self.config.moe_latent_size
            and self.shared_expert_overlap
            and isinstance(self.token_dispatcher, NVLSAllGatherVDispatcher)
        ):
            stream = SharedExpertMLP.stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._latent_shared_expert_output = apply_module(self.shared_experts)(hidden_states)
        elif self.config.moe_latent_size:
            if self.shared_expert_overlap:
                if self.training:
                    raise AssertionError(
                        "Shared expert overlap with MoE latent projections is not supported "
                        "during training. Disable moe_shared_expert_overlap."
                    )
                raise AssertionError(
                    "Shared expert overlap with MoE latent projections requires the NVLS "
                    "inference dispatcher. Either disable moe_shared_expert_overlap or set "
                    "inference_moe_token_dispatcher_type='nvls'."
                )
        # Project the hidden_states from hidden dimension down to latent dimension.
        if self.config.moe_latent_size:
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs, shared_expert_input=shared_expert_input
        )
        return hidden_states, probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            hidden_states = _RegisterDelayedWgradForExperts.apply(self, hidden_states)
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8 or self.config.fp4:
                    shared_expert_output = te_checkpoint(
                        apply_module(self.shared_experts),
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        self.tp_group,
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        apply_module(self.shared_experts), False, hidden_states
                    )
            else:
                shared_expert_output = apply_module(self.shared_experts)(hidden_states)

        return shared_expert_output

    @internal_api
    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        if self.config.overlap_dispatch_backward_with_experts_wgrad:
            hidden_states = _RecordExpertDgradCompletion.apply(
                self._delayed_wgrad_event, hidden_states
            )
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        if hasattr(self, "_inference_token_dispatcher") and InferenceMode.is_active():
            routing_map = self.token_dispatcher.routing_map
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs, routing_map=routing_map
            )
        else:
            # NCCL-EP zero-copy: experts write fc2 output and fc1 dgrad straight into the combine /
            # dispatch symm buffers. Passed only when set (non-TEGroupedMLP experts don't accept
            # these kwargs).
            output_buffer, grad_input_buffer = self.token_dispatcher.get_expert_zero_copy_buffers()
            expert_kwargs = {}
            if output_buffer is not None:
                expert_kwargs["output_buffer"] = output_buffer
            if grad_input_buffer is not None:
                expert_kwargs["grad_input_buffer"] = grad_input_buffer
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs, **expert_kwargs
            )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication).
        """
        output = self.token_dispatcher.token_combine(output)
        return output

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Project the output back from latent dimension to hidden dimension after combine
        in latent dimension if needed. Combine expert output with shared_experts if needed.

        Operation order:
          1. combine_postprocess (un-permute, weight)
          2. latent projection (if moe_latent_size)
          3. routed-path norm: output_norm (legacy routed-only) or tied_norm or untied routed norm
          4. scalar_gate: sigmoid(alpha) * output (if moe_shortcut_scalar_gate)
          5. shared-path norm: tied_norm (same module as step 3) or untied shared norm. tied_norm
             reuses one weight-tied module for both paths; untied_norm uses two separate modules.
          6. combine with shared: vector_gate interpolation or plain addition
          7. add zero expert output
          8. post_norm (if moe_shortcut_post_norm)

        _latent_shared_expert_output is inference-only (latent-MoE + NVLS dispatcher with
        shared-expert overlap). It is populated in preprocess and joined here, after
        fc2_latent_proj, so the dimensions match the full hidden dim."""

        output, deferred_shared_expert_output = self.token_dispatcher.combine_postprocess(output)
        if deferred_shared_expert_output is not None:
            # Dispatcher computed the shared-expert output (via overlap) but deferred the
            # combine so we can apply the vector gate below instead of a plain sum.
            shared_expert_output = deferred_shared_expert_output
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)

        # Normalize routed path before combining. output_norm: legacy routed-only norm.
        # tied_norm: one shared norm for both paths. untied_norm: per-path routed norm.
        if self._shortcut_output_norm is not None:
            output = self._shortcut_output_norm(output)
        elif self._shortcut_tied_norm is not None:
            output = self._shortcut_tied_norm(output)
        elif self._shortcut_untied_routed_norm is not None:
            output = self._shortcut_untied_routed_norm(output)

        # Scale routed output with learned scalar gate
        if self._shortcut_scalar_gate is not None:
            output = torch.sigmoid(self._shortcut_scalar_gate) * output

        # Normalize shared path before combining. tied_norm reuses the SAME module as the routed
        # path above; untied_norm uses its own separate module.
        if self._shortcut_tied_norm is not None and shared_expert_output is not None:
            shared_expert_output = self._shortcut_tied_norm(shared_expert_output)
        elif self._shortcut_untied_shared_norm is not None and shared_expert_output is not None:
            shared_expert_output = self._shortcut_untied_shared_norm(shared_expert_output)

        # Combine with shared expert output
        if self._shortcut_gate_vector is not None and shared_expert_output is not None:
            gate = torch.sigmoid(self._shortcut_gate_vector)
            output = gate * output + (1 - gate) * shared_expert_output
        elif shared_expert_output is not None:
            output = output + shared_expert_output
        elif (
            isinstance(self.token_dispatcher, NVLSAllGatherVDispatcher)
            and self._latent_shared_expert_output is not None
        ):
            # This codepath is for inference-only shared-expert overlap of latent MoEs.
            # Must happen post-fc2_latent_proj so dimensions match.
            torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
            output = output + self._latent_shared_expert_output
            self._latent_shared_expert_output = None

        # Normalize combined output
        if self._shortcut_post_norm is not None:
            output = self._shortcut_post_norm(output)

        return output

    def shortcut_graph_participants(
        self,
    ) -> tuple[tuple[torch.nn.Module, ...], tuple[torch.nn.Parameter, ...]]:
        """Return modules and standalone parameters used by shortcut postprocess."""
        modules = []
        if self.config.moe_latent_size:
            modules.append(self.fc2_latent_proj)
        modules.extend(
            module
            for module in (
                self._shortcut_output_norm,
                self._shortcut_tied_norm,
                self._shortcut_untied_routed_norm,
                self._shortcut_untied_shared_norm,
                self._shortcut_post_norm,
            )
            if module is not None
        )
        parameters = tuple(
            parameter
            for parameter in (self._shortcut_scalar_gate, self._shortcut_gate_vector)
            if parameter is not None
        )
        return tuple(modules), parameters

    def launch_dispatch_async(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        ready_event: torch.cuda.Event = None,
        route_grad_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        route_grad_ready_event: Optional[torch.cuda.Event] = None,
        backward_dependency: Optional[torch.Tensor] = None,
    ):
        """Launch ONLY the A2A dispatch on a side CUDA stream.

        Only the NCCL All-to-All collective runs on the side stream. NCCL uses
        dedicated NVLink/NIC hardware that overlaps with SM compute regardless of
        CUDA_DEVICE_MAX_CONNECTIONS=1. All other compute stays on the main stream.

        Args:
            hidden_states: Pre-processed routed input (after preprocess/dispatch_preprocess).
            probs: Routing probabilities.
            ready_event: Event recorded when the forward inputs are ready for the side stream.
            route_grad_buffers: Stable destinations for dispatch backward's two input gradients.
            route_grad_ready_event: Event recorded after both gradient buffers are populated.
            backward_dependency: CUDA-graph output whose backward must wait for dispatch backward.
        """
        if MoELayer._parallel_stream is None:
            # High-priority side stream so the hybrid-ep A2A collectives are not
            # starved behind main-stream compute
            MoELayer._parallel_stream = torch.cuda.Stream(priority=-1)

        s = MoELayer._parallel_stream

        if ready_event is not None:
            s.wait_event(ready_event)
        else:
            s.wait_stream(torch.cuda.current_stream())

        hidden_states.record_stream(s)
        probs.record_stream(s)

        if route_grad_buffers is not None:
            if route_grad_ready_event is None or backward_dependency is None:
                raise ValueError(
                    "Persistent async dispatch requires a backward dependency and "
                    "route-gradient-ready event"
                )
            dispatched_input, dispatched_probs = _AsyncDispatchToPersistentGradBuffers.apply(
                hidden_states,
                probs,
                backward_dependency,
                self,
                s,
                route_grad_buffers[0],
                route_grad_buffers[1],
                route_grad_ready_event,
            )
        else:
            with torch.cuda.stream(s):
                dispatched_input, dispatched_probs = self.dispatch(hidden_states, probs)
        self._cached_dispatch_output = (dispatched_input, dispatched_probs)

    def wait_dispatch(self):
        """Wait for A2A dispatch to complete and return results to the main stream.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (dispatched_input, probs)
        """
        s = MoELayer._parallel_stream
        torch.cuda.current_stream().wait_stream(s)

        dispatched_input, dispatched_probs = self._cached_dispatch_output
        self._cached_dispatch_output = None
        # The outputs were allocated on the side stream and are consumed on the
        # main stream. Keep their storage alive through those main-stream consumers.
        main_stream = torch.cuda.current_stream()
        dispatched_input.record_stream(main_stream)
        dispatched_probs.record_stream(main_stream)

        return dispatched_input, dispatched_probs

    def launch_combine_async(
        self,
        output: torch.Tensor,
        persistent_output_factory: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ready_event: torch.cuda.Event | None = None,
        grad_ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """Launch ONLY the A2A combine on a side CUDA stream.

        Only the NCCL All-to-All collective runs on the side stream, overlapping
        with shared expert compute on the main stream.

        Args:
            output: Expert output ready for combine (after combine_preprocess).
            persistent_output_factory: Callable returning stable storage for the combined
                output. The copy into that storage stays connected to autograd.
            ready_event: Optional event recorded after the persistent output is populated.
            grad_ready_event: Event recorded inside the fused backward graph when the gradient
                for the combined output is ready for combine backward.
        """
        combine_stream = MoELayer._parallel_stream
        combine_stream.wait_stream(torch.cuda.current_stream())
        output.record_stream(combine_stream)

        if persistent_output_factory is not None:
            if ready_event is None or grad_ready_event is None:
                raise ValueError(
                    "Persistent async combine requires forward-ready and gradient-ready events"
                )
            # Apply this bridge on main so its autograd node has the fused graph's canonical
            # stream. The bridge itself launches combine forward/backward on the side stream.
            combined = _AsyncCombineToPersistentBuffer.apply(
                output,
                self,
                combine_stream,
                persistent_output_factory,
                ready_event,
                grad_ready_event,
            )
            from megatron.core.transformer.cuda_graphs import mark_cuda_graph_prebound_input

            mark_cuda_graph_prebound_input(combined)
            set_tensor_grad_fn_sequence_sr(combined, torch.iinfo(torch.int).max)
        else:
            with torch.cuda.stream(combine_stream):
                combined = self.combine(output)
                if ready_event is not None:
                    ready_event.record(combine_stream)
        # The persistent path returns the tensor directly to the fused graph. Keeping an
        # additional cache reference would retain its private combine autograd graph until the
        # next forward. The eager path is retrieved later through wait_combine().
        self._cached_combine_output = combined if persistent_output_factory is None else None
        return combined

    def wait_combine(self):
        """Wait for A2A combine to complete and return results to the main stream.

        Returns:
            torch.Tensor: Combined expert output after A2A.
        """
        s = MoELayer._parallel_stream
        torch.cuda.current_stream().wait_stream(s)

        combined = self._cached_combine_output
        self._cached_combine_output = None

        combined.record_stream(torch.cuda.current_stream())
        # Boost the A2A_combine grad_fn's sequence_nr so autograd dispatches
        # the side-stream combine backward BEFORE walking the shared_expert
        # backward chain on main. Without this, natural sr puts shared_expert
        # (forward later) ahead of combine_A2A → by the time A2A is queued on
        # side, shared_expert has already finished on main → no overlap.
        set_tensor_grad_fn_sequence_sr(combined, torch.iinfo(torch.int).max)
        return combined

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """This method is a combined method of route and preprocess. Deprecated."""

        probs, routing_map = self.route(hidden_states)
        hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
        return hidden_states, probs, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        shortcut_input: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.
        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        # Select the active token dispatcher based on whether the inference engine
        # is currently using the model. Only applies when the inference dispatcher
        # was set up (config.transformer_impl == "inference_optimized").
        if hasattr(self, "_inference_token_dispatcher"):
            if InferenceMode.is_active():
                self.token_dispatcher = self._inference_token_dispatcher
                self.shared_expert_overlap = (
                    self._inference_token_dispatcher.shared_experts is not None
                )
            else:
                self.token_dispatcher = self._training_token_dispatcher
                self.shared_expert_overlap = self.config.moe_shared_expert_overlap
        # Transpose from [bsz, seq_length] to [seq_length, bsz] to align with hidden_states
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(
            hidden_states, intermediate_tensors=None, padding_mask=None, shortcut_input=None
        ):
            # ScMoE: route on shortcut, shared expert on current hidden_states
            routed_input = shortcut_input if shortcut_input is not None else hidden_states

            try:
                if "route" in self.fwd_execution_map:
                    shared_expert_output = self.shared_experts_compute(hidden_states)
                    probs, routing_map = self.route(routed_input, padding_mask)
                    routed_input, probs = self.preprocess(
                        routed_input,
                        probs,
                        routing_map,
                        shared_expert_input=(
                            hidden_states
                            if (shortcut_input is not None and self.shared_expert_overlap)
                            else None
                        ),
                    )

                    if intermediate_tensors is not None:
                        return routed_input, probs, shared_expert_output

            except MoECudaGraphPartialCaptureSignal as e:
                # This signal is raised from the maybe_skip_or_early_return_by_cudagraph decorator.
                # It means we should early-return from the MoE layer forward pass.
                # This happens when we are partially capturing the CUDA graph of the MoE layer,
                # like cuda_graph_modules=["moe_router", "moe_preprocess"].
                # We need to return the intermediate tensors as CUDA graph outputs.
                return e.get_early_return_outputs(routed_input, shared_expert_output)

            if "expert_compute" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    routed_input, probs = intermediate_tensors

                dispatched_input, probs = self.dispatch(routed_input, probs)
                output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
                assert (
                    mlp_bias is None
                ), f"mlp_bias is not supported for {type(self.token_dispatcher)}"
                output = self.combine(output)

                if intermediate_tensors is not None:
                    return output, mlp_bias

            if "postprocess" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    output, shared_expert_output = intermediate_tensors

                output = self.postprocess(output, shared_expert_output)

                if intermediate_tensors is not None:
                    return output

            return output, mlp_bias

        if self.moe_layer_recompute and self.training:
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                    shortcut_input,
                )
            else:
                outputs = tensor_parallel.checkpoint(
                    custom_forward,
                    False,
                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                    shortcut_input,
                )
        else:
            outputs = custom_forward(
                hidden_states, intermediate_tensors, padding_mask, shortcut_input
            )

        return outputs

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        """Compute weight gradients for experts and shared experts."""
        from megatron.core.pipeline_parallel.utils import get_comm_stream

        # TODO(Wohox): replace the "routed_experts" and "shared_experts" arguments with better
        # naming to better explain that they are actually from different fine-grained callables,
        # or use scanning to decide which backward_dw should be called.
        if routed_experts:
            self.experts.backward_dw()
            if self.config.moe_latent_size and self.config.overlap_moe_expert_parallel_comm:
                # TODO(Wohox): fc2_latent_proj forward and backward are executed in comm stream,
                # so we execute its backward_dw in the comm stream too. But this may harm the
                # EP overlap performance. Better to check if there is a better way to handle this.
                comm_stream = get_comm_stream()
                with torch.cuda.stream(comm_stream):
                    self.fc2_latent_proj.backward_dw()
        if shared_experts:
            if self.use_shared_expert and not self.shared_expert_overlap:
                self.shared_experts.backward_dw()
            if self.config.moe_latent_size and self.config.overlap_moe_expert_parallel_comm:
                self.fc1_latent_proj.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)


class _RecordExpertDgradCompletion(torch.autograd.Function):
    """Autograd function that records a CUDA event when expert data gradients finish.

    Placed in the forward graph just before the expert computation so that during
    the backward pass, when the expert dgrad completes, we record an event. The
    subsequent ``_RegisterDelayedWgradForExperts`` waits on this event before
    launching the delayed wgrad computation on a separate CUDA stream.
    """

    @staticmethod
    def forward(ctx, event: torch.cuda.Event, *inputs):
        """Forward pass that stores the event and passes through inputs unchanged."""
        ctx.event = event
        return inputs[0] if len(inputs) == 1 else inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass that records the event when expert dgrad completes."""
        ctx.event.record(torch.cuda.current_stream())
        ctx.event = None
        return (None,) + grad_outputs


class _RegisterDelayedWgradForExperts(torch.autograd.Function):
    """Autograd function that orchestrates delayed wgrad computation for MoE experts.

    Placed in the forward graph at the dispatch boundary. During the backward pass,
    this function:
      1. Records an event on the current (backward) stream to signal the dgrad is done.
      2. Executes the delayed wgrad computation on a dedicated CUDA stream.
      3. Waits for the wgrad computation to complete.
      4. Invokes the registered gradient processing callback (e.g., FSDP reduce-scatter).
    """

    @staticmethod
    def forward(ctx, module: MoELayer, *inputs):
        """Forward pass that stores the MoE module and passes through inputs unchanged."""
        ctx.module = module
        return inputs[0] if len(inputs) == 1 else inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass that executes delayed wgrad computation on a separate stream."""
        module = ctx.module
        event = module._delayed_wgrad_event
        wgrad_stream = module._delayed_wgrad_stream

        wgrad_stream.wait_event(event)
        with torch.cuda.stream(wgrad_stream):
            nvtx_range_push("delayed_expert_wgrad")
            module.backward_dw(routed_experts=True, shared_experts=False)
            nvtx_range_pop("delayed_expert_wgrad")
            event.record(wgrad_stream)

        torch.cuda.current_stream().wait_event(event)

        for param in module.parameters():
            if getattr(param, "post_wgrad_grad_acc_hook", None) is not None:
                param.post_wgrad_grad_acc_hook()

        ctx.module = None
        return (None,) + grad_outputs
