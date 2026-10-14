# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal Mixture-of-Kittens adapter for MCore MoE training experiments.

This module intentionally keeps the integration narrow: MCore owns routing,
parameters, DDP, the optimizer, and the logical checkpoint format, while MoK
replaces dispatch, routed/shared expert computation, and combine.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.distributed import ProcessGroup

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend
from megatron.core.transformer.moe.megakernel.mok.route_adapter import routing_map_to_mok_inputs
from megatron.core.transformer.moe.megakernel.mok.runtime import (
    _gate_up_main_grad_arguments,
    _gate_up_weight_arguments,
    _MoKAutograd,
)
from megatron.core.transformer.moe.megakernel.mok.weights import (
    _native_single_grouped_weight_view,
    _native_split_weight_view,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    dummy_weight_gradient_buffer as _dummy_weight_gradient_buffer,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    main_grad_buffer as _main_grad_buffer,
)

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


_get_main_grad = operator.attrgetter("main_grad")


@dataclass(frozen=True, eq=False)
class _MainGradArguments:
    """Validated MOK main-grad arguments and the buffers they were built from."""

    parameters: tuple[nn.Parameter, ...]
    main_grads: tuple[torch.Tensor, ...]
    data_ptrs: tuple[int, ...]
    arguments: tuple
    # ``arguments`` in MOK's six-buffer gate/up order (see _gate_up_main_grad_arguments).
    mok_arguments: tuple

    def is_current(self) -> bool:
        """Return whether every parameter still owns the same ``main_grad`` at the same address."""
        try:
            current = map(_get_main_grad, self.parameters)
            if not all(map(operator.is_, current, self.main_grads)):
                return False
        except AttributeError:
            return False
        return self.data_ptrs == tuple(map(torch.Tensor.data_ptr, self.main_grads))


class MoKMegakernel(MegakernelBackend):
    """Execute MOK using trainable parameters owned by native MCore modules."""

    def __init__(
        self,
        config: TransformerConfig,
        ep_group: ProcessGroup,
        routed_experts: nn.Module,
        shared_experts: nn.Module,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        try:
            # This adapter targets our MCore-integration version of MOK rather than the
            # upstream package API:
            # https://github.com/QiZhangNV/mixture-of-kittens/tree/qizhang/mcore-integration
            from mok.functional import MoKConfig
        except ImportError as exc:
            raise ImportError(
                "moe_megakernel_backend='mok' requires the MCore integration fork of "
                "mixture-of-kittens; see the MOK installation instructions in "
                "megatron/core/transformer/moe/README.md"
            ) from exc

        if not config.gradient_accumulation_fusion:
            raise ValueError("MOK currently requires gradient_accumulation_fusion=True")
        if config.moe_mlp_glu_interleave_size is not None:
            raise ValueError("MOK requires non-interleaved native MCore routed FC1 weights")
        if config.moe_shared_expert_glu_interleave_size is not None:
            raise ValueError("MOK requires non-interleaved native shared FC1 weights")
        if config.moe_pad_expert_input_to_capacity:
            raise ValueError(
                "MOK supports at most moe_router_topk logical routes per token; "
                "use MOK internal expert padding instead of moe_pad_expert_input_to_capacity"
            )
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size
        shared_intermediate_size = config.moe_shared_expert_intermediate_size
        if shared_intermediate_size != self.intermediate_size:
            raise ValueError(
                "MOK requires routed and shared experts to use the same intermediate "
                f"size, got routed={self.intermediate_size}, "
                f"shared={shared_intermediate_size}"
            )
        self.topk = config.moe_router_topk
        self.swiglu_limit = config.activation_func_clamp_value
        self.use_mxfp8_weights = bool(
            config.fp8 is not None and config.fp8_recipe == "mxfp8" and config.fp8_param
        )
        self.native_single_grouped_weights = bool(config.moe_single_grouped_weight)
        try:
            self.mok_config = MoKConfig(**(config.moe_megakernel_backend_config or {}))
        except TypeError as exc:
            raise ValueError(f"Invalid MOK option in moe_megakernel_backend_config: {exc}") from exc

        if not hasattr(routed_experts, "linear_fc1") or not hasattr(routed_experts, "linear_fc2"):
            raise ValueError(
                "MOK requires grouped routed experts exposing linear_fc1 and linear_fc2"
            )

        fc1 = routed_experts.linear_fc1
        fc2 = routed_experts.linear_fc2
        actual_single_grouped = bool(getattr(fc1, "single_grouped_weight", False))
        if actual_single_grouped != bool(getattr(fc2, "single_grouped_weight", False)):
            raise ValueError("MOK requires routed FC1 and FC2 to use the same weight layout")
        if actual_single_grouped != self.native_single_grouped_weights:
            raise ValueError(
                "MOK routed weight layout does not match config.moe_single_grouped_weight"
            )

        if self.native_single_grouped_weights:
            # Register aliases of the same Parameter objects so DDP's MOK forward
            # pre-hook waits for their overlap-param-gather buckets. named_parameters
            # deduplicates them; no additional payload storage is allocated.
            self.register_parameter("routed_fc1_weight", fc1.weight)
            self.register_parameter("routed_fc2_weight", fc2.weight)
        else:
            # Keep each native MCore expert parameter authoritative. MOK selects
            # its per-expert TMA descriptor inside the grouped-GEMM task, so no
            # dense expert-major payload or duplicate optimizer parameter is
            # required. The aliases let MOK participate in parameter-gather hooks;
            # the source expert module remains registered as the canonical owner.
            self._routed_fc1_parameter_names = []
            self._routed_fc2_parameter_names = []
            for expert_idx in range(self.num_local_experts):
                fc1_param = fc1.get_parameter(f"weight{expert_idx}")
                fc2_param = fc2.get_parameter(f"weight{expert_idx}")
                fc1_name = f"routed_fc1_weight{expert_idx}"
                fc2_name = f"routed_fc2_weight{expert_idx}"
                self.register_parameter(fc1_name, fc1_param)
                self.register_parameter(fc2_name, fc2_param)
                self._routed_fc1_parameter_names.append(fc1_name)
                self._routed_fc2_parameter_names.append(fc2_name)
            self._routed_fc1_parameter_names = tuple(self._routed_fc1_parameter_names)
            self._routed_fc2_parameter_names = tuple(self._routed_fc2_parameter_names)

        self._register_shared_weights(shared_experts, use_output_gate=config.moe_shared_expert_gate)

        # MegatronModule.set_is_first_microbatch discovers this attribute and resets it
        # once per optimizer iteration, matching TE's weight-cache lifecycle.
        self.is_first_microbatch = True
        # Cached MOK-consumable views, not weight copies: payloads alias MCore/TE
        # storage. BF16 weights update in place; MXFP8 converted scales refresh
        # once per optimizer iteration while retaining graph-captured addresses.
        self._routed_weight_view_cache = None
        # Host-side state resolved once instead of on every call, which matters with
        # 2 * num_local_experts split routed parameters. _reset_derived_state clears
        # these and the weight views.
        self._routed_parameter_cache = None
        self._routed_dummy_gradient_buffers = None
        self._main_grad_arguments_cache = None
        self._shared_fc1_gate_up_cache = None
        # Per-layer TE CUDA graphs with the moe scope replay MOK without Python. The layer then
        # refreshes the cached weight views eagerly before each replay (refresh_weight_views, after
        # the parameter all-gather hooks), so a capture must neither record nor consume the refresh.
        cuda_graph_modules = getattr(config, "cuda_graph_modules", None) or []
        self._weight_views_refreshed_outside_graph = (
            getattr(config, "cuda_graph_impl", "none") == "transformer_engine"
            and CudaGraphModule.moe in cuda_graph_modules
        )
        # Set once a CUDA graph captured the cached views: their addresses must stay valid.
        self._weight_views_captured = False

    def _routed_parameter_groups(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the (FC1, FC2, FC1 + FC2) routed parameter aliases, resolved once."""
        groups = self._routed_parameter_cache
        if groups is None:
            if self.native_single_grouped_weights:
                fc1 = (self.routed_fc1_weight,)
                fc2 = (self.routed_fc2_weight,)
            else:
                fc1 = tuple(getattr(self, name) for name in self._routed_fc1_parameter_names)
                fc2 = tuple(getattr(self, name) for name in self._routed_fc2_parameter_names)
            groups = (fc1, fc2, fc1 + fc2)
            self._routed_parameter_cache = groups
        return groups

    @property
    def routed_fc1_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the MCore-owned routed FC1 parameters."""
        return self._routed_parameter_groups()[0]

    @property
    def routed_fc2_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the MCore-owned routed FC2 parameters."""
        return self._routed_parameter_groups()[1]

    @property
    def autograd_routed_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return routed parameters tracked by the MOK autograd bridge."""
        return self._routed_parameter_groups()[2]

    def main_grad_arguments(self):
        """Return MOK logical main grads and optional per-expert descriptors.

        DDP assigns each ``main_grad`` once, so the validation and descriptor tables are reused
        until a parameter's ``main_grad`` is replaced or its storage moves (for example, a
        grad-buffer offload and reload).
        """
        return self._current_main_grad_arguments().arguments

    def mok_main_grad_arguments(self):
        """Return ``main_grad_arguments()`` in MOK's six-buffer gate/up order.

        The shared FC1 gate/up views are cached with the arguments, so MOK receives the same
        tensor objects on every call and validates them only once.
        """
        return self._current_main_grad_arguments().mok_arguments

    def _current_main_grad_arguments(self) -> _MainGradArguments:
        cached = self._main_grad_arguments_cache
        if cached is not None and cached.is_current():
            return cached

        routed = self.autograd_routed_parameters
        parameters = (self.shared_fc1_weight, self.shared_fc2_weight, *routed)
        main_grads = tuple(_main_grad_buffer(param) for param in parameters)
        num_routed_fc1 = len(self.routed_fc1_parameters)
        fc1_main_grads = main_grads[2 : 2 + num_routed_fc1]
        fc2_main_grads = main_grads[2 + num_routed_fc1 :]
        logical_main_grads = (main_grads[0], fc1_main_grads[0], main_grads[1], fc2_main_grads[0])
        if self.native_single_grouped_weights:
            arguments = (logical_main_grads, None)
        else:
            from mok import ops

            fc1_table = ops.make_routed_d_weight_storage_table(list(fc1_main_grads))
            fc2_table = ops.make_routed_d_weight_storage_table(list(fc2_main_grads))
            arguments = (logical_main_grads, (fc1_table, fc2_table))
        cached = _MainGradArguments(
            parameters=parameters,
            main_grads=main_grads,
            data_ptrs=tuple(grad.data_ptr() for grad in main_grads),
            arguments=arguments,
            mok_arguments=_gate_up_main_grad_arguments(*arguments, self.intermediate_size),
        )
        self._main_grad_arguments_cache = cached
        return cached

    def shared_fc1_gate_up(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the gate and up row views of the shared FC1 weight.

        The views are reused until the parameter or its storage changes, so MOK receives the same
        tensor objects on every call and validates them only once.
        """
        shared_fc1 = self.shared_fc1_weight
        cached = self._shared_fc1_gate_up_cache
        if cached is not None and cached[0] is shared_fc1 and cached[1] == shared_fc1.data_ptr():
            return cached[2]
        # Views of a detached alias: kernel arguments only, outside any autograd graph.
        views = _gate_up_weight_arguments(shared_fc1.detach(), None, self.intermediate_size)[:2]
        self._shared_fc1_gate_up_cache = (shared_fc1, shared_fc1.data_ptr(), views)
        return views

    def finish_routed_weight_gradients(self) -> tuple[torch.Tensor, ...]:
        """Finalize and return routed weight gradients.

        Same contract as ``parameter_bridge.finish_weight_gradient``, with each parameter's
        gradient-sentinel buffer looked up once rather than on every backward.
        """
        parameters = self.autograd_routed_parameters
        buffers = self._routed_dummy_gradient_buffers
        if buffers is None:
            buffers = tuple(_dummy_weight_gradient_buffer(param) for param in parameters)
            self._routed_dummy_gradient_buffers = buffers
        for param in parameters:
            param.grad_added_to_main_grad = True
        return tuple(map(torch.Tensor.detach, buffers))

    def _reset_derived_state(self) -> None:
        """Drop all state derived from the aliased parameters and their main grads."""
        if getattr(self, "_weight_views_captured", False):
            raise RuntimeError(
                "MOK weight views are referenced by captured CUDA graphs; delete the graphs "
                "(TECudaGraphHelper.delete_cuda_graphs) before reloading or moving the MOK "
                "parameters"
            )
        self._routed_weight_view_cache = None
        self._routed_parameter_cache = None
        self._routed_dummy_gradient_buffers = None
        self._main_grad_arguments_cache = None
        self._shared_fc1_gate_up_cache = None
        self.is_first_microbatch = True

    @torch.no_grad()
    def _register_shared_weights(self, shared: nn.Module, *, use_output_gate: bool = False) -> None:
        """Validate and alias MCore-owned native BF16 shared weights."""
        fc1_ref = shared.linear_fc1.weight
        fc2_ref = shared.linear_fc2.weight
        from megatron.core.fp8_utils import is_float8tensor

        i, h = self.intermediate_size, self.hidden_size
        if not isinstance(fc1_ref, nn.Parameter) or not isinstance(fc2_ref, nn.Parameter):
            raise RuntimeError("MOK shared FC1 and FC2 must be MCore-owned Parameters")
        if tuple(fc1_ref.shape) != (2 * i, h) or tuple(fc2_ref.shape) != (h, i):
            raise RuntimeError(
                "MOK requires native combined shared FC1/FC2 shapes "
                f"{(2 * i, h)} and {(h, i)}, got "
                f"{tuple(fc1_ref.shape)} and {tuple(fc2_ref.shape)}"
            )
        if (
            is_float8tensor(fc1_ref)
            or is_float8tensor(fc2_ref)
            or fc1_ref.dtype != torch.bfloat16
            or fc2_ref.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "MOK shared FC1 and FC2 must be constructed as native BF16 parameters"
            )
        if not fc1_ref.is_contiguous() or not fc2_ref.is_contiguous():
            raise RuntimeError("MOK shared FC1 and FC2 parameters must be contiguous")

        # These are aliases, not extra ownership. The native shared_experts
        # module remains registered and emits the canonical checkpoint entries.
        self.register_parameter("shared_fc1_weight", fc1_ref)
        self.register_parameter("shared_fc2_weight", fc2_ref)
        output_gate = getattr(shared, "gate_weight", None) if use_output_gate else None
        if use_output_gate and (
            not isinstance(output_gate, nn.Parameter)
            or is_float8tensor(output_gate)
            or output_gate.dtype != torch.bfloat16
            or tuple(output_gate.shape) != (1, h)
            or not output_gate.is_contiguous()
        ):
            raise RuntimeError(
                "MOK shared output gate must be a native contiguous BF16 Parameter "
                "with shape [1, H]"
            )
        # Include the native gate in parameter-gather hooks without adding a
        # second optimizer parameter or checkpoint entry.
        self.register_parameter("shared_output_gate_weight", output_gate)

    def refresh_weight_views(self) -> None:
        """Eagerly build or refresh the cached routed-weight views outside a CUDA graph.

        Called before every replay of a graph that contains this megakernel; it refreshes only
        when ``quantized_routed_weights`` would (first call after ``set_is_first_microbatch``).
        """
        if self._routed_weight_view_cache is None or (
            self.use_mxfp8_weights and self.is_first_microbatch
        ):
            self.quantized_routed_weights()

    def cuda_graphs_deleted(self) -> None:
        """No CUDA graph references the cached weight views anymore: allow resets again."""
        self._weight_views_captured = False

    @torch.no_grad()
    def quantized_routed_weights(self):
        """Alias native payloads and refresh the converted scales after parameter sync.

        With overlapped parameter gathering, DDP's pre-hook on our aliases finishes
        all-gather before forward.
        Grad-buffer reuse requantizes into TE storage and clears the staging buffer;
        the MOK payload views reference TE storage, not that temporary buffer.
        Scale conversion still needs to run after each optimizer update, in place
        so descriptors keep their addresses. It does not requantize the weights.
        """
        if (
            getattr(self, "_weight_views_refreshed_outside_graph", False)
            and torch.cuda.is_current_stream_capturing()
        ):
            # The graph reads the cached, address-stable views; refresh_weight_views updates them
            # before every replay. Leave is_first_microbatch for that eager refresh.
            if self._routed_weight_view_cache is None:
                raise RuntimeError("MOK weight views must be built before CUDA graph capture")
            self._weight_views_captured = True
            return self._routed_weight_view_cache
        needs_weight_view_update = self._routed_weight_view_cache is None or (
            self.use_mxfp8_weights and self.is_first_microbatch
        )
        if needs_weight_view_update:
            cached_fc1, cached_fc2 = (
                (None, None)
                if self._routed_weight_view_cache is None
                else self._routed_weight_view_cache
            )
            fc1_weight_view = self._routed_weight_view(
                self.routed_fc1_parameters,
                rows=2 * self.intermediate_size,
                columns=self.hidden_size,
                cached_view=cached_fc1,
            )
            fc2_weight_view = self._routed_weight_view(
                self.routed_fc2_parameters,
                rows=self.hidden_size,
                columns=self.intermediate_size,
                cached_view=cached_fc2,
            )
            self._routed_weight_view_cache = (fc1_weight_view, fc2_weight_view)

        self.is_first_microbatch = False
        return self._routed_weight_view_cache

    def _routed_weight_view(self, parameters, *, rows, columns, cached_view):
        """Select the storage adapter using this backend's layout and precision."""
        if self.native_single_grouped_weights:
            return _native_single_grouped_weight_view(
                parameters[0],
                num_experts=self.num_local_experts,
                rows=rows,
                columns=columns,
                use_mxfp8=self.use_mxfp8_weights,
                cached_view=cached_view,
            )
        return _native_split_weight_view(
            parameters,
            rows=rows,
            columns=columns,
            use_mxfp8=self.use_mxfp8_weights,
            cached_view=cached_view,
        )

    # Checkpoint contract: native MCore expert modules own the canonical parameters,
    # optimizer state, and checkpoint shards. This supports regular/distributed
    # save-resume when the backend, weight layout, and model/parallel configuration
    # are unchanged. Baseline<->MOK, single<->non-single, interleaved-layout, and
    # legacy-checkpoint conversion are deliberately outside this integration.
    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Emit no aliases; native expert modules own all checkpoint shards."""
        del prefix, sharded_offsets, metadata
        return {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Emit no aliases in regular state dicts either."""
        del destination, prefix, keep_vars

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Skip parameter aliases and invalidate state derived from native weights."""
        del state_dict, prefix, local_metadata, strict
        del missing_keys, unexpected_keys, error_msgs
        self._reset_derived_state()

    def _apply(self, *args, **kwargs):
        """Preserve externally attached parameter attributes across device/dtype moves.

        The routed/shared weights registered on this module are ALIASES of parameters owned by TE
        modules (``TEGroupedLinear`` etc.). ``torch.nn.Module._apply`` moves traceable wrapper
        subclasses (TE quantized tensors such as ``MXFP8Tensor``) with
        ``torch.utils.swap_tensors``, which exchanges the parameter object's whole ``__dict__``.
        TE's own modules re-attach the externally attached state afterwards
        (``TransformerEngineBaseModule._apply``), but this plain module's visit of the *same*
        object (e.g. during ``model.cuda()``) swaps it again and nothing restores it, so mcore's
        ``allreduce`` / ``partition_dim`` / ``tensor_model_parallel`` stamps and TE's
        ``_high_precision_init_val`` are lost. DDP / the layer-wise optimizer then treat the routed
        experts as data-parallel replicas (grads all-reduced across EP ranks, master weights
        sharded over dp_cp, identical experts across the EP group). Mirror TE's fix here.

        The conversion can also replace parameter objects or their storage, so all derived state
        is reset.
        """
        snapshots = {
            name: (param, dict(param.__dict__))
            for name, param in self._parameters.items()
            if param is not None
        }
        out = super()._apply(*args, **kwargs)
        for name, (old_param, attrs) in snapshots.items():
            new_param = self._parameters.get(name)
            if new_param is None:
                continue
            for key, value in attrs.items():
                if key in new_param.__dict__:
                    continue
                if isinstance(value, MethodType) and value.__self__ is old_param:
                    value = MethodType(value.__func__, new_param)
                setattr(new_param, key, value)
        self._reset_derived_state()
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the fused MoE megakernel (dispatch, routed/shared experts, combine).

        Args:
            hidden_states: Input activations of shape ``[..., hidden_size]``.
            probs: Router probabilities, flattened to ``[num_tokens, num_experts]``.
            routing_map: Token-to-expert routing map matching ``probs``.
            padding_mask: Optional batch-first ``[bsz, seq_length]`` mask of the sequence-first
                tokens, True for padding. Padding tokens get no routes, like dropless HybridEP:
                MOK skips them and their routed output is zero.

        Returns:
            MoE output with the same shape as ``hidden_states``.
        """
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        probs = probs.reshape(x.shape[0], -1)
        routing_map = routing_map.reshape(x.shape[0], -1)

        # Compact the authoritative route set directly into MOK's fixed [tokens, K]
        # representation. Missing routes, including all routes of padding tokens, are
        # encoded as expert -1 with zero weight.
        router_weights, top_experts = routing_map_to_mok_inputs(
            probs, routing_map, self.topk, padding_mask
        )

        output = _MoKAutograd.apply(
            self,
            x,
            router_weights,
            top_experts,
            self.shared_output_gate_weight,
            *self.autograd_routed_parameters,
            self.shared_fc1_weight,
            self.shared_fc2_weight,
        )
        return output.view(original_shape)
