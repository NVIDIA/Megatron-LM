# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal Mixture-of-Kittens adapter for MCore MoE training experiments.

This module intentionally keeps the integration narrow: MCore owns routing,
parameters, DDP, the optimizer, and the logical checkpoint format, while MoK
replaces dispatch, routed/shared expert computation, and combine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed import ProcessGroup

from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend
from megatron.core.transformer.moe.megakernel.mok.route_adapter import routing_map_to_mok_inputs
from megatron.core.transformer.moe.megakernel.mok.runtime import _MoKAutograd
from megatron.core.transformer.moe.megakernel.mok.weights import (
    _native_single_grouped_weight_view,
    _native_split_weight_view,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    finish_weight_gradient as _finish_weight_gradient,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    main_grad_buffer as _main_grad_buffer,
)

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


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
            from mok.functional import MoKConfig
        except ImportError as exc:
            raise ImportError(
                "moe_megakernel_backend='mok' requires the latest mixture-of-kittens package "
                "on PYTHONPATH"
            ) from exc

        # TODO: Add a materialized-wgrad adapter before allowing non-fused accumulation.
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
        if config.moe_shared_expert_gate:
            raise ValueError("MOK does not support MCore's optional shared-expert output gate")

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
        self.mok_config = MoKConfig(
            fwd_num_comm_sms=config.mok_fwd_num_comm_sms,
            bwd_num_comm_sms=config.mok_bwd_num_comm_sms,
            minibatch_size=config.mok_minibatch_size,
            macrobatch_size=config.mok_macrobatch_size,
            schedule_capacity_multiplier=config.mok_schedule_capacity_multiplier,
            all_gather_top_experts_chunk_bytes=config.mok_all_gather_top_experts_chunk_bytes,
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

        self._register_shared_weights(shared_experts)

        # MegatronModule.set_is_first_microbatch discovers this attribute and resets it
        # once per optimizer iteration, matching TE's weight-cache lifecycle.
        self.is_first_microbatch = True
        # Cached MOK-consumable views, not weight copies: payloads alias MCore/TE
        # storage. BF16 weights update in place; MXFP8 converted scales refresh
        # once per optimizer iteration while retaining graph-captured addresses.
        self._routed_weight_view_cache = None
        self._split_main_grad_descriptor_cache = None

    @property
    def routed_fc1_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.native_single_grouped_weights:
            return (self.routed_fc1_weight,)
        return tuple(getattr(self, name) for name in self._routed_fc1_parameter_names)

    @property
    def routed_fc2_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.native_single_grouped_weights:
            return (self.routed_fc2_weight,)
        return tuple(getattr(self, name) for name in self._routed_fc2_parameter_names)

    @property
    def autograd_routed_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.routed_fc1_parameters + self.routed_fc2_parameters

    def main_grad_arguments(self):
        """Return MOK logical main grads and optional per-expert descriptors."""
        shared_fc1_grad = _main_grad_buffer(self.shared_fc1_weight)
        shared_fc2_grad = _main_grad_buffer(self.shared_fc2_weight)
        fc1_main_grads = tuple(_main_grad_buffer(param) for param in self.routed_fc1_parameters)
        fc2_main_grads = tuple(_main_grad_buffer(param) for param in self.routed_fc2_parameters)
        main_grads = (shared_fc1_grad, fc1_main_grads[0], shared_fc2_grad, fc2_main_grads[0])
        if self.native_single_grouped_weights:
            return main_grads, None

        from mok import ops

        fingerprint = tuple(
            (grad.data_ptr(), grad.dtype, tuple(grad.shape))
            for grad in fc1_main_grads + fc2_main_grads
        )
        if (
            self._split_main_grad_descriptor_cache is None
            or self._split_main_grad_descriptor_cache[0] != fingerprint
        ):
            fc1_table = ops.make_routed_d_weight_storage_table(list(fc1_main_grads))
            fc2_table = ops.make_routed_d_weight_storage_table(list(fc2_main_grads))
            self._split_main_grad_descriptor_cache = (fingerprint, (fc1_table, fc2_table))
        return main_grads, self._split_main_grad_descriptor_cache[1]

    def finish_routed_weight_gradients(self) -> tuple[torch.Tensor, ...]:
        return tuple(_finish_weight_gradient(param) for param in self.autograd_routed_parameters)

    @torch.no_grad()
    def _register_shared_weights(self, shared: nn.Module) -> None:
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

    @torch.no_grad()
    def quantized_routed_weights(self):
        """Build or refresh cached MOK routed-weight views."""
        needs_weight_view_update = self._routed_weight_view_cache is None or (
            self.use_mxfp8_weights and self.is_first_microbatch
        )
        if needs_weight_view_update:
            cached_fc1, cached_fc2 = (
                (None, None)
                if self._routed_weight_view_cache is None
                else self._routed_weight_view_cache
            )
            if self.native_single_grouped_weights:
                fc1_weight_view = _native_single_grouped_weight_view(
                    self.routed_fc1_parameters[0],
                    num_experts=self.num_local_experts,
                    rows=2 * self.intermediate_size,
                    columns=self.hidden_size,
                    use_mxfp8=self.use_mxfp8_weights,
                    cached_view=cached_fc1,
                )
                fc2_weight_view = _native_single_grouped_weight_view(
                    self.routed_fc2_parameters[0],
                    num_experts=self.num_local_experts,
                    rows=self.hidden_size,
                    columns=self.intermediate_size,
                    use_mxfp8=self.use_mxfp8_weights,
                    cached_view=cached_fc2,
                )
            else:
                fc1_weight_view = _native_split_weight_view(
                    self.routed_fc1_parameters,
                    rows=2 * self.intermediate_size,
                    columns=self.hidden_size,
                    use_mxfp8=self.use_mxfp8_weights,
                    cached_view=cached_fc1,
                )
                fc2_weight_view = _native_split_weight_view(
                    self.routed_fc2_parameters,
                    rows=self.hidden_size,
                    columns=self.intermediate_size,
                    use_mxfp8=self.use_mxfp8_weights,
                    cached_view=cached_fc2,
                )
            self._routed_weight_view_cache = (fc1_weight_view, fc2_weight_view)

        self.is_first_microbatch = False
        return self._routed_weight_view_cache

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
        self._routed_weight_view_cache = None
        self._split_main_grad_descriptor_cache = None
        self.is_first_microbatch = True

    def forward(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        probs = probs.reshape(x.shape[0], -1)
        routing_map = routing_map.reshape(x.shape[0], -1)

        # Compact the authoritative route set directly into MOK's fixed [tokens, K]
        # representation. Missing routes are encoded as expert -1 with zero weight.
        router_weights, top_experts = routing_map_to_mok_inputs(probs, routing_map, self.topk)

        output = _MoKAutograd.apply(
            self,
            x,
            router_weights,
            top_experts,
            *self.autograd_routed_parameters,
            self.shared_fc1_weight,
            self.shared_fc2_weight,
        )
        return output.view(original_shape)
