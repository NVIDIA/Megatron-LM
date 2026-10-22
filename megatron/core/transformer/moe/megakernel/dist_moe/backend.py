# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dist-MoE routed experts with native MCore parameters and DDP ownership."""

from __future__ import annotations

from types import MethodType

import torch

from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend
from megatron.core.transformer.moe.megakernel.parameter_bridge import main_grad_buffer
from megatron.core.transformer.moe.megakernel.route_adapter import routing_map_to_indices

from .runtime import DistMoEBackendOptions, _DistMoEAutograd, storage_signature, validate_config
from .weights import prepare_weight


class DistMoEMegakernel(MegakernelBackend):
    """Replace routed compute; routing and optional shared experts remain native."""

    supports_cuda_graphs = True

    def __init__(self, config, ep_group, routed_experts, shared_experts, num_local_experts):
        super().__init__()
        del shared_experts
        validate_config(config)
        self.config = config
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size
        self.topk = config.moe_router_topk
        self.use_mxfp8 = config.fp8 is not None
        self.options = DistMoEBackendOptions(**(config.moe_megakernel_backend_config or {}))
        shapes = (
            (num_local_experts, 2 * self.intermediate_size, self.hidden_size),
            (num_local_experts, self.hidden_size, self.intermediate_size),
        )
        for name, linear, shape in zip(
            ("routed_fc1_weight", "routed_fc2_weight"),
            (routed_experts.linear_fc1, routed_experts.linear_fc2),
            shapes,
        ):
            if not getattr(linear, "single_grouped_weight", False):
                raise ValueError("dist_moe requires actual single-grouped expert parameters")
            weight = linear.weight
            if tuple(weight.shape) != shape or not weight.requires_grad or not weight.is_leaf:
                raise ValueError(f"dist_moe requires trainable native weight with shape {shape}")
            # Aliases trigger the executing module's DDP parameter-gather pre-hook.
            self.register_parameter(name, weight)
        if self.owners[0] is self.owners[1]:
            raise ValueError("dist_moe does not support tied expert weights")
        self.context = None
        self._live = False
        self._gradient_binding = None
        self._weight_cache = None
        self._weight_binding = None
        self._prepared = None
        self._graph_bound = False
        self._graph_binding = None
        self.is_first_microbatch = True
        routed_experts.register_load_state_dict_pre_hook(self._before_expert_load)

    @property
    def owners(self):
        """Return canonical parameters in Dist W13/W2 order."""
        return self.routed_fc1_weight, self.routed_fc2_weight

    def _grads(self):
        grads = tuple(main_grad_buffer(param) for param in self.owners)
        if tuple(storage_signature(grad) for grad in grads) != self._gradient_binding:
            raise RuntimeError("dist_moe main_grad storage changed; close and reinitialize runtime")
        return grads

    def _weight_signature(self):
        def binding(tensor):
            return None if tensor is None else (tensor._version, storage_signature(tensor))

        return tuple(
            (
                id(param),
                param._version,
                binding(getattr(param, "rowwise_data", param)),
                *(
                    binding(getattr(param, name, None))
                    for name in ("columnwise_data", "scale_inv", "columnwise_scale_inv")
                ),
            )
            for param in self.owners
        )

    def _check_backward_bindings(self):
        if self._weight_signature() != self._weight_binding:
            raise RuntimeError("dist_moe parameter storage/version changed during backward")
        return self._grads()

    def _prepare_weights(self, first_microbatch=False):
        binding = self._weight_signature()
        if self._prepared is None or first_microbatch or binding != self._weight_binding:
            cached = (None, None) if self._weight_cache is None else self._weight_cache
            prepared = [
                prepare_weight(param, use_mxfp8=self.use_mxfp8, cached=cache)
                for param, cache in zip(self.owners, cached)
            ]
            self._prepared = tuple(item[0] for item in prepared)
            self._weight_cache = tuple(item[1] for item in prepared)
            self._weight_binding = binding
        self.is_first_microbatch = False

    def _graph_storage_binding(self):
        operands = []
        for weight in self._prepared:
            if self.use_mxfp8:
                operands.extend(
                    (weight.fprop_data, weight.fprop_scale, weight.dgrad_data, weight.dgrad_scale)
                )
            else:
                operands.append(weight)
        return (
            tuple(id(owner) for owner in self.owners),
            tuple(storage_signature(tensor) for tensor in (*operands, *self._grads())),
        )

    def cuda_graph_capture_prepare(self):
        """Materialize stable operands before the local runner begins capture."""
        if self.context is None or self._live:
            raise RuntimeError("dist_moe capture requires an initialized, idle runtime")
        self._prepare_weights(first_microbatch=True)
        binding = self._graph_storage_binding()
        if self._graph_bound and binding != self._graph_binding:
            raise RuntimeError("dist_moe CUDA graph storage changed; delete and recapture graphs")
        self._graph_binding = binding
        self._graph_bound = True

    def cuda_graph_prepare_replay(self, is_first_microbatch, grad_enabled):
        """Refresh scales after native DDP gathering, outside the captured graph."""
        if not self._graph_bound or self.context is None or self._live:
            raise RuntimeError("dist_moe replay requires an idle captured runtime")
        self._prepare_weights(first_microbatch=is_first_microbatch)
        if self._graph_storage_binding() != self._graph_binding:
            raise RuntimeError("dist_moe CUDA graph storage changed; delete and recapture graphs")
        self._live = grad_enabled

    def cuda_graph_prepare_backward(self):
        """Reject stale graph operands before the GPU can read or write them."""
        if not self._live:
            raise RuntimeError("dist_moe backward replay requires a live forward")
        self._check_backward_bindings()

    def cuda_graph_finish_replay(self):
        """Release this frame; the local runner completes native DDP owner edges."""
        self._live = False

    def cuda_graph_release(self):
        """Release borrowed graph storage before runtime or model teardown."""
        if not self._graph_bound:
            return
        if self._live:
            raise RuntimeError("cannot delete dist_moe graphs with an unfinished backward")
        torch.cuda.synchronize(self.owners[0].device)
        self._graph_bound = False
        self._graph_binding = None

    def forward(self, hidden_states, probs, routing_map):
        """Execute with borrowed payloads and direct optimizer-visible accumulation."""
        if self.context is None:
            raise RuntimeError("call initialize_dist_moe_runtime after wrapping the model in DDP")
        if self._live:
            raise RuntimeError("dist_moe requires each microbatch backward before its next forward")
        capturing = torch.cuda.is_current_stream_capturing()
        if torch.cuda.current_stream(hidden_states.device).cuda_stream != self._stream and not (
            capturing and self._graph_bound
        ):
            raise RuntimeError("dist_moe must execute on the stream used to initialize its runtime")
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1]).contiguous()
        if not 1 <= x.shape[0] <= self.context.config.max_num_local_input_tokens:
            raise ValueError("dist_moe input exceeds its initialized physical-token bound")
        scores, ids = routing_map_to_indices(
            probs.reshape(x.shape[0], -1), routing_map.reshape(x.shape[0], -1), self.topk
        )
        if not capturing:
            self._prepare_weights(first_microbatch=self.is_first_microbatch)
        elif not self._graph_bound or self._graph_storage_binding() != self._graph_binding:
            raise RuntimeError("dist_moe capture requires stable, prepared graph operands")
        self._grads()
        self._live = torch.is_grad_enabled()
        output = _DistMoEAutograd.apply(self, x, ids, scores, *self.owners)
        return output.view(shape)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Native expert modules own all checkpoint entries."""
        return {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        pass

    def _invalidate_weights(self):
        if self._graph_bound:
            raise RuntimeError("delete_cuda_graphs before invalidating dist_moe weights")
        self._weight_cache = self._prepared = self._weight_binding = None
        self.is_first_microbatch = True

    def _before_expert_load(self, module, state_dict, prefix, local_metadata, *args):
        # Guard the native owner before its FC1/FC2 children can copy or replace
        # parameters. In-place loads retain DDP buffers and the idle context.
        if self._live or self._graph_bound:
            if self._graph_bound:
                raise RuntimeError("delete_cuda_graphs before loading dist_moe weights")
            raise RuntimeError("cannot load dist_moe weights with an unfinished backward")
        if local_metadata.get("assign_to_params_buffers", False):
            raise RuntimeError("dist_moe requires load_state_dict(assign=False) to preserve owners")
        self._invalidate_weights()

    def _load_from_state_dict(self, *args, **kwargs):
        if self._live:
            raise RuntimeError("cannot load dist_moe weights with an unfinished backward")
        self._invalidate_weights()

    def _apply(self, *args, **kwargs):
        if self.context is not None:
            raise RuntimeError("close dist_moe runtime before moving parameter storage")
        # TE wrapper swap_tensors can discard DDP metadata on a second alias visit.
        snapshots = {
            name: (param, dict(param.__dict__))
            for name, param in self._parameters.items()
            if param is not None
        }
        result = super()._apply(*args, **kwargs)
        for name, (old, attrs) in snapshots.items():
            param = self._parameters[name]
            for key, value in attrs.items():
                if key not in param.__dict__:
                    if isinstance(value, MethodType) and value.__self__ is old:
                        value = MethodType(value.__func__, param)
                    setattr(param, key, value)
        self._invalidate_weights()
        return result
