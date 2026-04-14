# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

from megatron.core.transformer.moe.router_replay import RouterReplay


class RoutingMetadata:
    """Manages routing indices metadata for MoE layers during inference.

    This class provides static buffers for CUDA graph compatibility when
    recording routing decisions. It holds a reference to the inference context
    to automatically determine whether to use static buffers based on CUDA graph state.

    Args:
        context (DynamicInferenceContext): The inference context.
        moe_router_topk (int): Number of experts selected per token.
    """

    def __init__(self, context: 'DynamicInferenceContext', moe_router_topk: int):
        self.context = context
        self.max_tokens = context.max_tokens
        self.moe_router_topk = moe_router_topk
        self.device = torch.cuda.current_device()

        # Static buffer allocated lazily in _ensure_buffer_allocated().
        # We defer allocation because RouterReplay instances don't exist yet at init time.
        self.routing_indices_buffer: Optional[torch.Tensor] = None
        self.num_moe_layers: Optional[int] = None

    def _ensure_buffer_allocated(self) -> None:
        """Allocate the static buffer if not already allocated.

        Gets the actual number of MoE layers from RouterReplay instances.
        """
        if self.routing_indices_buffer is not None:
            return

        self.num_moe_layers = len(RouterReplay.global_router_replay_instances)

        if self.num_moe_layers == 0:
            return

        # Static buffer for CUDA graph compatibility.
        # Shape: [max_tokens, num_moe_layers, moe_router_topk]
        self.routing_indices_buffer = torch.empty(
            (self.max_tokens, self.num_moe_layers, self.moe_router_topk),
            dtype=torch.int32,
            device=self.device,
        )

    def get_routing_indices(self) -> Optional[torch.Tensor]:
        """Get the recorded routing indices.

        Automatically uses the static buffer when CUDA graphs are active,
        otherwise retrieves from RouterReplay utility.

        Returns:
            Tensor of shape [num_tokens, num_moe_layers, topk] or None if not available.
        """
        if self.context.using_cuda_graph_this_step():
            # Return view of static buffer up to current token count.
            if self.routing_indices_buffer is None:
                return None
            # Only return up to active token count, to skip entries
            # for padding tokens.
            return self.routing_indices_buffer[: self.context.active_token_count]
        else:
            # Get from RouterReplay and stack into [num_tokens, num_layers, topk].
            recorded_data = RouterReplay.get_recorded_data()
            if recorded_data is None or len(recorded_data) == 0:
                return None
            if recorded_data[0] is None:
                return None
            # Stack: list of [num_tokens, topk] -> [num_tokens, num_layers, topk]
            return torch.stack(recorded_data, dim=1)

    def enable_static_buffer_recording(self) -> None:
        """Enable recording into the static buffer for CUDA graph compatibility.

        This sets up RouterReplay instances to copy routing indices into our
        pre-allocated static buffer instead of creating new tensors.
        Allocates the buffer lazily on first call.
        """
        self._ensure_buffer_allocated()
        if self.routing_indices_buffer is not None:
            RouterReplay.set_global_static_buffers(self.routing_indices_buffer)

    def disable_static_buffer_recording(self) -> None:
        """Disable static buffer recording, reverting to normal tensor assignment."""
        RouterReplay.clear_global_static_buffers()
