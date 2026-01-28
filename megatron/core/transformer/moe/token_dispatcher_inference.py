# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized AlltoAll Token Dispatcher with GPU-resident metadata.

This implementation keeps tokens_per_expert GPU-resident to enable use of
torch._grouped_mm without host synchronization.
"""

import torch
from typing import List, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


class InferenceAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """
    Inference-optimized AlltoAll token dispatcher.

    Key optimization: Returns tokens_per_expert as a GPU tensor (not moved to CPU)
    to enable torch._grouped_mm without host synchronization.

    Assumes tp_size == 1 (no tensor parallelism within experts).
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """
        Initialize the inference AlltoAll token dispatcher.

        Args are identical to MoEAlltoAllTokenDispatcher for compatibility.
        """
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Preprocess routing map, ensuring tokens_per_expert is created on GPU.

        For drop_and_pad mode, the parent creates tokens_per_expert on CPU via
        torch.full() without a device argument. We override to create it directly
        on GPU to avoid any host synchronization.

        For non-drop_and_pad mode, the parent creates it on GPU via routing_map.sum(),
        so we just call the parent.
        """
        if self.drop_and_pad:
            # Replicate parent's drop_and_pad logic but create tensor on GPU
            from megatron.core.transformer.moe.moe_utils import get_capacity

            num_tokens = routing_map.size(0) * self.config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts

            # Create on GPU (parent creates on CPU)
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
                device=routing_map.device,  # Same device as input (GPU)
            )

            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.tp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            return num_tokens_per_local_expert
        else:
            # Non-drop_and_pad: parent creates on GPU via routing_map.sum()
            return super().preprocess(routing_map)

    def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """No-op for single GPU inference - all metadata stays on GPU.

        For single GPU (ep_size=1, tp_size=1):
        - input_splits, output_splits, output_splits_tp are all None (no AlltoAll needed)
        - tokens_per_expert stays on GPU for torch._grouped_mm
        - No DtoH transfers or synchronization required

        This enables fully CUDA-graphable MoE forward pass.
        """
        # Validate single GPU assumptions
        assert self.ep_size == 1, (
            f"InferenceAlltoAllTokenDispatcher requires ep_size=1, got {self.ep_size}"
        )
        assert self.tp_size == 1, (
            f"InferenceAlltoAllTokenDispatcher requires tp_size=1, got {self.tp_size}"
        )
        assert self.input_splits is None, (
            "input_splits should be None for single GPU inference"
        )
        assert self.output_splits is None, (
            "output_splits should be None for single GPU inference"
        )
        assert self.output_splits_tp is None, (
            "output_splits_tp should be None for single GPU inference"
        )
        assert not isinstance(self.num_out_tokens, torch.Tensor), (
            "num_out_tokens should be a Python int for dropless single GPU inference, "
            f"got {type(self.num_out_tokens)}. Ensure moe_expert_capacity_factor is None "
            "and moe_router_padding_for_quantization is False."
        )
        assert tokens_per_expert.is_cuda, (
            "tokens_per_expert should be on GPU for single GPU inference"
        )


        # No DtoH transfers needed - return tokens_per_expert unchanged (stays on GPU!)
        return tokens_per_expert

