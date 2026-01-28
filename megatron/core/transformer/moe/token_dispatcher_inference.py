# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized AlltoAll Token Dispatcher with GPU-resident metadata.

This implementation keeps tokens_per_expert GPU-resident to enable use of
torch._grouped_mm without host synchronization.
"""

import torch
from typing import List, Optional, Tuple

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs
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
        """Move splits to CPU for AlltoAll, but keep tokens_per_expert on GPU.

        The parent class moves all tensors to CPU including tokens_per_expert.
        For inference with torch._grouped_mm, we need tokens_per_expert to stay
        on GPU to avoid host synchronization.

        This override:
        - Still moves input_splits, output_splits, etc. to CPU (required by AlltoAll)
        - Still does stream synchronization
        - But keeps tokens_per_expert on GPU (for torch._grouped_mm)
        """
        from megatron.core.transformer.moe.token_dispatcher import maybe_move_tensor_to_cpu

        if not self.drop_and_pad:
            if point == self.cuda_dtoh_point:
                # Move splits to CPU (required by torch.distributed.all_to_all_single)
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    # Move AlltoAll splits to CPU (required)
                    self.input_splits = maybe_move_tensor_to_cpu(
                        self.input_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits = maybe_move_tensor_to_cpu(
                        self.output_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits_tp = maybe_move_tensor_to_cpu(
                        self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                    )
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )
                    # NOTE: We intentionally do NOT move tokens_per_expert to CPU here.
                    # It stays on GPU for use with torch._grouped_mm.
                self.d2h_event = self.cuda_dtoh_stream.record_event()

            if point == self.cuda_sync_point:
                # Synchronize with the DtoH stream
                self.d2h_event.synchronize()

        # Return tokens_per_expert unchanged (stays on GPU!)
        return tokens_per_expert

