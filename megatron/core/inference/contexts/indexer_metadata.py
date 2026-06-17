# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

from megatron.core.transformer.experimental_attention_variant.indexer_replay import IndexerReplay


class IndexerMetadata:
    """Manages indexer topk indices metadata for DSA attention layers during inference.

    This class provides static buffers for CUDA graph compatibility when
    recording indexer top-k decisions. It holds a reference to the inference context
    to automatically determine whether to use static buffers based on CUDA graph state.

    Args:
        context (DynamicInferenceContext): The inference context.
        dsa_indexer_topk (int): Number of top-k tokens selected per query by the DSA indexer.
    """

    def __init__(self, context: "DynamicInferenceContext", dsa_indexer_topk: int):
        self.context = context
        self.max_tokens = context.max_tokens
        self.dsa_indexer_topk = dsa_indexer_topk
        self.device = torch.cuda.current_device()

        self.indexer_indices_buffer: Optional[torch.Tensor] = None
        self.num_dsa_layers: Optional[int] = None

    def _ensure_buffer_allocated(self) -> None:
        if self.indexer_indices_buffer is not None:
            return

        self.num_dsa_layers = len(IndexerReplay.global_indexer_replay_instances)

        if self.num_dsa_layers == 0:
            return

        self.indexer_indices_buffer = torch.empty(
            (self.max_tokens, self.num_dsa_layers, self.dsa_indexer_topk),
            dtype=torch.int32,
            device=self.device,
        )

    def get_indexer_indices(self) -> Optional[torch.Tensor]:
        if self.context.using_cuda_graph_this_step():
            if self.indexer_indices_buffer is None:
                return None
            return self.indexer_indices_buffer[: self.context.active_token_count]
        else:
            recorded_data = IndexerReplay.get_recorded_data()
            if recorded_data is None or len(recorded_data) == 0:
                return None
            if recorded_data[0] is None:
                return None
            return torch.stack(recorded_data, dim=1)

    def enable_static_buffer_recording(self) -> None:
        self._ensure_buffer_allocated()
        if self.indexer_indices_buffer is not None:
            IndexerReplay.set_global_static_buffers(self.indexer_indices_buffer)

    def disable_static_buffer_recording(self) -> None:
        IndexerReplay.clear_global_static_buffers()
