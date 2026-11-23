# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/transformer_layer.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit rotary inputs for Qwen vision-language CUDA graphs."""

from typing import Any

import torch

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


class Qwen3VLRotaryTransformerLayer(TransformerLayer):
    """Declare Qwen's batch-dependent rotary input before any forward runs.

    Use separate language and vision subclasses so TE's class-level capture
    wrappers do not propagate between independently captured stacks.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        *,
        rotary_dim: int,
        **kwargs: Any,
    ) -> None:
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("Qwen rotary_dim must be a positive even number.")
        super().__init__(config=config, submodules=submodules, **kwargs)
        self.rotary_dim = rotary_dim

    def get_layer_static_inputs(
        self, seq_length: int, micro_batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Reserve an FP32 rotary input that TE updates on every replay."""
        inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if (
            self.config.cuda_graph_modules
            and CudaGraphModule.attn not in self.config.cuda_graph_modules
        ):
            return inputs

        if getattr(self.config, "sequence_packing_scheduler", None) is not None:
            raise ValueError("Qwen rotary CUDA graph inputs require a fixed sequence layout.")
        cp_size = self.config.context_parallel_size
        if seq_length % cp_size:
            raise ValueError(
                "Qwen rotary CUDA graph sequence length must be divisible by context parallel size."
            )

        # Attention gathers sequence-parallel tokens before applying RoPE, so
        # its sequence axis is partitioned by CP but not by TP.
        inputs["rotary_pos_emb"] = torch.zeros(
            (seq_length // cp_size, micro_batch_size, 1, self.rotary_dim),
            dtype=torch.float32,
            device=inputs["hidden_states"].device,
        )
        return inputs
