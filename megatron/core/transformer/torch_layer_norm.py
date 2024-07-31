# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings

import torch

from megatron.core.transformer import TransformerConfig


class WrappedTorchLayerNorm(torch.nn.LayerNorm):

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,  ## TODO: unused arguments. See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        self.config = config
        assert (
            not self.config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch LayerNorm"

        assert (
            self.config.normalization == "LayerNorm"
        ), f'({self.config.normalization}) is not supported in by torch Layernorm'

        assert (
            not self.config.persist_layer_norm
        ), f"persist_layer_norm not supported by torch LayerNorm"

        assert (
            not self.config.sequence_parallel
        ), f"sequence parallel not supported by torch LayerNorm"

        assert (
            not self.config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        super().__init__(
            normalized_shape=hidden_size,  ## applied to last len(normalized_shape.size) dimensions
            eps=eps,
        )
