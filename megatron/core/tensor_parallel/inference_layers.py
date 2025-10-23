# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)

import transformer_engine as te

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig

from .mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

_compiled_rms_norm = torch.compile(F.rms_norm)


class InferenceLayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Inference optimized version of TELayerNormColumnParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            in_features=input_size,
            out_features=output_size,
            init_method=init_method,
            eps=config.layernorm_epsilon,
            normalization="RMSNorm"
        )

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"

    @torch.no_grad()
    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _compiled_rms_norm(
            input=x, normalized_shape=(x.size(-1),), weight=self.layer_norm_weight, eps=self.eps
        )
        if self.tp_size > 1:
            x = gather_from_sequence_parallel_region(
                x, group=self.tp_group, tensor_parallel_output_grad=False
            )
        x = F.linear(input=x, weight=self.weight)
        return x, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward
        """
        if self.training:
            # Training mode -> fallback to TE
            return super().forward(x), None
        else:
            return super().forward(x), None


class InferenceRowParallelLinear(te.pytorch.Linear):
    """
    Inference optimized version of TERowParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            in_features=input_size,
            out_features=output_size,
            init_method=init_method,
            bias=bias
        )

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"

    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.tp_size > 1:
            x = reduce_scatter_to_sequence_parallel_region(x, group=self.tp_group)
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward
        """
        if self.training:
            # Training mode -> fallback to TE
            return super().forward(x), None
        else:
            # Inference mode -> custom fw pass can be implemented here
            return super().forward(x), None 
