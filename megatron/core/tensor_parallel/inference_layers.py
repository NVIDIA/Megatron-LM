# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from typing import Callable, Optional

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none

try:
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.distributed import (
        gather_along_first_dim,
        reduce_scatter_along_first_dim,
    )
    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _te_rms_norm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    out, _, _ = tex.rmsnorm_fwd(
        x,
        weight,
        eps,
        None,
        None,
        TE_DType[torch.bfloat16],
        16,  # sm-margin
        False,  # zero centered gamma
    )
    out = out.view(*x_shape[:-1], -1)
    return out.to(x.dtype)


class InferenceLayerNormColumnParallelLinear(torch.nn.Module):
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
        assert HAVE_TE, "--use-inference-optimized-layers requires transformer engine"
        super().__init__()
        self.tp_group = get_tensor_model_parallel_group_if_none(
            tp_group, is_expert=is_expert
        )
        self.tp_size = dist.get_world_size(self.tp_group)
        assert output_size % self.tp_size == 0, (
            f"output_size ({output_size}) must be divisible by tp_size ({self.tp_size})"
        )
        
        # Parameter names "weight"  and "layer_norm_weight" are kept the 
        # same as in TELayerNormColumnParallelLinear for compatibility with
        # loading pretrained checkpoints.
        
        self.weight = torch.nn.Parameter(
            torch.empty(output_size // self.tp_size, input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype)
        )
        self.layer_norm_weight = torch.nn.Parameter(
            torch.empty(input_size, 
                        device=torch.cuda.current_device(), 
                        dtype=config.params_dtype)
        )
        self.eps = config.layernorm_epsilon

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _te_rms_norm_kernel(x=x, weight=self.layer_norm_weight, eps=self.eps)
        if self.tp_size > 1:
            x, _ = gather_along_first_dim(x, process_group=self.tp_group)
        x = torch.matmul(x, self.weight.t())
        return x, None


class InferenceRowParallelLinear(torch.nn.Module):
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
        assert HAVE_TE, "--use-inference-optimized-layers requires transformer engine"
        super().__init__()
        self.tp_group = get_tensor_model_parallel_group_if_none(
            tp_group, is_expert=is_expert
        )
        self.tp_size = dist.get_world_size(self.tp_group)
        assert input_size % self.tp_size == 0, (
            f"input_size ({input_size}) must be divisible by tp_size ({self.tp_size})"
        )
        
        # Parameter name "weight" is kept the 
        # same as in TERowParallelLinear for compatibility with
        # loading pretrained checkpoints.

        self.weight = torch.nn.Parameter(
            torch.empty(output_size, input_size // self.tp_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype)
        )
 
        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.weight.t())
        if self.tp_size > 1:
            x, _ = reduce_scatter_along_first_dim(x, tp_group=self.tp_group)
        return x, None
