# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from typing import Callable, Optional

import torch
import torch.nn.functional as F
import transformer_engine as te

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import get_tensor_model_parallel_group

from .mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

try:
    import transformer_engine as te
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch.constants import TE_DType
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear
    HAVE_TE=True 
except ImportError:
    HAVE_TE = False
    

def _te_rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float):
    out , _, _ = tex.rmsnorm_fwd(
            input,
            weight,
            eps,
            None,
            None,
            TE_DType[torch.bfloat16],
            16, # sm-margin
            False, # zero centered gamma
        )
    return out.to(input.dtype)

# def _te_gemm(input: torch.Tensor, weight: torch.Tensor):
#     output_shape = list(input.shape) 
#     output_shape[-1] = weight.size(0)
#     output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
#     tex.general_gemm(
#             weight,
#             input,
#             get_workspace(),
#             out_dtype=input.dtype,
#             quantization_params=None,
#             alpha=1.0,
#             beta=None,
#             accumulate=False,
#             out=output,
#             bias=None,
#             use_split_accumulator=_2X_ACC_FPROP,
#         )

class InferenceLayerNormColumnParallelLinear(TELayerNormColumnParallelLinear):
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
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group
        )

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"
        assert self.tp_size == 1

    @torch.no_grad()
    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        # make x 2D but restore original shape at the end
        x_shape = x.shape
        x = x.view(-1, x.size(-1))
        x = _te_rms_norm(
            input=x, weight=self.layer_norm_weight, eps=self.eps
        )
        if self.tp_size > 1:
            x = gather_from_sequence_parallel_region(
                x, group=self.tp_group, tensor_parallel_output_grad=False
            )
        x = torch.matmul(x, self.weight.t())
        x = x.view(*x_shape[:-1], -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward
        """
        if self.training:
            # Training mode -> fallback to TE
            return super().forward(x)
        else:
            return self._inference_forward(x), None


class InferenceRowParallelLinear(TERowParallelLinear):
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
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group
        )

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"

        assert self.tp_size == 1

    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.weight.t())
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
            return super().forward(x)
        else:
            # Inference mode -> custom fw pass can be implemented here
            return self._inference_forward(x), None
