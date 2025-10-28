# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from typing import Callable, Optional

import torch

from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import get_global_symmetric_memory_buffer
from megatron.core.inference.communication.torch_symm_triton import multimem_all_gather, multimem_reduce_scatter, multimem_fused_rs_res_rms_ag

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


def _te_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
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
            tp_group=tp_group,
        )

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"
        self.symmetric_memory_comms = True
        # when using fusion this should be False for layer 1,
        # true for other layers.
        self.skip_norm_and_all_gather = False


    @torch.no_grad()
    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        # make x 2D but restore original shape at the end
        if self.tp_size > 1:
            if self.symmetric_memory_comms:
                if not self.skip_norm_and_all_gather:
                    x = _te_rms_norm(x=x, weight=self.layer_norm_weight, eps=self.eps)
                assert x.dtype == torch.bfloat16
                dim_size = list(x.size())
                dim_size[0] *= self.tp_size
                symm_mem_buff, symm_mem_handle = get_global_symmetric_memory_buffer().get_tensor(dim_size, x.dtype)
                assert symm_mem_buff is not None
                if not self.skip_norm_and_all_gather:
                    multimem_all_gather(symm_mem_buff, x, symm_mem_handle)
                x = symm_mem_buff
            else:
                x = _te_rms_norm(x=x, weight=self.layer_norm_weight, eps=self.eps)
                x, _ = gather_along_first_dim(x, process_group=self.tp_group)
        x = torch.matmul(x, self.weight.t())
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
            tp_group=tp_group,
        )

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--use-inference-optimized-layers requires sequence parallelism"

        self.symmetric_memory_comms = True
        self.next_layer_norm_weights = None  # to be set externally if needed
        self.layer_norm_eps = config.layernorm_epsilon

    @torch.no_grad()
    def _inference_forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.tp_size > 1 and self.symmetric_memory_comms:
            # write output of matmul directly onto symmetric memory buffer
            assert x.dtype == torch.bfloat16
            dim_size = list(x.size())
            dim_size[-1] = self.weight.size(0)  
            symm_mem_buff, symm_mem_handle = get_global_symmetric_memory_buffer().get_tensor(dim_size, x.dtype)
            assert symm_mem_buff is not None
            torch.matmul(x, self.weight.t(), out=symm_mem_buff)
            x = symm_mem_buff
        else:
            x = torch.matmul(x, self.weight.t())
        
        is_using_fused_kernel = residual is not None 
        if is_using_fused_kernel:
            assert self.next_layer_norm_weights is not None

        if self.tp_size > 1:
            if self.symmetric_memory_comms: 
                if not is_using_fused_kernel:
                    dim_size = list(x.size()) 
                    dim_size[0] //= self.tp_size 
                    reduce_scatter_output = torch.empty(dim_size, dtype=x.dtype, device=x.device)
                    multimem_reduce_scatter(reduce_scatter_output, x, symm_mem_handle)
                    x = reduce_scatter_output
                else:
                    multimem_fused_rs_res_rms_ag(residual,
                                                 x,
                                                 symm_mem_handle,
                                                 residual,
                                                 self.next_layer_norm_weights,
                                                 self.layer_norm_eps)
                    return residual
            else:
                raise NotImplementedError
        return x

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward
        """
        if self.training:
            # Training mode -> fallback to TE
            assert residual is None
            return super().forward(x)
        else:
            # Inference mode -> custom fw pass can be implemented here
            return self._inference_forward(x, residual), None


#ColumnLayerNormLinear
