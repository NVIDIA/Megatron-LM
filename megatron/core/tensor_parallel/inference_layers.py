import torch
from typing import Callable, Optional

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear 
from megatron.core.model_parallel_config import ModelParallelConfig


class InferenceLayerNormColumnParallelLinear(TELayerNormColumnParallelLinear):
    def __init__(self,
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
                tp_group: Optional[torch.distributed.ProcessGroup] = None): 
        super().__init__(
            input_size=input_size,
            output_size=output_size,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Training mode -> fallback to TE
            return super().forward(x)
        else:
            # Inference mode -> custom fw pass can be implemented here
            return super().forward(x)

class InferenceRowParallelLinear(TERowParallelLinear):
    def __init__(self,
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
            tp_group: Optional[torch.distributed.ProcessGroup] = None):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Training mode -> fallback to TE
            return super().forward(x)
        else:
            # Inference mode -> custom fw pass can be implemented here
            return super().forward(x)
