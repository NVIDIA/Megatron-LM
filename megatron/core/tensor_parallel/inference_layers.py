from typing import Callable, Optional

import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig

from .mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

_compiled_rms_norm = torch.compile(F.rms_norm)


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

        assert config.normalization == "RMSNorm"
        assert not config.layernorm_zero_centered_gamma
        assert not bias

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

        if self.tp_size > 1:
            assert config.sequence_parallel

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
            return super().forward(x)
        else:
            return self._inference_forward(x)


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

        assert not bias

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
            tp_group=tp_group,
        )

        if self.tp_size > 1:
            assert config.sequence_parallel

    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(input=x, weight=self.weight)
        if self.tp_size > 1:
            x = reduce_scatter_to_sequence_parallel_region(x, group=self.tp_group)
        return x, None

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
            return self._inference_forward(x)
