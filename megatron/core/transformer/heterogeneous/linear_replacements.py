# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch.nn.functional as F
from torch import Tensor

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

try:
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _gather_from_tensor_parallel_region(x: Tensor, config: TransformerConfig) -> Tensor:
    if get_tensor_model_parallel_world_size() > 1:
        if config.sequence_parallel:
            # pad hidden dimension (last dimension) with zeros such that the valid data is placed in
            # indices [tp_rank * hidden/tp_size, (tp_rank+1) * hidden/tp_size),
            # and zeros fill the other parts.
            output_size = config.hidden_size
            output_size_per_partition = divide(output_size, get_tensor_model_parallel_world_size())

            pad_before = get_tensor_model_parallel_rank() * output_size_per_partition
            pad_after = output_size - pad_before - output_size_per_partition

            pad_shape = [0] * (x.ndim - 1) * 2 + [pad_before, pad_after]
            x = F.pad(x, pad_shape, "constant", 0)

            x = reduce_scatter_to_sequence_parallel_region(x)
        else:
            x = gather_from_tensor_model_parallel_region(x)

    return x


if HAVE_TE:

    class TELayerNormColumnParallelLinearGathered(TELayerNormColumnParallelLinear):
        """
        A linear replacement for TE Attention/MLP blocks.
        Supports gathering TP outputs when sequence parallel is enabled.
        """

        def __init__(self, config: TransformerConfig, tp_comm_buffer_name: str, *args, **kwargs):
            super().__init__(
                input_size=config.hidden_size,
                output_size=config.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

        def forward(self, x, **kwargs):
            out, bias = super().forward(x)
            assert bias is None, "bias should be None since we set skip_bias_add=False"

            out = _gather_from_tensor_parallel_region(out, self.config)

            return out, bias


class ColumnParallelLinearGathered(ColumnParallelLinear):
    """
    A linear replacement for local implementation of Attention/MLP blocks.
    Supports gathering TP outputs when sequence parallel is enabled.
    """

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

    def forward(
        self,
        input_: Tensor,
        weight: Tensor | None = None,
        runtime_gather_output: bool | None = None,
        **kwargs,
    ):
        out, bias = super().forward(input_, weight, runtime_gather_output)
        assert bias is None, "bias should be None since we set skip_bias_add=False"

        if runtime_gather_output or self.gather_output:
            raise ValueError("gathering TP outputs is not supported for linear replacement")

        out = _gather_from_tensor_parallel_region(out, self.config)

        return out, bias
