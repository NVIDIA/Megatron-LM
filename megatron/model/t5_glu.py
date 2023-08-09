import torch

from megatron.core import tensor_parallel
from megatron.model.module import MegatronModule


class T5GLU(MegatronModule):
    def __init__(
            self,
            in_features,
            out_features,
            *,
            config,
            init_method,
            activation_fn=torch.sigmoid,
            bias=False,
            gather_output=False,
    ):
        super().__init__()
        self.linear = tensor_parallel.ColumnParallelLinear(
            in_features,
            out_features,
            config=config,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )
        self.nonlinear = tensor_parallel.ColumnParallelLinear(
            in_features,
            out_features,
            config=config,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )
        self.activation_fn = activation_fn

    def forward(self, x):
        output = self.linear(x)[0] * self.activation_fn(self.nonlinear(x)[0])
        return output, None


class T5SwiGLU(T5GLU):
    def __init__(
            self,
            in_features,
            out_features,
            *,
            config,
            init_method,
            bias=False,
            gather_output=False,
    ):
        super().__init__(
            in_features,
            out_features,
            config=config,
            init_method=init_method,
            activation_fn=torch.nn.functional.silu,
            bias=bias,
            gather_output=gather_output,
        )
