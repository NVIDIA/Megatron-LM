# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
class XIELU(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super(XIELU, self).__init__(config)

        self.config = config

        self.alpha_p = nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0)
        )
        self.alpha_n = nn.Parameter(
            torch.log(
                torch.exp(torch.tensor(alpha_n_init - beta, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0)
        )
        self.beta = beta
        self.eps = torch.tensor(eps, dtype=torch.bfloat16, device='cuda')

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        num_layers = self.config.num_layers

        layer_idx = 0
        if 'layers.' in prefix:
            layer_idx = int(prefix.split('layers.')[1].split('.')[0])

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()

        return {
            f'{prefix}alpha_p': ShardedTensor(
                key=f'{prefix}alpha_p',
                data=self.alpha_p,
                global_shape=(num_layers,),
                global_offset=(layer_idx,),
                local_shape=(1,),
                axis_fragmentations=(num_layers,),
                replica_id=(tp_rank, pp_rank, dp_rank),
                dtype=self.alpha_p.dtype,
            ),
            f'{prefix}alpha_n': ShardedTensor(
                key=f'{prefix}alpha_n',
                data=self.alpha_n,
                global_shape=(num_layers,),
                global_offset=(layer_idx,),
                local_shape=(1,),
                axis_fragmentations=(num_layers,),
                replica_id=(tp_rank, pp_rank, dp_rank),
                dtype=self.alpha_n.dtype,
            )
        }

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return torch.where(x > 0,
                           alpha_p * x * x + self.beta * x,
                           alpha_n * torch.expm1(torch.min(x, self.eps)) - alpha_n * x + self.beta * x)


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
