# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


# Trying to apply @jit_fuser / @torch.compile to XIELU class causes issues with sharded_state_dict naming
@jit_fuser
def compiled_xielu(x, alpha_p, alpha_n, beta, eps):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x)


@jit_fuser
def compiled_xiprelu(x, alpha_p, alpha_n, beta):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * x * x + beta * x)


@jit_fuser
def compiled_xiprelup(x, alpha_p, alpha_n, power_p, power_n, beta, eps):
    return torch.where(x > 0,
                      alpha_p * torch.pow(torch.max(x, eps), power_p) + beta * x,
                      alpha_n * torch.pow(torch.abs(torch.min(x, -eps)), power_n) + beta * x)


class XIELU(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super(XIELU, self).__init__(config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
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
        return compiled_xielu(x, alpha_p, alpha_n, self.beta, self.eps)


class XIPReLU(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5):
        super(XIPReLU, self).__init__(config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
        self.beta = beta

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
        alpha_n = F.softplus(self.alpha_n)
        return compiled_xiprelu(x, alpha_p, alpha_n, self.beta)


class XIPReLUP(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, power_p_init=2, power_n_init=2, beta=0.5, eps=1e-6):
        super(XIPReLU, self).__init__(config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
        self.power_p = nn.Parameter(torch.log(torch.exp(torch.tensor(power_p_init - 1.0, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
        self.power_n = nn.Parameter(torch.log(torch.exp(torch.tensor(power_n_init - 1.0, dtype=torch.bfloat16, device='cuda')) - 1.0).unsqueeze(0))
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
            ),
             f'{prefix}power_p': ShardedTensor(
                key=f'{prefix}power_p',
                data=self.power_p,
                global_shape=(num_layers,),
                global_offset=(layer_idx,),
                local_shape=(1,),
                axis_fragmentations=(num_layers,),
                replica_id=(tp_rank, pp_rank, dp_rank),
                dtype=self.power_p.dtype,
            ),
            f'{prefix}power_n': ShardedTensor(
                key=f'{prefix}power_n',
                data=self.power_n,
                global_shape=(num_layers,),
                global_offset=(layer_idx,),
                local_shape=(1,),
                axis_fragmentations=(num_layers,),
                replica_id=(tp_rank, pp_rank, dp_rank),
                dtype=self.power_n.dtype,
            )
        }

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = F.softplus(self.alpha_n)
        power_p = 1 + F.softplus(self.power_p)
        power_n = 1 + F.softplus(self.power_n)
        return compiled_xiprelup(x, alpha_p, alpha_n, power_p, power_n, self.beta, self.eps)


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
