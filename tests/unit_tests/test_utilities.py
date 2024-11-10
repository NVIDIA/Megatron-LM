import os

import torch.distributed
from megatron.core.device_utils import get_current_device, get_distributed_backend, get_local_device_count
from megatron.core.device_utils import get_distributed_init_method
import torch

from megatron.core.dist_checkpointing.strategies.base import deinit_async_calls, init_async_calls
from megatron.core.dist_checkpointing.strategies.torch import deinit_shard_default_strategies, init_shard_default_strategies
import megatron.core.parallel_state as ps


class TestModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, bias: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)]
        )


class Utils:

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    inited = False

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized():
            print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
            
            init_method = get_distributed_init_method()
            backend = get_distributed_backend()  
 
            torch.distributed.init_process_group(backend=backend, 
                                                 world_size=Utils.world_size, 
                                                 rank=Utils.rank, init_method=init_method)

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = int(os.environ['WORLD_SIZE']) if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ['RANK'])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        if not Utils.inited:
            return
        torch.distributed.barrier()
        ps.destroy_model_parallel()
        deinit_shard_default_strategies()
        deinit_async_calls()
        Utils.inited = False

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        init_shard_default_strategies(process_group=ps.get_default_process_group())
        init_async_calls(process_group=ps.get_default_process_group())
        get_current_device()
        Utils.inited = True