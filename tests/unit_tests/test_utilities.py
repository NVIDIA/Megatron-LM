import os
from megatron.core.device_utils import get_distributed_backend, get_local_device_count
from megatron.core.device_utils import get_distributed_init_method
import torch
import megatron.core.parallel_state as ps


class TestModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, bias: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)]
        )


class Utils:

    world_size = get_local_device_count()
    rank = int(os.environ['LOCAL_RANK'])
    inited = False

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized() and Utils.rank >= 0:
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
        Utils.world_size = get_local_device_count() if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ['LOCAL_RANK'])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        if not Utils.inited:
            return
        ps.destroy_model_parallel()
        torch.distributed.barrier()
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
        Utils.inited = True
