import os
import torch
import megatron.core.parallel_state as ps

class Utils:

    world_size = torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
            torch.cuda.set_device(Utils.rank)
            torch.distributed.init_process_group( world_size=Utils.world_size, rank=Utils.rank)
            torch.distributed.barrier()

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if torch.distributed.is_initialized() and Utils.world_size != torch.distributed.get_world_size():
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ['LOCAL_RANK'])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1, virtual_pipeline_model_parallel_size = None, pipeline_model_parallel_split_rank = None, **kwargs):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank, **kwargs)