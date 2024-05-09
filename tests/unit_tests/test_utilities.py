import os
import torch
import megatron.core.parallel_state as ps

class Utils:

    world_size = torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized():
            print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
            torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '6000')
            init_method += master_ip + ':' + master_port
            torch.distributed.init_process_group(backend='nccl', world_size=Utils.world_size, rank=Utils.rank, init_method=init_method)
        
    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1, virtual_pipeline_model_parallel_size = None, pipeline_model_parallel_split_rank = None):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank)