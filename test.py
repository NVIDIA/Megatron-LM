import os
import torch

def main():
    rank = torch.cuda.current_device()
    world_size = torch.cuda.device_count()
    print(f'Initializing torch.distributed with rank: {rank}, world_size: {world_size}')
    torch.cuda.set_device(rank % torch.cuda.device_count())
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=init_method)

if __name__ == '__main__':
    main()
