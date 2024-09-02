

import os
from megatron.core import mpu
from megatron.core.device_utils import get_current_device, get_xla_model
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_groups, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from tests.unit_tests.test_utilities import Utils
import torch

xm = get_xla_model()

def test_all_reduce():
    Utils.initialize_model_parallel(4,2)
    assert torch.distributed.is_initialized()
 
    inputs = torch.ones(
            (2,2),
            dtype=torch.float,
            device=get_current_device(),
            requires_grad=False,
        )
    
    rank = int(os.environ['RANK'])
    
    xm = get_xla_model()
    if xm:
        tp_list = list(xm.all_gather(inputs, groups=get_tensor_model_parallel_groups()).split(inputs.size()[0]))
    else:
        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())

    print(f"after all_gather tp_list: {tp_list}")
    torch.distributed.barrier()
    Utils.destroy_model_parallel()
    print("SUCCESS!")

if __name__ == "__main__":
    test_all_reduce()