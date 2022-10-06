import torch
import megatron.core.tensor_parallel.utils as util

def test_split_tensor_along_last_dim():
    input_tensor = torch.rand((3,4))
    torch.equal(input_tensor[0:2,0:2], util.split_tensor_along_last_dim(input_tensor,2)[0])
    torch.equal(input_tensor[2:,2:], util.split_tensor_along_last_dim(input_tensor,2)[1])
