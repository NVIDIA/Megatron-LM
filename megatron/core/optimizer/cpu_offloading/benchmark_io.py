# This file tries to benchmark non-blocking copy/to ops on pageable/pinned tensors.
import torch
from torch.utils.benchmark import Timer

@torch.inference_mode()
def timer(cmd):
    median = (
        Timer(cmd, globals=globals())
        .adaptive_autorange(min_run_time=1.0, max_run_time=20.0)
        .median
        * 1000
    )
    print(f"{cmd}: {median: 4.4f} ms")
    return median

cpu_datas = []

# A tensor in pageable memory
pageable_tensor = torch.randn(1_000_000)

# A tensor in page-locked (pinned) memory
pinned_tensor = torch.randn(1_000_000, pin_memory=True)

# A CUDA tensor
cuda_tensor = torch.randn(1_000_000, device='cuda')

# Runtimes:
pageable_to_device = timer("pageable_tensor.to('cuda:0')")
pinned_to_device = timer("pinned_tensor.to('cuda:0')")
device_copy_to_pageable = timer("pageable_tensor.data.copy_(cuda_tensor)")
device_copy_to_pinned = timer("pinned_tensor.data.copy_(cuda_tensor)")
device_to_host = timer("cuda_tensor.cpu()")
device_to_host_add_list = timer("cpu_datas.append(cuda_tensor.cpu())")

