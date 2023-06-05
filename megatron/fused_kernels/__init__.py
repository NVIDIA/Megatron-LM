import torch


def load(args):
    if torch.version.hip is None:
        print("running on CUDA devices")
        from megatron.fused_kernels.cuda import load as load_kernels
    else:
        print("running on ROCm devices")
        from megatron.fused_kernels.rocm import load as load_kernels

    load_kernels(args)
