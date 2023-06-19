import os
import torch


# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(args):
    if torch.version.hip is None:
        print("running on CUDA devices")
        from megatron.fused_kernels.cuda import load as load_kernels
    else:
        print("running on ROCm devices")
        from megatron.fused_kernels.rocm import load as load_kernels

    load_kernels(args)
