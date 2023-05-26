import torch


def load(args):
    if args.use_kernels_from_apex:
        return

    if torch.version.hip is None:
        if torch.distributed.get_rank() == 0:
            print("running on CUDA devices")
        from megatron.fused_kernels.cuda import load as load_kernels
    else:
        if torch.distributed.get_rank() == 0:
            print("running on ROCm devices")
        from megatron.fused_kernels.rocm import load as load_kernels

    load_kernels(args)
