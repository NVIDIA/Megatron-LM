def load(args):
    if args.use_kernels_from_apex:
        return

    if args.device == "cuda":
        from megatron.fused_kernels.cuda import load as load_kernels
    elif args.device == "rocm":
        from megatron.fused_kernels.rocm import load as load_kernels

    load_kernels(args)
