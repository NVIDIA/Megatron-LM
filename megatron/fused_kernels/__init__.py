# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import pathlib
import subprocess

from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(args):

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
        cpp_extension.CUDA_HOME)
    compute_cap = _get_compute_cap()
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')
        if int(bare_metal_minor) >= 7 and compute_cap == '9.0':
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_70,code=sm_70',
                               '--use_fast_math'] + extra_cuda_flags + cc_flag,
            verbose=(args.rank == 0)
        )

    # ==============
    # Fused softmax.
    # ==============

    if args.masked_softmax_fusion:
        extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '--expt-relaxed-constexpr',
                            '--expt-extended-lambda']
        
        # Upper triangular softmax.
        sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
                 srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']
        scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_upper_triang_masked_softmax_cuda",
            sources, extra_cuda_flags)

        # Masked softmax.
        sources=[srcpath / 'scaled_masked_softmax.cpp',
                 srcpath / 'scaled_masked_softmax_cuda.cu']
        scaled_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_masked_softmax_cuda", sources, extra_cuda_flags)

        # Softmax
        sources=[srcpath / 'scaled_softmax.cpp',
                 srcpath / 'scaled_softmax_cuda.cu']
        scaled_softmax_cuda = _cpp_extention_load_helper(
            "scaled_softmax_cuda", sources, extra_cuda_flags)

    # =================================
    # Mixed precision fused layer norm.
    # =================================

    extra_hopper_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                          '-U__CUDA_NO_HALF_CONVERSIONS__']

    extra_cuda_flags = ['-maxrregcount=50']
    sources=[srcpath / 'layer_norm_cuda.cpp',
             srcpath / 'layer_norm_cuda_kernel.cu']
    fused_mix_prec_layer_norm_cuda = _cpp_extention_load_helper(
        "fused_mix_prec_layer_norm_cuda", sources, extra_cuda_flags + extra_hopper_flags)

    # =================================
    # Fused gradient accumulation to weight gradient computation of linear layer
    # =================================

    if args.gradient_accumulation_fusion:
        sources=[srcpath / 'fused_weight_gradient_dense.cpp',
                 srcpath / 'fused_weight_gradient_dense.cu']
        fused_dense_cuda = _cpp_extention_load_helper(
            "fused_dense_cuda", sources, extra_hopper_flags)


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def _get_compute_cap():
    raw_output = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"],
                                         universal_newlines=True)
    output = raw_output.split()
    # example output on A100 ['compute_cap', '8.0', '8.0', '8.0', '8.0', '8.0', '8.0', '8.0', '8.0']
    return output[1]

def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
