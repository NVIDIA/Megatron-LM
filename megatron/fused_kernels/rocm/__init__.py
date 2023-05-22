# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
from torch.utils import cpp_extension
from megatron.fused_kernels.utils import _create_build_dir


# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(args):

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags, extra_include_paths):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'] + extra_cuda_flags,
            extra_include_paths=extra_include_paths,
            verbose=(args.rank == 0)
        )

    # ==============
    # Fused softmax.
    # ==============

    extra_include_paths=[os.path.abspath(srcpath)]

    if args.masked_softmax_fusion:
        extra_cuda_flags = ['-D__HIP_NO_HALF_OPERATORS__=1', '-D__HIP_NO_HALF_CONVERSIONS__=1']
        
        # Upper triangular softmax.
        sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
                 srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']
        scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_upper_triang_masked_softmax_cuda",
            sources, extra_cuda_flags, extra_include_paths)

        # Masked softmax.
        sources=[srcpath / 'scaled_masked_softmax.cpp',
                 srcpath / 'scaled_masked_softmax_cuda.cu']
        scaled_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_masked_softmax_cuda", sources, extra_cuda_flags, extra_include_paths)

    # =================================
    # Mixed precision fused layer norm.
    # =================================

    extra_cuda_flags = []

    sources=[srcpath / 'layer_norm_cuda.cpp',
             srcpath / 'layer_norm_cuda_kernel.cu']
    fused_mix_prec_layer_norm_cuda = _cpp_extention_load_helper(
        "fused_layer_norm_cuda", sources, extra_cuda_flags, extra_include_paths)
