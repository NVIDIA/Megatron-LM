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

import pathlib
import subprocess
from torch.utils import cpp_extension

def load_scaled_upper_triang_masked_softmax_fusion_kernel():

    def get_cuda_bare_metal_version(cuda_dir):
        raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], 
                                             universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]

        return raw_output, bare_metal_major, bare_metal_minor

    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')

    srcpath = pathlib.Path(__file__).parent.absolute()
    scaled_upper_triang_masked_softmax_cuda = cpp_extension.load(
        name='scaled_upper_triang_masked_softmax_cuda', 
        sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp', 
                 srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu'], 
        extra_cflags=['-O3',],
        extra_cuda_cflags=['-O3',
                           '-gencode', 'arch=compute_70,code=sm_70',
                           '-U__CUDA_NO_HALF_OPERATORS__',
                           '-U__CUDA_NO_HALF_CONVERSIONS__',
                           '--expt-relaxed-constexpr',
                           '--expt-extended-lambda',
                           '--use_fast_math'] + cc_flag,
        verbose=True)
