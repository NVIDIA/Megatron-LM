# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates 


# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import subprocess

setup_dir = os.path.dirname(os.path.realpath(__file__))
includes_path = os.path.join(setup_dir, 'includes')
hadamard_includes_path = os.path.join(setup_dir, 'hadamard','includes')
nvcc_args = [
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    '-DBF16_AVAILABLE',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_90,code=sm_90',
]

source_files = [
    os.path.join(setup_dir, 'quantization', 'pt_binding.cpp'),
    os.path.join(setup_dir, 'quantization', 'swizzled_quantize.cu'),
    os.path.join(setup_dir, 'quantization', 'swizzled_quantize_ht.cu'),
    os.path.join(setup_dir, 'quantization', 'quant_reduce.cu'),
    os.path.join(setup_dir, 'quantization', 'quant_reduce_ht.cu'),
    os.path.join(setup_dir, 'quantization', 'stochastic_quantize.cu'),
    os.path.join(setup_dir, 'quantization', 'sub_quantize.cu'),
    os.path.join(setup_dir, 'quantization', 'dequantize_add.cu'),
    os.path.join(setup_dir, 'quantization', 'stochastic_quantize_ht.cu'),
    os.path.join(setup_dir, 'quantization', 'dequantize.cu'),
    os.path.join(setup_dir, 'quantization', 'dequant_reduce_quant.cu'),
    os.path.join(setup_dir, 'hadamard', 'hadamard_binding.cpp'),
    os.path.join(setup_dir, 'hadamard', 'fast_hadamard_transform_cuda.cu'),
]

# print compile log when failed, useful when debugging
class BuildExtensionForDebug(BuildExtension):
    def run(self):
        try:
            super().run()
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed with exit code {e.returncode}")
            print(e.output.decode())
        print(f"Successfully complied quantization module")

setup(
    name='quantization_cuda',
    ext_modules=[
        CUDAExtension('quantization_cuda', source_files, include_dirs=[includes_path, hadamard_includes_path], extra_compile_args={'cxx': [], 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtensionForDebug
    }
)