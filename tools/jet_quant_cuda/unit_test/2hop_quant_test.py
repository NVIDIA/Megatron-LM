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

import subprocess
import sys
import os
from importlib import import_module
import torch
from dequant_function import *
from tool_function import *

def unittest_dequant(groupsize=128, N=12800000, quant_module=None, inter_dp_size=1):
    # tensor = torch.normal(std=10, mean=0, size=(N, ), device=torch.cuda.current_device(), dtype=torch.float)
    tensor = torch.rand(size=(N, ), device=torch.cuda.current_device(), dtype=torch.float)
    print(tensor[:128])
    print(tensor.shape, tensor.dtype)

    # Prepare quantized tensor
    quant_tensor_torch = quantize_4bits(tensor, groupsize)
    quantized_tensor_view = quant_tensor_torch.view(N // groupsize, -1)
    x_quant = quantized_tensor_view[:, :groupsize // 2].clone()
    x_scale = quantized_tensor_view[:, groupsize // 2:].clone()
    print(x_scale.view(torch.float32)[:10])

    # Use pytorch dequant
    dequantized_tensor = dequantize_4bits(x_quant, x_scale, groupsize)
    dequantized_tensor = dequantized_tensor.view(-1)
    dequantized_tensor = (sum(list(dequantized_tensor.chunk(inter_dp_size)))).view(-1)
    dequant_tensor_torch = dequantized_tensor.view(-1)
    # print(dequant_tensor_torch[:128], dequant_tensor_torch.shape)

    # cuda Prepare quantized tensor
    inter_quant_group = N // groupsize
    quant_tensor_cuda, quant_scales_cuda = quant_module.stochastic_quantize(tensor, inter_quant_group, 4, quant_module.Symmetric)
    print(quant_scales_cuda[:10])

    # Use quant_module reduce
    dequant_output, = quant_module.quantized_reduction(quant_tensor_cuda, 
                                                        quant_scales_cuda, 
                                                        inter_quant_group, 
                                                        inter_quant_group // inter_dp_size, 
                                                        4, 
                                                        quant_module.Symmetric,
                                                        inter_dp_size)
    # dequant_output = dequantize_4bits(quant_tensor_cuda, quant_scales_cuda, groupsize)

    dequant_tensor_cuda = dequant_output.view(-1)
    # print(dequant_tensor_cuda[:128], dequant_tensor_cuda.shape)

    tensor = (sum(list(tensor.chunk(inter_dp_size)))).view(-1)
    abs_error_norm, rela_error_norm = analysis_diff(tensor, dequant_tensor_torch)
    print(f"torch version quantization, absolute error norm: {abs_error_norm}, relative error norm: {rela_error_norm}")

    abs_error_norm, rela_error_norm = analysis_diff(tensor, dequant_tensor_cuda)
    print(f"cuda version quantization, absolute error norm: {abs_error_norm}, relative error norm: {rela_error_norm}")

    print(tensor[:128], '\n', dequant_tensor_torch[:128], '\n', dequant_tensor_cuda[:128])
    print(tensor[:128].abs().max(), dequant_tensor_torch[:128].abs().max(), dequant_tensor_cuda[:128].abs().max())

def unittest_dequant_second(groupsize=128, N=25600000, quant_module=None, inter_dp_size=1, hadamard=False):
    # tensor = torch.normal(std=10, mean=0, size=(N, ), device=torch.cuda.current_device(), dtype=torch.float)
    # tensor = torch.rand(size=(N, ), device=torch.cuda.current_device(), dtype=torch.float)
    tensor = torch.load('tensor.pt', map_location='cuda')[:N].to(torch.bfloat16)
    # tensor = tensor / 1_000
    # print(tensor[:128])
    print(tensor.shape, tensor.dtype)
    h_tensor = tensor
    if hadamard is True:
        h_tensor = fast_hadamard_transform(h_tensor, k=5, normalize=True)
    
    # Prepare quantized tensor
    quant_tensor_torch = quantize_8bits(h_tensor.clone(), groupsize)
    quantized_tensor_view = quant_tensor_torch.view(N // groupsize, -1)
    x_quant = quantized_tensor_view[:, :groupsize].clone()
    x_scale = quantized_tensor_view[:, groupsize:].clone()
    # print(x_scale.view(torch.float32)[:10])

    # Use pytorch dequant
    dequantized_tensor = dequantize_nbits(x_quant, x_scale, groupsize)
    dequantized_tensor = dequantized_tensor.view(-1)
    dequantized_tensor = (sum(list(dequantized_tensor.chunk(inter_dp_size)))).view(-1)
    dequant_tensor_torch = dequantized_tensor.view(-1)

    # fake second quantize
    N = dequant_tensor_torch.nelement()
    quant_tensor_torch = quantize_4bits(dequant_tensor_torch, groupsize)
    quantized_tensor_view = quant_tensor_torch.view(N // groupsize, -1)
    x_quant = quantized_tensor_view[:, :groupsize // 2].clone()
    x_scale = quantized_tensor_view[:, groupsize // 2:].clone()

    dequantized_tensor = dequantize_4bits(x_quant, x_scale, groupsize)
    dequantized_tensor = dequantized_tensor.view(-1)
    dequantized_tensor = (sum(list(dequantized_tensor.chunk(inter_dp_size)))).view(-1)
    dequant_tensor_torch = dequantized_tensor.view(-1)

    if hadamard is True:
        dequant_tensor_torch = fast_hadamard_transform(dequant_tensor_torch, k=5, normalize=True)
    print(dequant_tensor_torch.shape, dequant_tensor_torch.dtype)

    # cuda Prepare quantized tensor
    N = tensor.nelement()
    print(N)
    inter_quant_group = N // groupsize
    quant_tensor_cuda, quant_scales_cuda = quant_module.stochastic_quantize(h_tensor.clone(), inter_quant_group, 8, quant_module.Symmetric)


    # Use quant_module reduce
    dequant_output, = quant_module.quantized_reduction(quant_tensor_cuda, 
                                                        quant_scales_cuda, 
                                                        inter_quant_group, 
                                                        inter_quant_group // inter_dp_size, 
                                                        8, 
                                                        quant_module.Symmetric,
                                                        inter_dp_size)
    N = dequant_output.nelement()
    print(N)
    inter_quant_group = N // groupsize
    quant_tensor_cuda, quant_scales_cuda = quant_module.stochastic_quantize(dequant_output, inter_quant_group, 4, quant_module.Symmetric)


    # second quant
    dequant_output, = quant_module.quantized_reduction(quant_tensor_cuda, 
                                                        quant_scales_cuda, 
                                                        inter_quant_group, 
                                                        inter_quant_group // inter_dp_size, 
                                                        4, 
                                                        quant_module.Symmetric,
                                                        inter_dp_size)
    
    dequant_tensor_cuda = dequant_output.view(-1)
    if hadamard is True:
        dequant_tensor_cuda = fast_hadamard_transform(dequant_tensor_cuda, k=5, normalize=True)
    print(dequant_tensor_cuda.shape)

    # tensor = (sum(list(tensor.chunk(inter_dp_size)))).view(-1)
    tensor = torch.sum(torch.stack(torch.chunk(tensor, inter_dp_size, dim=-1)), dim=0)
    tensor = torch.sum(torch.stack(torch.chunk(tensor, inter_dp_size, dim=-1)), dim=0)

    abs_error_norm, rela_error_norm = analysis_diff(tensor, dequant_tensor_torch)
    print(f"torch version quantization, absolute error norm: {abs_error_norm}, relative error norm: {rela_error_norm}")

    abs_error_norm, rela_error_norm = analysis_diff(tensor, dequant_tensor_cuda)
    print(f"cuda version quantization, absolute error norm: {abs_error_norm}, relative error norm: {rela_error_norm}")

    # print(tensor[:128], '\n', dequant_tensor_torch[:128], '\n', dequant_tensor_cuda[:128])
    # print(tensor[:128].abs().max(), dequant_tensor_torch[:128].abs().max(), dequant_tensor_cuda[:128].abs().max())

if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.set_printoptions(sci_mode=False)

    pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
    print('pkg path:', pkg_path)

    quantization_module = build_and_import_module(pkg_path, 'quantization_cuda')
    # unittest_dequant(quant_module=quantization_module, inter_dp_size=1)
    unittest_dequant_second(quant_module=quantization_module, inter_dp_size=8, hadamard=False)
