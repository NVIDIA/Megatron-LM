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

import torch

def quantize_4bits(x, groupsize=-1):
    bits = 4

    assert len(list(x.shape)) == 1
    assert groupsize % 2 == 0
    x_shape = list(x.size())[0]
    d = 2 ** (bits - 1)-1 ###

    if groupsize == -1:
        norm = torch.max(torch.abs(x))
        group_x = x
    else:
        assert list(x.shape)[0] % groupsize == 0
        group_x = x.view(
            -1,
            groupsize,
        )
        norm, _ = torch.max(group_x.abs(), -1, keepdim=True)
        norm[norm==0] = 2 ** (bits - 1) - 1 ###

    # level_float = d * torch.abs(group_x) / norm
    level_float = d * torch.clamp(torch.abs(group_x) / norm, max=1)
    previous_level = torch.floor(level_float)
    # is_next_level = 0.5 < (level_float - previous_level)
    is_next_level = torch.rand(group_x.size(), device=group_x.device) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    scale = norm.to(torch.float) / d
    scale = scale.view(torch.int8)
    x_quant = torch.sign(group_x) * new_level
    x_quant = x_quant.to(torch.int8)
    x_quant = x_quant.view(x_shape)
    # print('x_quant before tensor:', x_quant)
    x_quant = use_1int8_represent_2int4(int4_input=x_quant).view(-1, groupsize // 2)

    # print('x_scale before tensor:', scale.view(torch.float32))

    return torch.cat((x_quant, scale), 1)
    # return x_quant, scale

def quantize_8bits(x, groupsize=-1):
    bits = 8

    assert len(list(x.shape)) == 1
    assert groupsize % 2 == 0
    x_shape = list(x.size())[0]
    d = 2 ** (bits - 1)-1 ###

    if groupsize == -1:
        norm = torch.max(torch.abs(x))
        group_x = x
    else:
        assert list(x.shape)[0] % groupsize == 0
        group_x = x.view(
            -1,
            groupsize,
        )
        norm, _ = torch.max(group_x.abs(), -1, keepdim=True)
        norm[norm==0] = 2 ** (bits - 1) - 1 ###

    # level_float = d * torch.abs(group_x) / norm
    level_float = d * torch.clamp(torch.abs(group_x) / norm, max=1)
    previous_level = torch.floor(level_float)
    # is_next_level = 0.5 < (level_float - previous_level)
    is_next_level = torch.rand(group_x.size(), device=group_x.device) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    scale = norm.float() / d
    scale = scale.view(torch.int8)
    x_quant = torch.sign(group_x) * new_level
    x_quant = x_quant.to(torch.int8)
    x_quant = x_quant.view(-1, groupsize)

    return torch.cat((x_quant, scale), 1)

def dequantize_4bits(x, s, groupsize=-1):

    x = use_2int4_represent_1int8(x).to(torch.float32)
    s = s.view(torch.float32)
    # print('x_scale tensor:', s.view(torch.float32))
    # print('x_quant', x)

    if groupsize == -1:
        group_x = x
    else:
        group_x = x.view(
            -1,
            groupsize,
        )
    group_x.mul_(s)
    x_dequant = group_x.view(-1)

    return x_dequant

def dequantize_nbits(x: torch.int8, s:torch.float32, groupsize=-1):
    x = x.to(torch.float32)
    s = s.view(torch.float32).view(-1, 1)

    if groupsize == -1:
        group_x = x
    else:
        group_x = x.view(
            -1,
            groupsize,
        )
    group_x.mul_(s)
    x_dequant = group_x.view(-1)

    return x_dequant

def use_1int8_represent_2int4(int4_input):
    assert len(list(int4_input.shape)) == 1
    assert list(int4_input.shape)[0] % 2 == 0
    half = list(int4_input.shape)[0] // 2
    a, b = int4_input[::2], int4_input[1::2]

    packed = (a << 4) | (b & 0b00001111)

    return packed

def use_2int4_represent_1int8(int8_input):
    a_unpacked = int8_input >> 4
    b_unpacked = int8_input << 4 >> 4

    unpacked = torch.stack((a_unpacked.view(-1), b_unpacked.view(-1))).transpose(0, 1).flatten()

    return unpacked

def fast_hadamard_transform(input_, k=5, normalize=False):
    dim_size = list(input_.size())
    m = 1 << k
    assert dim_size[-1] % m == 0, 'size of last dim must be divisible by 2^k'
    x = input_.unsqueeze(-1)
    for _ in range(k):
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    result = x.squeeze(-2) / 2**(k / 2) if normalize else x.squeeze(-2)
    return result.view(input_.size())