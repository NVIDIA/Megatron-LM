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

import torch

class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function) :
    """
       Fused operation which performs following three operations in sequence
       1. Scale the tensor. 
       2. Apply upper triangular mask (typically used in gpt models).
       3. Perform softmax.
    """
    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda
        scale_t = torch.tensor([scale])

        softmax_results =  \
            scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda
        softmax_results, scale_t = ctx.saved_tensors

        input_grads =   \
            scaled_upper_triang_masked_softmax_cuda.backward(output_grads,                             
                                                 softmax_results,                          
                                                 scale_t[0])
        return input_grads, None

class FusedScaleMaskSoftmax(torch.nn.Module):
    """
       fused operation: scaling + mask + softmax
       Arguments:
           input_in_fp16: flag to indicate if input in fp16 data format.
           upper_triang_mask: if true, apply upper triangular masking.
                              (used in gpt family networks)
           mask_func: mask function to be applied.
           softmax_in_fp32: if true, softmax in performed at fp32 precision.
           scale: scaling factor used in input tensor scaling.

    """
    def __init__(self, input_in_fp16, upper_triang_mask, 
                 mask_func, softmax_in_fp32, scale):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.upper_triang_mask = upper_triang_mask
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert self.scale is None or softmax_in_fp32, \
            'softmax should be in fp32 when scaled'

    def forward(self, input, mask):
        # [b, np, s, s]
        data_size = input.size()
        assert input.dim() == 4 

        # invoke custom kernel for implicit uuper triangular masking 
        if self.input_in_fp16 and self.upper_triang_mask and \
           data_size[-1] <= 2048 and input.size()[2] == input.size()[3]:
            input = input.view(-1, data_size[2], data_size[3])
            scale = self.scale if self.scale is not None  else 1.0
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale) 
            probs = probs.view(*data_size)
        else:
            if self.input_in_fp16 and self.softmax_in_fp32:
                input = input.float()

            mask_output = self.mask_func(input, mask)           
            if self.scale is not None:
                mask_output = mask_output * self.scale             
            probs = torch.nn.Softmax(dim=-1)(mask_output)

            if self.input_in_fp16 and self.softmax_in_fp32:
                probs = probs.half()

        return probs
