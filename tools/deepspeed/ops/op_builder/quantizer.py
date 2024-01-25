# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder
from deepspeed.accelerator import get_accelerator
import math

class QuantizerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_QUANTIZER"
    NAME = "quantizer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.quantizer.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/quantization/pt_binding.cpp',
            'csrc/quantization/fake_quantizer.cu',
            'csrc/quantization/quantize.cu',
            'csrc/quantization/quantize_intX.cu',
            'csrc/quantization/dequantize.cu',
            'csrc/quantization/swizzled_quantize.cu',
            'csrc/quantization/quant_reduce.cu',
        ]

    def include_paths(self):
        return ['csrc/includes']

    def extra_ldflags(self):
        return ['-lcurand']

class CUDAQuantizer:
    async_flag = True
    target_group_size = 8000  # the optimal size is 4k, so we set the target to be below 8k
    group_size_cache = dict()
    quantizer_cuda_module = None

    def __init__(self) -> None:
        if CUDAQuantizer.quantizer_cuda_module is None:
            CUDAQuantizer.quantizer_cuda_module = QuantizerBuilder().load()

    def quantize(self, param, groups=None, quantization_bits=4):
        if groups is None:
            try:
                groups = self.group_size_cache[param.numel()]
            except KeyError:
                groups = math.ceil(param.numel() / self.target_group_size)
                while groups < param.numel():
                    if param.numel() % (8 * groups) == 0:
                        break
                    groups += 1
                while True:
                    if param.numel() % (8 * groups * 2) == 0 and param.numel(
                    ) / groups > self.target_group_size:  #hard limit of 16k group_size
                        groups *= 2
                    else:
                        break
                assert (
                    param.numel() % (8 * groups) == 0
                ), f"Qantized weight requires the number of weights be a multiple of 8. Yet {param.numel()} cannot be divided by 8*{groups}"
                assert (param.numel() / groups < 16000), f"{param.numel()} / {groups} is larger than 16k"
                assert param.numel(
                ) > groups, f"Adaptive grouping algorithm cannot find a group size for input tensor of size {param.numel()}"
                self.group_size_cache[param.numel()] = groups
        return self.quantizer_cuda_module.quantize(param.to(get_accelerator().device_name()), groups, quantization_bits,
                                                   self.quantizer_cuda_module.Symmetric)

    def dequantize(self, quantized_param, scale, quantization_bits=4):
        return self.quantizer_cuda_module.dequantize(quantized_param, scale, scale.numel(), quantization_bits,
                                                     self.quantizer_cuda_module.Symmetric)