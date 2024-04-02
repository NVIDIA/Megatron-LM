// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cassert>
#include <vector>
#include "quantization.h"

std::vector<at::Tensor> quantized_reduction(at::Tensor& input_vals,
                            at::Tensor& input_scales,
                            int in_groups,
                            int out_groups,
                            int num_bits,
                            quantize::Type quant_type,
                            int devices_per_node)
{

    auto output_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int granularity = 8 / num_bits;
    std::vector<long int> sz(input_vals.sizes().begin(), input_vals.sizes().end());
    sz[sz.size() - 1] = sz.back() / devices_per_node * granularity;
    const int elems_per_in_tensor = at::numel(input_vals) / devices_per_node;
    auto output = torch::empty(sz, output_options);

    const int elems_per_in_group = elems_per_in_tensor / (in_groups / devices_per_node);
    const int elems_per_out_group = elems_per_in_tensor / out_groups;

    launch_dequant_reduce((float*)output.data_ptr(),
                          (const int8_t*)input_vals.data_ptr(),
                          (const float*)input_scales.data_ptr(),
                          devices_per_node,
                          num_bits,
                          quant_type,
                          out_groups,
                          elems_per_out_group,
                          elems_per_in_tensor,
                          in_groups / devices_per_node,
                          elems_per_in_group,
                          at::cuda::getCurrentCUDAStream());
    return {output};
}

std::vector<at::Tensor> ds_swizzle_quant(at::Tensor& input_vals,
                                         int groups,
                                         int num_bits,
                                         quantize::Type quant_type,
                                         int pipeline_size,
                                         int nodes,
                                         int devices_per_node)
{
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
    auto scales = torch::empty({groups, scales_elems}, scales_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    const int quantization_scalar = 8 / num_bits;
    const int compressed_vals = at::numel(input_vals) / quantization_scalar;

    auto output = torch::empty({compressed_vals}, output_options);
    const int elems_per_group = at::numel(input_vals) / groups;
    
    // Check data type and launch appropriate kernel
    if (input_vals.scalar_type() == at::ScalarType::Half) {
        launch_swizzled_quant((int8_t*)output.data_ptr(),
                            (float*)scales.data_ptr(),
                            (__half*)input_vals.data_ptr(),
                            num_bits,
                            quant_type,
                            groups,
                            elems_per_group,
                            pipeline_size,
                            nodes,
                            devices_per_node,
                            at::cuda::getCurrentCUDAStream());
    } else if (input_vals.scalar_type() == at::ScalarType::Float) {
        launch_swizzled_quant((int8_t*)output.data_ptr(),
                            (float*)scales.data_ptr(),
                            (float*)input_vals.data_ptr(),
                            num_bits,
                            quant_type,
                            groups,
                            elems_per_group,
                            pipeline_size,
                            nodes,
                            devices_per_node,
                            at::cuda::getCurrentCUDAStream());
    }


    return {output, scales};
}

std::vector<at::Tensor> stochastic_quantize(at::Tensor& input_vals,
                                            int groups,
                                            int numBits,
                                            quantize::Type quantType)
{
    auto dtype = at::kFloat;
    auto params_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int param_elems = (quantize::requires_offset(quantType)) ? 2 : 1;
    auto params = torch::empty({groups, param_elems}, params_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    auto output_sizes = input_vals.sizes().vec();
    output_sizes[output_sizes.size() - 1] /= numBits == 8 ? 1 : 2;
    auto output = torch::empty(output_sizes, output_options);

    const int elems_per_group = at::numel(input_vals) / groups;

    if (input_vals.scalar_type() == at::ScalarType::Half) {
        launch_quant((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (__half*)input_vals.data_ptr(),
                    groups,
                    elems_per_group,
                    numBits,
                    quantType,
                    at::cuda::getCurrentCUDAStream());
        return {output, params};
    } else if (input_vals.scalar_type() == at::ScalarType::Float) {
        launch_quant((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (float*)input_vals.data_ptr(),
                    groups,
                    elems_per_group,
                    numBits,
                    quantType,
                    at::cuda::getCurrentCUDAStream());
        return {output, params};
    } else {
        throw std::runtime_error("Unsupported input tensor data type.");
    }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::enum_<quantize::Type>(m, "QuantizationType")
        .value("Symmetric", quantize::Type::Symmetric)
        .value("Asymmetric", quantize::Type::Asymmetric)
        .export_values();
    m.def("swizzle_quant", &ds_swizzle_quant);
    m.def("quantized_reduction", &quantized_reduction);
    m.def("stochastic_quantize", &stochastic_quantize);
}
