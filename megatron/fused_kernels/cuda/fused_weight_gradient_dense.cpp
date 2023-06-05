#include <torch/torch.h>
#include <torch/extension.h>

#include <vector>
#include <stdio.h>

#include "type_shim.h"


template <typename T>
int wgrad_gemm_accum_fp32_cuda(T *input, T *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);

void wgrad_gemm_accum_fp32(const at::Tensor input, const at::Tensor d_output, at::Tensor d_weight) {
    at::Tensor input_2d, d_output_2d;
    // input tensor: collapse to the first dim
    auto in_sizes = input.sizes();
    if (input.dim() > 2) {
        input_2d = input.view({-1, in_sizes[in_sizes.size() - 1]});
    } else {
        input_2d = input;
    }
    // d_output tensor: collapse to the first dim
    auto d_out_sizes = d_output.sizes();
    if (d_output.dim() > 2) {
        d_output_2d = d_output.view({-1, d_out_sizes[d_out_sizes.size() - 1]});
    } else {
        d_output_2d = d_output;
    }

    int hidden_dim = input_2d.size(0);
    int in_dim = input_2d.size(1);
    int out_dim = d_weight.size(0);

    DISPATCH_HALF_BFLOAT_AND_FLOAT(input_2d.scalar_type(), "wgrad_gemm_accum_fp32",
        int result = wgrad_gemm_accum_fp32_cuda<scalar_t>(
            input_2d.data_ptr<scalar_t>(),
            d_output_2d.data_ptr<scalar_t>(),
            d_weight.data_ptr<float>(),
            in_dim,
            hidden_dim,
            out_dim);
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wgrad_gemm_accum_fp32", &wgrad_gemm_accum_fp32, "wgrad gemm accum in fp32");
}
