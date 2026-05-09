// Option 3: hand-written CUDA fused (Derf|DyT) + Linear forward.
//
// Performs y = (gamma * f(alpha*x + s) + beta) @ W^T  [+ b_lin]
// where f = erf for Derf, f = tanh for DyT.
//
// Naive thread-per-output-element kernel: each thread loops over K, applying
// the norm in registers and accumulating one output. Correctness-first
// (no SMEM tiling, no tensor cores). Won't beat cuBLAS — the goal is a
// fair "what does plain CUDA fusion buy?" data point next to Triton.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIG(x)


template <int NORM_KIND>
__device__ __forceinline__ float apply_norm(
    float x, float gamma, float beta, float alpha, float s
) {
    float z = alpha * x + s;
    float fz;
    if constexpr (NORM_KIND == 0) {
        fz = erff(z);
    } else {
        fz = tanhf(z);
    }
    return gamma * fz + beta;
}


// One thread per (m, n) output element. block.x indexes N (contiguous),
// blockIdx.y indexes M. Each thread loops over the K dim.
template <typename scalar_t, int NORM_KIND, bool HAS_S, bool HAS_B_LIN>
__global__ void norm_linear_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w_lin,
    const scalar_t* __restrict__ w_norm,
    const scalar_t* __restrict__ b_norm,
    const scalar_t* __restrict__ alpha_p,
    const scalar_t* __restrict__ s_p,
    const scalar_t* __restrict__ b_lin,
    scalar_t* __restrict__ out,
    int M, int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const float alpha = static_cast<float>(*alpha_p);
    const float s = HAS_S ? static_cast<float>(*s_p) : 0.0f;

    float accum = 0.0f;
    const scalar_t* x_row = x + m * K;
    const scalar_t* w_row = w_lin + n * K;

    for (int k = 0; k < K; ++k) {
        float xv = static_cast<float>(x_row[k]);
        float gamma = static_cast<float>(w_norm[k]);
        float beta = static_cast<float>(b_norm[k]);
        float pre = apply_norm<NORM_KIND>(xv, gamma, beta, alpha, s);
        accum += pre * static_cast<float>(w_row[k]);
    }
    if constexpr (HAS_B_LIN) {
        accum += static_cast<float>(b_lin[n]);
    }
    out[m * N + n] = static_cast<scalar_t>(accum);
}


template <int NORM_KIND, bool HAS_S>
torch::Tensor _launch_fwd(
    torch::Tensor x_2d,
    torch::Tensor w_lin,
    torch::Tensor w_norm,
    torch::Tensor b_norm,
    torch::Tensor alpha,
    torch::Tensor s,
    torch::Tensor b_lin
) {
    int M = x_2d.size(0);
    int K = x_2d.size(1);
    int N = w_lin.size(0);

    auto out = torch::empty({M, N}, x_2d.options());

    constexpr int THREADS_X = 128;
    dim3 block(THREADS_X, 1);
    dim3 grid((N + THREADS_X - 1) / THREADS_X, M);

    bool has_b_lin = b_lin.defined() && b_lin.numel() > 0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x_2d.scalar_type(), "norm_linear_fwd", [&] {
        if (has_b_lin) {
            norm_linear_fwd_kernel<scalar_t, NORM_KIND, HAS_S, true><<<grid, block>>>(
                x_2d.data_ptr<scalar_t>(),
                w_lin.data_ptr<scalar_t>(),
                w_norm.data_ptr<scalar_t>(),
                b_norm.data_ptr<scalar_t>(),
                alpha.data_ptr<scalar_t>(),
                HAS_S ? s.data_ptr<scalar_t>() : nullptr,
                b_lin.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                M, N, K);
        } else {
            norm_linear_fwd_kernel<scalar_t, NORM_KIND, HAS_S, false><<<grid, block>>>(
                x_2d.data_ptr<scalar_t>(),
                w_lin.data_ptr<scalar_t>(),
                w_norm.data_ptr<scalar_t>(),
                b_norm.data_ptr<scalar_t>(),
                alpha.data_ptr<scalar_t>(),
                HAS_S ? s.data_ptr<scalar_t>() : nullptr,
                nullptr,
                out.data_ptr<scalar_t>(),
                M, N, K);
        }
    });
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch: ", cudaGetErrorString(err));
    return out;
}


torch::Tensor derf_linear_fwd(
    torch::Tensor x,
    torch::Tensor w_lin,
    torch::Tensor w_norm,
    torch::Tensor b_norm,
    torch::Tensor alpha,
    torch::Tensor s,
    torch::Tensor b_lin  /* empty () when no bias */
) {
    CHECK_INPUT(x); CHECK_INPUT(w_lin); CHECK_INPUT(w_norm); CHECK_INPUT(b_norm);
    CHECK_INPUT(alpha); CHECK_INPUT(s);
    if (b_lin.defined() && b_lin.numel() > 0) CHECK_INPUT(b_lin);

    auto orig_shape = x.sizes().vec();
    int K = orig_shape.back();
    auto x_2d = x.reshape({-1, K}).contiguous();

    auto out_2d = _launch_fwd<0, true>(x_2d, w_lin, w_norm, b_norm, alpha, s, b_lin);

    int N = w_lin.size(0);
    orig_shape.back() = N;
    return out_2d.reshape(orig_shape);
}


torch::Tensor dyt_linear_fwd(
    torch::Tensor x,
    torch::Tensor w_lin,
    torch::Tensor w_norm,
    torch::Tensor b_norm,
    torch::Tensor alpha,
    torch::Tensor b_lin  /* empty () when no bias */
) {
    CHECK_INPUT(x); CHECK_INPUT(w_lin); CHECK_INPUT(w_norm); CHECK_INPUT(b_norm);
    CHECK_INPUT(alpha);
    if (b_lin.defined() && b_lin.numel() > 0) CHECK_INPUT(b_lin);

    auto orig_shape = x.sizes().vec();
    int K = orig_shape.back();
    auto x_2d = x.reshape({-1, K}).contiguous();

    auto s_dummy = torch::zeros({}, alpha.options());
    auto out_2d = _launch_fwd<1, false>(x_2d, w_lin, w_norm, b_norm, alpha, s_dummy, b_lin);

    int N = w_lin.size(0);
    orig_shape.back() = N;
    return out_2d.reshape(orig_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("derf_linear_fwd", &derf_linear_fwd, "Fused Derf + Linear forward (CUDA)");
    m.def("dyt_linear_fwd", &dyt_linear_fwd, "Fused DyT + Linear forward (CUDA)");
}
