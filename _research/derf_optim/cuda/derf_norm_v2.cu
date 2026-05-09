// Option 6: hand-tuned CUDA Derf|DyT elementwise norm kernel.
//
// Goal: match TE's hand-tuned RMSNorm kernel on the throughput-relevant axes
// (vectorised loads, register utilisation, occupancy). No matmul fusion;
// pairs with cuBLAS via F.linear or TE's general_gemm. Tests whether the
// gap between Option 4 (Triton norm + cuBLAS, 230 TFLOP/s) and TE's
// baseline (310) is in the norm kernel itself.
//
// Layout: each block handles ROWS_PER_BLOCK rows; each thread handles
// ELTS_PER_THREAD elements per row in a vectorised LDG.128 load. Compute
// in fp32 for `erf`/`tanh` accuracy, store back as input dtype.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIG(x)

// 8 bf16/fp16 elements = 16 bytes (LDG.128). 4 fp32 elements = 16 bytes.
// AT_DISPATCH_FLOATING_TYPES_AND2 surfaces c10::BFloat16 / c10::Half wrappers
// (binary-compatible with __nv_bfloat16 / __half) plus float and double.
template <typename scalar_t> struct VecBytes;
template <> struct VecBytes<c10::BFloat16> { static constexpr int VEC = 8; };
template <> struct VecBytes<c10::Half>     { static constexpr int VEC = 8; };
template <> struct VecBytes<float>         { static constexpr int VEC = 4; };
template <> struct VecBytes<double>        { static constexpr int VEC = 2; };


template <int NORM_KIND>
__device__ __forceinline__ float apply_fz(float z) {
    if constexpr (NORM_KIND == 0) {
        return erff(z);
    } else {
        return tanhf(z);
    }
}


template <typename scalar_t, int NORM_KIND, bool HAS_S, int VEC>
__global__ void derf_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w_norm,   // [K]
    const scalar_t* __restrict__ b_norm,   // [K]
    const scalar_t* __restrict__ alpha_p,  // scalar
    const scalar_t* __restrict__ s_p,      // scalar (or unused)
    scalar_t* __restrict__ out,
    int M, int K
) {
    const int row = blockIdx.x;  // one block per row of [M, K]
    if (row >= M) return;

    const float alpha = static_cast<float>(*alpha_p);
    const float s = HAS_S ? static_cast<float>(*s_p) : 0.0f;

    // Each thread handles K / (blockDim.x) elements per row, in vectors of VEC.
    // For K=1024, blockDim.x=128: each thread handles 8 elements = 1 vector.
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    // Vector loads via reinterpret_cast.
    using vec_t = typename std::aligned_storage<sizeof(scalar_t) * VEC, sizeof(scalar_t) * VEC>::type;
    const vec_t* x_row = reinterpret_cast<const vec_t*>(x + row * K);
    const vec_t* w_vec = reinterpret_cast<const vec_t*>(w_norm);
    const vec_t* b_vec = reinterpret_cast<const vec_t*>(b_norm);
    vec_t* out_row = reinterpret_cast<vec_t*>(out + row * K);

    const int n_vec_per_row = K / VEC;  // assumes K % VEC == 0

    for (int v = tid; v < n_vec_per_row; v += n_threads) {
        // Load VEC elements of x, w, b.
        scalar_t x_arr[VEC];
        scalar_t w_arr[VEC];
        scalar_t b_arr[VEC];
        *reinterpret_cast<vec_t*>(x_arr) = x_row[v];
        *reinterpret_cast<vec_t*>(w_arr) = w_vec[v];
        *reinterpret_cast<vec_t*>(b_arr) = b_vec[v];

        scalar_t out_arr[VEC];
        #pragma unroll
        for (int e = 0; e < VEC; ++e) {
            float xv = static_cast<float>(x_arr[e]);
            float fz = apply_fz<NORM_KIND>(alpha * xv + s);
            float ov = static_cast<float>(w_arr[e]) * fz + static_cast<float>(b_arr[e]);
            out_arr[e] = static_cast<scalar_t>(ov);
        }
        out_row[v] = *reinterpret_cast<vec_t*>(out_arr);
    }
}


template <int NORM_KIND, bool HAS_S>
torch::Tensor _launch_norm(
    torch::Tensor x_2d,
    torch::Tensor w_norm,
    torch::Tensor b_norm,
    torch::Tensor alpha,
    torch::Tensor s
) {
    int M = x_2d.size(0);
    int K = x_2d.size(1);
    auto out = torch::empty_like(x_2d);

    const int n_threads = 128;
    const dim3 block(n_threads);
    const dim3 grid(M);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x_2d.scalar_type(), "derf_norm", [&] {
        constexpr int VEC = VecBytes<scalar_t>::VEC;
        TORCH_CHECK(K % VEC == 0, "K must be divisible by VEC=", VEC);
        derf_norm_kernel<scalar_t, NORM_KIND, HAS_S, VEC><<<grid, block>>>(
            x_2d.data_ptr<scalar_t>(),
            w_norm.data_ptr<scalar_t>(),
            b_norm.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            HAS_S ? s.data_ptr<scalar_t>() : nullptr,
            out.data_ptr<scalar_t>(),
            M, K);
    });
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "derf_norm_kernel: ", cudaGetErrorString(err));
    return out;
}


torch::Tensor derf_norm_fwd(
    torch::Tensor x,
    torch::Tensor w_norm,
    torch::Tensor b_norm,
    torch::Tensor alpha,
    torch::Tensor s
) {
    CHECK_INPUT(x); CHECK_INPUT(w_norm); CHECK_INPUT(b_norm);
    CHECK_INPUT(alpha); CHECK_INPUT(s);
    auto orig_shape = x.sizes().vec();
    int K = orig_shape.back();
    auto x_2d = x.reshape({-1, K}).contiguous();
    auto out_2d = _launch_norm<0, true>(x_2d, w_norm, b_norm, alpha, s);
    return out_2d.reshape(orig_shape);
}


torch::Tensor dyt_norm_fwd(
    torch::Tensor x,
    torch::Tensor w_norm,
    torch::Tensor b_norm,
    torch::Tensor alpha
) {
    CHECK_INPUT(x); CHECK_INPUT(w_norm); CHECK_INPUT(b_norm); CHECK_INPUT(alpha);
    auto orig_shape = x.sizes().vec();
    int K = orig_shape.back();
    auto x_2d = x.reshape({-1, K}).contiguous();
    auto s_dummy = torch::zeros({}, alpha.options());
    auto out_2d = _launch_norm<1, false>(x_2d, w_norm, b_norm, alpha, s_dummy);
    return out_2d.reshape(orig_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("derf_norm_fwd", &derf_norm_fwd, "Derf elementwise norm forward (CUDA, vec-load)");
    m.def("dyt_norm_fwd",  &dyt_norm_fwd,  "DyT elementwise norm forward (CUDA, vec-load)");
}
