// Option 3: hand-written CUDA fused (Derf|DyT) + Linear forward.
//
// Performs y = (gamma * f(alpha*x + s) + beta) @ W^T  [+ b_lin]
// where f = erf for Derf and f = tanh for DyT.
//
// Tiled SMEM matmul (no tensor cores) — correctness-first reference. Beats
// the unfused path by avoiding the round-trip through global memory between
// the norm and the matmul. Won't beat cuBLAS-only for the matmul itself but
// is a fair "what does CUDA-level fusion buy?" data point next to Triton.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIG(x)

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;

// Derf prologue applied per-element after loading x, before matmul accumulate.
template <int NORM_KIND>
__device__ __forceinline__ float apply_norm(
    float x, float gamma, float beta, float alpha, float s
) {
    float z = alpha * x + s;
    float fz;
    if constexpr (NORM_KIND == 0) {
        fz = erff(z);  // Derf
    } else {
        fz = tanhf(z);  // DyT
    }
    return gamma * fz + beta;
}

// Forward kernel: out[m, n] = sum_k pre[m, k] * w[n, k]
// where pre[m, k] = gamma[k] * f(alpha*x[m, k] + s) + beta[k].
// One thread block computes a BLOCK_M x BLOCK_N tile.
template <typename scalar_t, int NORM_KIND, bool HAS_S>
__global__ void norm_linear_fwd_kernel(
    const scalar_t* __restrict__ x,         // [M, K]
    const scalar_t* __restrict__ w_lin,     // [N, K]
    const scalar_t* __restrict__ w_norm,    // [K]
    const scalar_t* __restrict__ b_norm,    // [K]
    const scalar_t* __restrict__ alpha_p,   // scalar
    const scalar_t* __restrict__ s_p,       // scalar (or unused)
    const scalar_t* __restrict__ b_lin,     // [N] or null
    scalar_t* __restrict__ out,             // [M, N]
    int M, int N, int K
) {
    int row = blockIdx.x * BLOCK_M + threadIdx.y;  // M
    int col = blockIdx.y * BLOCK_N + threadIdx.x;  // N

    __shared__ float pre_smem[BLOCK_M][BLOCK_K];
    __shared__ float w_smem[BLOCK_N][BLOCK_K];

    float alpha = static_cast<float>(*alpha_p);
    float s = HAS_S ? static_cast<float>(*s_p) : 0.0f;

    float accum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load x tile, apply norm, store to smem (one thread per tile element).
        if (threadIdx.x < BLOCK_K) {
            int kk = k0 + threadIdx.x;
            int m = row;
            float xv = (m < M && kk < K) ? static_cast<float>(x[m * K + kk]) : 0.0f;
            float gamma = (kk < K) ? static_cast<float>(w_norm[kk]) : 0.0f;
            float beta = (kk < K) ? static_cast<float>(b_norm[kk]) : 0.0f;
            pre_smem[threadIdx.y][threadIdx.x] = (m < M && kk < K)
                ? apply_norm<NORM_KIND>(xv, gamma, beta, alpha, s)
                : 0.0f;
        }
        // Load w tile.
        if (threadIdx.y < BLOCK_K) {
            int kk = k0 + threadIdx.y;
            int n = col;
            w_smem[threadIdx.x][threadIdx.y] = (n < N && kk < K)
                ? static_cast<float>(w_lin[n * K + kk])
                : 0.0f;
        }
        __syncthreads();

        // Compute partial sum for this k-tile.
        if (row < M && col < N) {
            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                accum += pre_smem[threadIdx.y][kk] * w_smem[threadIdx.x][kk];
            }
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (b_lin != nullptr) {
            accum += static_cast<float>(b_lin[col]);
        }
        out[row * N + col] = static_cast<scalar_t>(accum);
    }
}

template <int NORM_KIND, bool HAS_S>
torch::Tensor _launch_fwd(
    torch::Tensor x_2d,
    torch::Tensor w_lin,
    torch::Tensor w_norm,
    torch::Tensor b_norm,
    torch::Tensor alpha,
    torch::Tensor s,
    torch::Tensor b_lin  /* empty () when no bias */
) {
    int M = x_2d.size(0);
    int K = x_2d.size(1);
    int N = w_lin.size(0);

    auto out = torch::empty({M, N}, x_2d.options());

    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x_2d.scalar_type(), "norm_linear_fwd", [&] {
        norm_linear_fwd_kernel<scalar_t, NORM_KIND, HAS_S><<<grid, block>>>(
            x_2d.data_ptr<scalar_t>(),
            w_lin.data_ptr<scalar_t>(),
            w_norm.data_ptr<scalar_t>(),
            b_norm.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            HAS_S ? s.data_ptr<scalar_t>() : nullptr,
            (b_lin.defined() && b_lin.numel() > 0) ? b_lin.data_ptr<scalar_t>() : nullptr,
            out.data_ptr<scalar_t>(),
            M, N, K
        );
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
