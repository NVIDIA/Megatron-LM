// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Sparse MLA backward with attention sink for the DS4.1 mega attention core
// (64 heads, K = V, d = 512). Reads the unrotated query in the kernel's fused
// layout and the output gradient in the fused output layout, applies the RoPE
// VJPs itself and writes dq in the fused query layout. One CTA per query row holds
// all 64 heads and walks the row's index list in blocks of 32 KV slots twice:
//   pass 1: S = Q K^T, dP = dO K^T, P = exp(S - LSE), D = sum_j P dP;
//   pass 2: dS = scale * P (dP - D); dQ += dS K; dK_j += dS^T Q + P^T dO.
// bf16 mma.sync (m16n8k16) with FP32 accumulation; dK is added to the FP32
// dKV buffer with atomics. D is reduced in a fixed order and written out so
// the caller forms dsink = -sum_rows exp(sink - LSE) * D.
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int H = 64;
constexpr int D = 512;
constexpr int B = 32;  // KV slots per block
constexpr int ROPE = 64;
constexpr int THREADS = 512;

struct Smem {
  __nv_bfloat16 q[H * D];
  __nv_bfloat16 dout[H * D];
  __nv_bfloat16 k[B * D];
  __nv_bfloat16 p[H * B];
  __nv_bfloat16 ds[H * B];
  float red[2][H][B + 8];  // the k-split halves' partial S / dP
  float dpart[2][H];
  float delta[H];
  float lse2[H];
  int idx[B];
};

// 1024-byte rows of 64 16-byte chunks, chunk XOR-swizzled by row so that the
// 8 rows an ldmatrix phase reads hit distinct bank groups.
__device__ __forceinline__ int wide(int r, int c) {
  return r * D + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7);
}
// 64-byte rows of 4 chunks.
__device__ __forceinline__ int narrow(int r, int c) {
  return r * B + ((((c >> 3) ^ ((r >> 1) & 3))) << 3) + (c & 7);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// S = Q K^T and dP = dO K^T for one KV block. Warp (mt, kh, nh) multiplies
// heads [16 mt, +16) by slots [16 nh, +16) over dims [256 kh, +256); the kh = 1
// half hands its partial sums to kh = 0 through shared memory (fixed order).
// Returns whether this warp holds reduced tiles: acc[t] covers slots
// [16 nh + 8 t, +8) in the m16n8 accumulator layout.
__device__ __forceinline__ bool score_tiles(Smem& s, int warp, int lane, float (&acc_s)[2][4],
                                            float (&acc_dp)[2][4]) {
  const int mt = warp & 3, kh = (warp >> 2) & 1, nh = warp >> 3;
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc_s[t][i] = acc_dp[t][i] = 0.f;
  const int arow = 16 * mt + (lane & 15);
  const int brow = 16 * nh + (lane & 7) + ((lane >> 4) << 3);
  const int bk = ((lane >> 3) & 1) * 8;
#pragma unroll 4
  for (int k = 256 * kh; k < 256 * kh + 256; k += 16) {
    uint32_t a[4], g[4], b[4];
    ldsm_x4(a, &s.q[wide(arow, k + (lane >> 4) * 8)]);
    ldsm_x4(g, &s.dout[wide(arow, k + (lane >> 4) * 8)]);
    ldsm_x4(b, &s.k[wide(brow, k + bk)]);
    mma(acc_s[0], a, b[0], b[1]);
    mma(acc_s[1], a, b[2], b[3]);
    mma(acc_dp[0], g, b[0], b[1]);
    mma(acc_dp[1], g, b[2], b[3]);
  }
  const int h0 = 16 * mt + (lane >> 2);
  if (kh == 1) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int c = 16 * nh + 8 * t + 2 * (lane & 3);
      *reinterpret_cast<float2*>(&s.red[0][h0][c]) = make_float2(acc_s[t][0], acc_s[t][1]);
      *reinterpret_cast<float2*>(&s.red[0][h0 + 8][c]) = make_float2(acc_s[t][2], acc_s[t][3]);
      *reinterpret_cast<float2*>(&s.red[1][h0][c]) = make_float2(acc_dp[t][0], acc_dp[t][1]);
      *reinterpret_cast<float2*>(&s.red[1][h0 + 8][c]) = make_float2(acc_dp[t][2], acc_dp[t][3]);
    }
  }
  __syncthreads();
  if (kh == 1) return false;
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int c = 16 * nh + 8 * t + 2 * (lane & 3);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = h0 + (i >> 1) * 8, slot = c + (i & 1);
      acc_s[t][i] += s.red[0][h][slot];
      acc_dp[t][i] += s.red[1][h][slot];
    }
  }
  return true;
}

__device__ __forceinline__ float2 rotate(float2 x, float c, float s) {
  // The mega forward kernel's apply_rope: fma({x, x}, {c, s}, {-y s, y c}).
  return make_float2(fmaf(x.x, c, -x.y * s), fmaf(x.x, s, x.y * c));
}

__global__ void __launch_bounds__(THREADS, 1)
    mega_attention_bwd_kernel(const __nv_bfloat16* __restrict__ q_fused,
                              const __nv_bfloat16* __restrict__ grad_fused,
                              const int* __restrict__ positions,
                              const float* __restrict__ cos_sin,
                              const __nv_bfloat16* __restrict__ kv, int64_t kv_rows,
                              const int* __restrict__ indices, int64_t index_stride,
                              const int* __restrict__ lens, const float* __restrict__ lse2,
                              float scale, float scale_log2, __nv_bfloat16* __restrict__ dq_fused,
                              float* __restrict__ dkv, float* __restrict__ delta_out,
                              int row0, int row1, float* __restrict__ scratch,
                              __nv_bfloat16* __restrict__ partial) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // This CTA's P / dP of one row, [2][H][index_stride] FP32: pass 1 writes
  // what pass 2 reads (same thread, same addresses), so pass 2 does not
  // recompute S and dP. Small enough per CTA to stay in L2.
  float* sp = scratch + (int64_t)blockIdx.x * 2 * H * index_stride;
  float* sdp = sp + (int64_t)H * index_stride;
  for (int row = row0 + blockIdx.x; row < row1; row += gridDim.x) {
  const int len = lens[row];
  const int nblocks = (len + B - 1) / B;
  const float* cs = cos_sin + (int64_t)positions[row] * ROPE;

  {
    // q: fused layout [d / 16][h][16]; grad: [group][d / 32][h % 8][32].
    const __nv_bfloat16* qrow = q_fused + (int64_t)row * H * D;
    const __nv_bfloat16* grow = grad_fused + (int64_t)row * H * D;
    for (int c = tid; c < H * D / 8; c += THREADS) {
      const int f = c * 8;
      cp_async16(&s.q[wide((f & 1023) >> 4, (f >> 10) * 16 + (f & 15))], qrow + f);
      const int rem = f & 4095;
      cp_async16(&s.dout[wide((f >> 12) * 8 + ((rem & 255) >> 5), (rem >> 8) * 32 + (rem & 31))],
                 grow + f);
    }
    if (tid < H) s.lse2[tid] = lse2[(int64_t)row * H + tid];
    cp_async_wait_all();
    __syncthreads();
    // Q RoPE as the forward kernel (FP32 FMA, RN to bf16); dO = R(theta) do,
    // the transpose of the output's conjugate RoPE.
    for (int p = tid; p < H * (ROPE / 2); p += THREADS) {
      const int h = p / (ROPE / 2), i = p % (ROPE / 2), d = D - ROPE + 2 * i;
      const float c = cs[i], sn = cs[ROPE / 2 + i];
      __nv_bfloat162* qp = reinterpret_cast<__nv_bfloat162*>(&s.q[wide(h, d)]);
      __nv_bfloat162* gp = reinterpret_cast<__nv_bfloat162*>(&s.dout[wide(h, d)]);
      const float2 qy = rotate(__bfloat1622float2(*qp), c, sn);
      const float2 gy = rotate(__bfloat1622float2(*gp), c, sn);
      *qp = __floats2bfloat162_rn(qy.x, qy.y);
      *gp = __floats2bfloat162_rn(gy.x, gy.y);
    }
    __syncthreads();
  }

  auto load_kv = [&](int block) {
    if (tid < B) {
      const int slot = block * B + tid;
      int id = slot < len ? indices[(int64_t)row * index_stride + slot] : -1;
      s.idx[tid] = (id >= 0 && id < kv_rows) ? id : -1;
    }
    __syncthreads();
    for (int c = tid; c < B * D / 8; c += THREADS) {
      const int r = c / (D / 8), col = (c % (D / 8)) * 8;
      const int id = s.idx[r];
      if (id >= 0) {
        cp_async16(&s.k[wide(r, col)], kv + (int64_t)id * D + col);
      } else {
        *reinterpret_cast<uint4*>(&s.k[wide(r, col)]) = make_uint4(0, 0, 0, 0);
      }
    }
    cp_async_wait_all();
    __syncthreads();
  };

  const int mt = warp & 3, nh = warp >> 3;
  const int h0 = 16 * mt + (lane >> 2), h1 = h0 + 8;
  const int c0 = 16 * nh + 2 * (lane & 3);  // + 8 t + (i & 1)

  // ---- pass 1: D = sum_j P_j dP_j ----
  float dsum0 = 0.f, dsum1 = 0.f;
  for (int block = 0; block < nblocks; ++block) {
    load_kv(block);
    float acc_s[2][4], acc_dp[2][4];
    if (score_tiles(s, warp, lane, acc_s, acc_dp)) {
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int h = i < 2 ? h0 : h1, slot = c0 + 8 * t + (i & 1);
          const float p =
              s.idx[slot] >= 0 ? exp2f(acc_s[t][i] * scale_log2 - s.lse2[h]) : 0.f;
          (i < 2 ? dsum0 : dsum1) += p * acc_dp[t][i];
          acc_s[t][i] = p;
        }
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int64_t c = block * B + c0 + 8 * t;
        *reinterpret_cast<float2*>(sp + h0 * index_stride + c) = make_float2(acc_s[t][0], acc_s[t][1]);
        *reinterpret_cast<float2*>(sp + h1 * index_stride + c) = make_float2(acc_s[t][2], acc_s[t][3]);
        *reinterpret_cast<float2*>(sdp + h0 * index_stride + c) = make_float2(acc_dp[t][0], acc_dp[t][1]);
        *reinterpret_cast<float2*>(sdp + h1 * index_stride + c) = make_float2(acc_dp[t][2], acc_dp[t][3]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    dsum0 += __shfl_xor_sync(0xffffffff, dsum0, off);
    dsum1 += __shfl_xor_sync(0xffffffff, dsum1, off);
  }
  if (((warp >> 2) & 1) == 0 && (lane & 3) == 0) {
    s.dpart[nh][h0] = dsum0;
    s.dpart[nh][h1] = dsum1;
  }
  __syncthreads();
  if (tid < H) {
    const float d = s.dpart[0][tid] + s.dpart[1][tid];
    s.delta[tid] = d;
    delta_out[(int64_t)row * H + tid] = d;
  }
  __syncthreads();

  // ---- pass 2 ----
  // dQ: this warp owns dims [32 warp, +32) of all 64 heads.
  float acc_q[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc_q[a][b][i] = 0.f;
  const int dcol = 32 * warp;

  for (int block = 0; block < nblocks; ++block) {
    load_kv(block);
    {
      if (((warp >> 2) & 1) == 0) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int64_t g = block * B + c0 + 8 * t;
          const float2 p0 = *reinterpret_cast<const float2*>(sp + h0 * index_stride + g);
          const float2 p1 = *reinterpret_cast<const float2*>(sp + h1 * index_stride + g);
          const float2 d0 = *reinterpret_cast<const float2*>(sdp + h0 * index_stride + g);
          const float2 d1 = *reinterpret_cast<const float2*>(sdp + h1 * index_stride + g);
          const float pv[4] = {p0.x, p0.y, p1.x, p1.y};
          const float dp[4] = {d0.x, d0.y, d1.x, d1.y};
          float dsv[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int h = i < 2 ? h0 : h1;
            dsv[i] = scale * pv[i] * (dp[i] - s.delta[h]);
          }
          const int c = c0 + 8 * t;
          *reinterpret_cast<uint32_t*>(&s.p[narrow(h0, c)]) = pack_bf16(pv[0], pv[1]);
          *reinterpret_cast<uint32_t*>(&s.p[narrow(h1, c)]) = pack_bf16(pv[2], pv[3]);
          *reinterpret_cast<uint32_t*>(&s.ds[narrow(h0, c)]) = pack_bf16(dsv[0], dsv[1]);
          *reinterpret_cast<uint32_t*>(&s.ds[narrow(h1, c)]) = pack_bf16(dsv[2], dsv[3]);
        }
      }
    }
    __syncthreads();

    // dQ[64 x 32 dims] += dS[64 x 32 slots] . K[32 slots x 32 dims]
#pragma unroll
    for (int ks = 0; ks < B; ks += 16) {
      uint32_t bfrag[4][2];
#pragma unroll
      for (int n2 = 0; n2 < 2; ++n2) {
        // x4.trans: matrices (k0-7, n), (k8-15, n), (k0-7, n+8), (k8-15, n+8)
        uint32_t r[4];
        const int krow = ks + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ncol = dcol + 16 * n2 + (lane >> 4) * 8;
        ldsm_x4_t(r, &s.k[wide(krow, ncol)]);
        bfrag[2 * n2][0] = r[0];
        bfrag[2 * n2][1] = r[1];
        bfrag[2 * n2 + 1][0] = r[2];
        bfrag[2 * n2 + 1][1] = r[3];
      }
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        uint32_t a[4];
        ldsm_x4(a, &s.ds[narrow(16 * m + (lane & 15), ks + (lane >> 4) * 8)]);
#pragma unroll
        for (int n = 0; n < 4; ++n) mma(acc_q[m][n], a, bfrag[n][0], bfrag[n][1]);
      }
    }

    // dK[32 slots x 32 dims] = dS^T . Q + P^T . dO over the 64 heads.
    float acc_k[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc_k[a][b][i] = 0.f;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      const __nv_bfloat16* left = part == 0 ? s.ds : s.p;
      const __nv_bfloat16* right = part == 0 ? s.q : s.dout;
#pragma unroll
      for (int ks = 0; ks < H; ks += 16) {
        uint32_t bfrag[4][2];
#pragma unroll
        for (int n2 = 0; n2 < 2; ++n2) {
          uint32_t r[4];
          const int krow = ks + (lane & 7) + ((lane >> 3) & 1) * 8;
          const int ncol = dcol + 16 * n2 + (lane >> 4) * 8;
          ldsm_x4_t(r, &right[wide(krow, ncol)]);
          bfrag[2 * n2][0] = r[0];
          bfrag[2 * n2][1] = r[1];
          bfrag[2 * n2 + 1][0] = r[2];
          bfrag[2 * n2 + 1][1] = r[3];
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // A[m = slot][k = head] from left[head][slot]: x4.trans of
          // (k0-7, m0-7), (k0-7, m8-15), (k8-15, m0-7), (k8-15, m8-15).
          uint32_t a[4];
          const int g = lane >> 3;
          const int hrow = ks + (lane & 7) + (g >> 1) * 8;
          const int scol = 16 * m + (g & 1) * 8;
          ldsm_x4_t(a, &left[narrow(hrow, scol)]);
#pragma unroll
          for (int n = 0; n < 4; ++n) mma(acc_k[m][n], a, bfrag[n][0], bfrag[n][1]);
        }
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int slot = 16 * m + (lane >> 2) + 8 * half;
        const int id = s.idx[slot];
        if (id < 0) continue;
        if (partial != nullptr) {
          // Deterministic mode: this row's slot partial, reduced later in a
          // fixed order by segment_reduce_kernel.
          __nv_bfloat16* dst = partial +
                               ((int64_t)(row - row0) * index_stride + block * B + slot) * D +
                               dcol + 2 * (lane & 3);
#pragma unroll
          for (int n = 0; n < 4; ++n)
            *reinterpret_cast<__nv_bfloat162*>(dst + 8 * n) =
                __floats2bfloat162_rn(acc_k[m][n][2 * half], acc_k[m][n][2 * half + 1]);
          continue;
        }
        float* dst = dkv + (int64_t)id * D + dcol + 2 * (lane & 3);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          atomicAdd(reinterpret_cast<float2*>(dst + 8 * n),
                    make_float2(acc_k[m][n][2 * half], acc_k[m][n][2 * half + 1]));
        }
      }
    }
    __syncthreads();
  }

  // dq of the unrotated query: R(theta)^T on the RoPE dims, fused layout.
  __nv_bfloat16* qrow = dq_fused + (int64_t)row * H * D;
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int h = 16 * m + (lane >> 2) + 8 * half;
        const int d = dcol + 8 * n + 2 * (lane & 3);
        float2 g = make_float2(acc_q[m][n][2 * half], acc_q[m][n][2 * half + 1]);
        if (d >= D - ROPE) {
          const int i = (d - (D - ROPE)) / 2;
          g = rotate(g, cs[i], -cs[ROPE / 2 + i]);
        }
        *reinterpret_cast<__nv_bfloat162*>(qrow + (d >> 4) * (H * 16) + h * 16 + (d & 15)) =
            __floats2bfloat162_rn(g.x, g.y);
      }
    __syncthreads();
  }
}

// dkv[r] += the chunk's partials of KV row r, summed in sorted (row-major)
// order. One 128-thread block per KV row, each warp a quarter of the dims; the
// row's pairs are [starts[r], starts[r + 1]) of the sorted order.
__global__ void segment_reduce_kernel(const int64_t* __restrict__ starts,
                                      const int64_t* __restrict__ order, int64_t flat0,
                                      const __nv_bfloat16* __restrict__ partial,
                                      float* __restrict__ dkv) {
  const int64_t r = blockIdx.x;
  const int64_t begin = starts[r], end = starts[r + 1];
  if (begin == end) return;
  const int lane = threadIdx.x & 31;
  const int dim = (threadIdx.x >> 5) * 128 + 4 * lane;
  constexpr int U = 8;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int64_t q = begin; q < end; q += U) {
    uint2 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u)
      if (q + u < end)
        v[u] = __ldcs(reinterpret_cast<const uint2*>(partial + (order[q + u] - flat0) * D + dim));
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (q + u < end) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[u]);
        const float2 f0 = __bfloat1622float2(h[0]), f1 = __bfloat1622float2(h[1]);
        acc[0] += f0.x; acc[1] += f0.y; acc[2] += f1.x; acc[3] += f1.y;
      }
    }
  }
  float4* dst = reinterpret_cast<float4*>(dkv + r * D + dim);
  float4 d = *dst;
  d.x += acc[0]; d.y += acc[1]; d.z += acc[2]; d.w += acc[3];
  *dst = d;
}

}  // namespace

// dkv += the fixed-order sums of chunk `chunk_index`'s bf16 partials.
void segment_reduce(const torch::Tensor& starts, const torch::Tensor& order, int64_t chunk_index,
                    int64_t flat0, const torch::Tensor& partial, torch::Tensor& dkv) {
  segment_reduce_kernel<<<dkv.size(0), 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
      starts.data_ptr<int64_t>() + chunk_index * dkv.size(0), order.data_ptr<int64_t>(), flat0,
      reinterpret_cast<const __nv_bfloat16*>(partial.data_ptr()), dkv.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward(torch::Tensor q, torch::Tensor grad, torch::Tensor positions,
              torch::Tensor cos_sin, torch::Tensor kv, torch::Tensor indices, torch::Tensor lens,
              torch::Tensor lse2, double scale, torch::Tensor dq, torch::Tensor dkv,
              torch::Tensor delta, torch::Tensor starts, torch::Tensor order,
              int64_t chunk_rows) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(q.dim() == 3 && q.size(1) == H && q.size(2) == D);
  const int64_t rows = q.size(0);
  TORCH_CHECK(grad.numel() == rows * H * D && grad.scalar_type() == at::kBFloat16 &&
              grad.is_contiguous());
  TORCH_CHECK(positions.numel() == rows && positions.scalar_type() == at::kInt &&
              positions.is_contiguous());
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == ROPE &&
              cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous());
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == D && kv.scalar_type() == at::kBFloat16 &&
              kv.is_contiguous());
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == rows &&
              indices.scalar_type() == at::kInt && indices.stride(1) == 1);
  TORCH_CHECK(lens.numel() == rows && lens.scalar_type() == at::kInt && lens.is_contiguous());
  TORCH_CHECK(lse2.sizes() == at::IntArrayRef({rows, H}) && lse2.scalar_type() == at::kFloat &&
              lse2.is_contiguous());
  TORCH_CHECK(dq.sizes() == q.sizes() && dq.scalar_type() == at::kBFloat16 && dq.is_contiguous());
  TORCH_CHECK(dkv.sizes() == kv.sizes() && dkv.scalar_type() == at::kFloat &&
              dkv.is_contiguous());
  TORCH_CHECK(delta.sizes() == lse2.sizes() && delta.scalar_type() == at::kFloat &&
              delta.is_contiguous());
  if (rows == 0) return;
  const size_t smem = sizeof(Smem);
  cudaFuncSetAttribute(mega_attention_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem);
  const float scale_f = static_cast<float>(scale);
  const float scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  int device = 0, sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  const int64_t grid = std::min<int64_t>(rows, sms);
  TORCH_CHECK(indices.stride(0) % 2 == 0 && indices.size(1) % B == 0);
  auto scratch = torch::empty({grid, 2, H, indices.stride(0)}, dkv.options());
  const bool deterministic = starts.numel() > 0;
  const int64_t chunk = deterministic ? chunk_rows : rows;
  torch::Tensor partial;
  if (deterministic) {
    TORCH_CHECK(starts.scalar_type() == at::kLong && order.scalar_type() == at::kLong &&
                starts.numel() == (rows + chunk_rows - 1) / chunk_rows * kv.size(0) + 1);
    partial = torch::empty({chunk * indices.stride(0), D}, dkv.options().dtype(at::kBFloat16));
  }
  auto stream = c10::cuda::getCurrentCUDAStream();
  for (int64_t row0 = 0, c = 0; row0 < rows; row0 += chunk, ++c) {
    const int64_t row1 = std::min(rows, row0 + chunk);
    mega_attention_bwd_kernel<<<std::min<int64_t>(grid, row1 - row0), THREADS, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), positions.data_ptr<int>(),
        cos_sin.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
        kv.size(0), indices.data_ptr<int>(), indices.stride(0), lens.data_ptr<int>(),
        lse2.data_ptr<float>(), scale_f, scale_log2,
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), dkv.data_ptr<float>(),
        delta.data_ptr<float>(), (int)row0, (int)row1, scratch.data_ptr<float>(),
        deterministic ? reinterpret_cast<__nv_bfloat16*>(partial.data_ptr()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (deterministic) segment_reduce(starts, order, c, row0 * indices.stride(0), partial, dkv);
  }
}

void backward_sm100(torch::Tensor q, torch::Tensor grad, torch::Tensor positions,
                    torch::Tensor cos_sin, torch::Tensor kv, torch::Tensor indices,
                    torch::Tensor lens, torch::Tensor lse2, double scale, torch::Tensor dq,
                    torch::Tensor dkv, torch::Tensor delta, torch::Tensor starts,
                    torch::Tensor order, int64_t chunk_rows);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &backward);
  m.def("backward_sm100", &backward_sm100);
}
