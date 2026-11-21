// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// sm_100a version of the mega attention backward (same contract as
// mega_attention_bwd.cu): tcgen05 UMMA with tensor-memory accumulators and
// TMA gather4 KV loads. KV slots are the MMA M dimension (blocks of 128), so
// every GEMM is M = 128 with operands in their natural layouts:
//   S^T, dP^T [slot x head]  = K . Q^T, K . dO^T           (pass 1 and pass 2)
//   dK        [slot x dim]   = dS^T . Q + P^T . dO          (pass 2)
//   dQ^T      [dim x head]  += K^T . dS^T                   (pass 2)
// One persistent CTA per SM walks query rows; per row, pass 1 forms
// D = sum_j P_j dP_j and pass 2 forms dS = scale * P (dP - D).
// Warp roles: warp 0 issues gather4 loads into a 2-stage ring of
// [128 slots x 128 dims] tiles, warp 1 issues UMMA, warps 4-7 (one KV slot per
// thread) turn S / dP into P / dS, warps 8-11 drain dK from tensor memory into
// the FP32 dKV buffer (atomics) or into per-row partials (deterministic mode).
// Warps 4-11 also load Q / dO (with RoPE) and store dq (inverse RoPE).
#include <cudaTypedefs.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "sm100_ptx.cuh"

namespace {

using namespace sm100;

constexpr int H = 64;
constexpr int D = 512;
constexpr int BK = 128;     // KV slots per block (UMMA M)
constexpr int TILE = 128;   // dims per ring stage
constexpr int NS = 2;       // ring stages
constexpr int ROPE = 64;
constexpr int THREADS = 512;
constexpr uint32_t STAGE_BYTES = BK * TILE * 2;

// Tensor memory columns.
constexpr uint32_t T_DQ = 0;     // 4 tiles of [128 dims x 64 heads]
// Two 128-column buffers at T_BUF + 128 * (u & 1), used in one sequence u:
// pass 1 block -> S^T (64 columns) | dP^T (64), pass 2 chunk -> dK
// [128 slots x 128 dims]. Use u waits on tfree[u & 1] with parity
// ((u >> 1) & 1) ^ 1 and is published on tfull[u & 1].
constexpr uint32_t T_BUF = 256;

constexpr uint32_t ID_S = idesc(128, 64, false, false);
constexpr uint32_t ID_DK = idesc(128, 128, false, true);
constexpr uint32_t ID_DQ = idesc(128, 64, true, true);

struct Smem {
  __nv_bfloat16 q[H * D];      // [8 slabs][64 heads][64], also the dq staging
  __nv_bfloat16 dout[H * D];
  __nv_bfloat16 k[NS][BK * TILE];  // [2 slabs][128 slots][64]
  __nv_bfloat16 pt[BK * H];    // P^T  [slot][head]
  __nv_bfloat16 dst[BK * H];   // dS^T [slot][head]
  float lse2[H];
  float delta[H];
  float red[4][H];
  uint64_t kfull[NS], kempty[NS];
  uint64_t tfull[2], tfree[2], pds_full, pds_free;
  uint64_t q_ready, row_done, dq_free;
  uint32_t tmem;
};

__device__ __forceinline__ float2 rotate(float2 x, float c, float s) {
  return make_float2(fmaf(x.x, c, -x.y * s), fmaf(x.x, s, x.y * c));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void red_add_v4(float* dst, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "f"(a), "f"(b),
               "f"(c), "f"(d)
               : "memory");
}

__device__ __forceinline__ int slot_id(const int* __restrict__ indices, int64_t index_stride,
                                       int row, int slot, int len, int64_t kv_rows) {
  if (slot >= len) return -1;
  const int id = indices[(int64_t)row * index_stride + slot];
  return (id >= 0 && id < kv_rows) ? id : -1;
}

__global__ void __launch_bounds__(THREADS, 1)
    mega_attention_bwd_sm100_kernel(
        const __grid_constant__ CUtensorMap kv_map, const __nv_bfloat16* __restrict__ q_fused,
        const __nv_bfloat16* __restrict__ grad_fused, const int* __restrict__ positions,
        const float* __restrict__ cos_sin, int64_t kv_rows, const int* __restrict__ indices,
        int64_t index_stride, const int* __restrict__ lens, const float* __restrict__ lse2,
        float scale, float scale_log2, __nv_bfloat16* __restrict__ dq_fused,
        float* __restrict__ dkv, float* __restrict__ delta_out, int row0, int row1,
        float* __restrict__ scratch, int scratch_slots, __nv_bfloat16* __restrict__ partial) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    if (smem_u32(smem_raw) & 1023) __trap();
    for (int i = 0; i < NS; ++i) {
      mbar_init(&s.kfull[i], 1);
      mbar_init(&s.kempty[i], 1);
    }
    for (int i = 0; i < 2; ++i) {
      mbar_init(&s.tfull[i], 1);
      mbar_init(&s.tfree[i], 4);
    }
    mbar_init(&s.pds_full, 4);
    mbar_init(&s.pds_free, 1);
    mbar_init(&s.q_ready, 1);
    mbar_init(&s.row_done, 1);
    mbar_init(&s.dq_free, 1);
    fence_mbar_init();
  }
  if (warp == 2) tmem_alloc(&s.tmem, 512);
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = s.tmem;

  const uint32_t q_addr = smem_u32(s.q), do_addr = smem_u32(s.dout);
  const uint32_t pt_addr = smem_u32(s.pt), ds_addr = smem_u32(s.dst);

  // Warpgroups 0 and 3 (MMA, gather4 issue) need few registers; 1 and 2
  // (softmax, dK drain) get the rest.
  if (warp < 4 || warp >= 12) {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
  if (warp != 1) {
    // ---- gather4 producers ----
    // TMA issue throughput scales with issuing warps, not lanes: lanes 0-4
    // of warps 0, 2, 3, 12-15 each own one group of 4 slots.
    constexpr int LANES = 5, ISSUERS = 7 * LANES, GROUPS = BK / 4;
    constexpr int PER = (GROUPS + ISSUERS - 1) / ISSUERS;
    const int pw = warp == 0 ? 0 : (warp < 4 ? warp - 1 : warp - 9);
    const int issuer = pw * LANES + lane;
    if (lane < LANES) {
      int ks = 0;
      uint32_t kphase = 0;
      for (int row = row0 + blockIdx.x; row < row1; row += gridDim.x) {
        const int next = row + gridDim.x;
        if (issuer == 0 && next < row1) {
          prefetch_l2(q_fused + (int64_t)next * H * D, H * D * 2);
          prefetch_l2(grad_fused + (int64_t)next * H * D, H * D * 2);
        }
        const int len = lens[row];
        const int nb = max(1, (len + BK - 1) / BK);
        for (int pass = 0; pass < 2; ++pass) {
          for (int b = 0; b < nb; ++b) {
            int4 rows[PER];
#pragma unroll
            for (int k = 0; k < PER; ++k) {
              const int slot = b * BK + 4 * (issuer + ISSUERS * k);
              rows[k] = make_int4(slot_id(indices, index_stride, row, slot, len, kv_rows),
                                  slot_id(indices, index_stride, row, slot + 1, len, kv_rows),
                                  slot_id(indices, index_stride, row, slot + 2, len, kv_rows),
                                  slot_id(indices, index_stride, row, slot + 3, len, kv_rows));
            }
            for (int st = 0; st < D / TILE; ++st) {
              mbar_wait(&s.kempty[ks], kphase ^ 1);
              if (issuer == 0) mbar_expect_tx(&s.kfull[ks], STAGE_BYTES);
#pragma unroll
              for (int k = 0; k < PER; ++k) {
                const int g = issuer + ISSUERS * k;
                if (g < GROUPS) {
#pragma unroll
                  for (int slab = 0; slab < 2; ++slab)
                    tma_gather4(&kv_map, &s.kfull[ks], &s.k[ks][slab * BK * 64 + 4 * g * 64],
                                st * TILE + slab * 64, rows[k]);
                }
              }
              if (++ks == NS) {
                ks = 0;
                kphase ^= 1;
              }
            }
          }
        }
      }
    }
  } else {
    // ---- UMMA issuer ----
    if (lane == 0) {
      int ks = 0;
      uint32_t kphase = 0, u = 0, n_pds = 0, n_row = 0;
      auto buffer = [&]() {
        mbar_wait(&s.tfree[u & 1], ((u >> 1) & 1) ^ 1);
        tc_after_sync();
        return tmem + T_BUF + 128 * (u & 1);
      };
      auto publish = [&]() {
        umma_commit(&s.tfull[u & 1]);
        ++u;
      };
      auto k_full = [&]() {
        mbar_wait(&s.kfull[ks], kphase);
        tc_after_sync();
      };
      auto k_release = [&]() {
        umma_commit(&s.kempty[ks]);
        if (++ks == NS) {
          ks = 0;
          kphase ^= 1;
        }
      };
      auto scores = [&]() {
        const uint32_t tb = buffer();
        for (int st = 0; st < D / TILE; ++st) {
          k_full();
          const uint32_t ka = smem_u32(s.k[ks]);
#pragma unroll
          for (int k = 0; k < TILE / 16; ++k) {
            const uint32_t slab = k >> 2, off = (k & 3) * 32;
            const uint64_t a = sdesc(ka + slab * BK * 128 + off, 0, 1024);
            const uint32_t qo = (2 * st + slab) * H * 128 + off;
            const uint32_t acc = (st | k) != 0;
            umma(tb, a, sdesc(q_addr + qo, 0, 1024), ID_S, acc);
            umma(tb + 64, a, sdesc(do_addr + qo, 0, 1024), ID_S, acc);
          }
          k_release();
        }
        publish();
      };
      for (int row = row0 + blockIdx.x; row < row1; row += gridDim.x) {
        const int nb = max(1, (lens[row] + BK - 1) / BK);
        mbar_wait(&s.q_ready, n_row & 1);
        tc_after_sync();
        for (int b = 0; b < nb; ++b) scores();
        for (int b = 0; b < nb; ++b) {
          mbar_wait(&s.pds_full, n_pds & 1);
          if (b == 0) mbar_wait(&s.dq_free, (n_row & 1) ^ 1);
          tc_after_sync();
          for (int c = 0; c < D / TILE; ++c) {
            const uint32_t tb = buffer();
#pragma unroll
            for (int k = 0; k < H / 16; ++k) {
              const uint32_t bo = 2 * c * H * 128 + k * 2048;
              umma(tb, sdesc(ds_addr + k * 32, 0, 1024),
                   sdesc(q_addr + bo, H * 128, 1024), ID_DK, k > 0);
            }
#pragma unroll
            for (int k = 0; k < H / 16; ++k) {
              const uint32_t bo = 2 * c * H * 128 + k * 2048;
              umma(tb, sdesc(pt_addr + k * 32, 0, 1024),
                   sdesc(do_addr + bo, H * 128, 1024), ID_DK, 1);
            }
            publish();
            k_full();
            const uint32_t ka = smem_u32(s.k[ks]);
#pragma unroll
            for (int k = 0; k < BK / 16; ++k)
              umma(tmem + T_DQ + 64 * c, sdesc(ka + k * 2048, BK * 128, 1024),
                   sdesc(ds_addr + k * 2048, BK * 128, 1024), ID_DQ, (b | k) != 0);
            k_release();
          }
          umma_commit(&s.pds_free);
          ++n_pds;
        }
        umma_commit(&s.row_done);
        ++n_row;
      }
    }
  }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
    // ---- warps 4-11 ----
    const int t2 = tid - 128;       // 0..255
    const int t = t2 & 127;         // TMEM lane = KV slot / dim within a tile
    const bool softmax = warp < 8;
    const uint32_t lane_base = tmem + ((32 * (warp & 3)) << 16);
    // This CTA's P | dP of one row, [block][float4 of P | dP][slot] so a
    // warp's accesses are contiguous; pass 1 writes, pass 2 reads (same thread).
    float* my_scratch = scratch + (int64_t)blockIdx.x * scratch_slots * 2 * H;
    uint32_t u = 0, n_pds = 0, n_row = 0;
    for (int row = row0 + blockIdx.x; row < row1; row += gridDim.x) {
      const int len = lens[row];
      const int nb = max(1, (len + BK - 1) / BK);
      const float* cs = cos_sin + (int64_t)positions[row] * ROPE;

      // Q and dO into the swizzled slabs, RoPE as the forward kernel. The
      // softmax warps own the row boundary (this, and dq below) so the drain
      // warps' last reductions overlap it.
      if (softmax) {
        const __nv_bfloat16* qrow = q_fused + (int64_t)row * H * D;
        const __nv_bfloat16* grow = grad_fused + (int64_t)row * H * D;
#pragma unroll 1
        for (int c0 = t; c0 < H * D / 8; c0 += 8 * 128) {
          uint4 qbuf[8], gbuf[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            qbuf[u] = __ldcs(reinterpret_cast<const uint4*>(qrow) + c0 + 128 * u);
            gbuf[u] = __ldcs(reinterpret_cast<const uint4*>(grow) + c0 + 128 * u);
          }
#pragma unroll
          for (int u = 0; u < 8; ++u) {
          const int f = (c0 + 128 * u) * 8;
          const int qh = (f & 1023) >> 4, qd = (f >> 10) * 16 + (f & 15);
          const int rem = f & 4095;
          const int gh = (f >> 12) * 8 + ((rem & 255) >> 5), gd = (rem >> 8) * 32 + (rem & 31);
          uint4 qv = qbuf[u];
          uint4 gv = gbuf[u];
          if (qd >= D - ROPE) {
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&qv);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int r = (qd - (D - ROPE)) / 2 + i;
              p[i] = __float22bfloat162_rn(rotate(__bfloat1622float2(p[i]), cs[r], cs[ROPE / 2 + r]));
            }
          }
          if (gd >= D - ROPE) {
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&gv);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int r = (gd - (D - ROPE)) / 2 + i;
              p[i] = __float22bfloat162_rn(rotate(__bfloat1622float2(p[i]), cs[r], cs[ROPE / 2 + r]));
            }
          }
          *reinterpret_cast<uint4*>(&s.q[(qd >> 6) * H * 64 + swz(qh, qd & 63)]) = qv;
          *reinterpret_cast<uint4*>(&s.dout[(gd >> 6) * H * 64 + swz(gh, gd & 63)]) = gv;
          }
        }
        if (t < H) s.lse2[t] = lse2[(int64_t)row * H + t];
        fence_async_smem();
        named_sync(2, 128);
        if (t == 0) mbar_arrive(&s.q_ready);
      }

      if (softmax) {
        // ---- pass 1: D ----
        float dacc[H];
#pragma unroll
        for (int h = 0; h < H; ++h) dacc[h] = 0.f;
        for (int b = 0; b < nb; ++b) {
          // Invalid slots read zero-filled K rows (finite S, dP); masked here.
          const float vmask =
              slot_id(indices, index_stride, row, b * BK + t, len, kv_rows) >= 0 ? 1.f : 0.f;
          float4* sc = reinterpret_cast<float4*>(my_scratch) + (int64_t)b * BK * (H / 2) + t;
          const uint32_t tb = lane_base + T_BUF + 128 * (u & 1);
          mbar_wait(&s.tfull[u & 1], (u >> 1) & 1);
          tc_after_sync();
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            float sv[32], dp[32];
            tmem_ld32(tb + 32 * half, sv);
            tmem_ld32(tb + 64 + 32 * half, dp);
            tmem_wait_ld();
            if (half == 1) {
              tc_before_sync();
              __syncwarp();
              if (lane == 0) mbar_arrive(&s.tfree[u & 1]);
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) {
              const int h = 32 * half + i;
              sv[i] = ex2(fmaf(sv[i], scale_log2, -s.lse2[h])) * vmask;
              dacc[h] = fmaf(sv[i], dp[i], dacc[h]);
            }
            // [P (64) | dP (64)] of this slot, FP32.
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
              sc[(32 * half + i) / 4 * BK] = make_float4(sv[i], sv[i + 1], sv[i + 2], sv[i + 3]);
              sc[(H + 32 * half + i) / 4 * BK] = make_float4(dp[i], dp[i + 1], dp[i + 2], dp[i + 3]);
            }
          }
          ++u;
        }
        u += 4 * nb;  // pass 2 chunks belong to the drain warps
        // This row's pass 1 buffers are published: the drain may now wait on
        // its uses without running two phases ahead of a tfull barrier.
        named_arrive(3, 256);
#pragma unroll
        for (int h = 0; h < H; ++h) {
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) dacc[h] += __shfl_xor_sync(0xffffffff, dacc[h], off);
        }
        if (lane == 0) {
#pragma unroll
          for (int h = 0; h < H; ++h) s.red[warp - 4][h] = dacc[h];
        }
        named_sync(2, 128);
        if (t < H) {
          const float d = (s.red[0][t] + s.red[1][t]) + (s.red[2][t] + s.red[3][t]);
          s.delta[t] = d;
          delta_out[(int64_t)row * H + t] = d;
        }
        named_sync(2, 128);

        // ---- pass 2: P^T and dS^T rows for the UMMA ----
        for (int b = 0; b < nb; ++b) {
          // P / dS packed to bf16 before waiting for the buffers.
          const float4* sc = reinterpret_cast<const float4*>(my_scratch) + (int64_t)b * BK * (H / 2) + t;
          uint32_t pp[H / 2], dd[H / 2];
#pragma unroll
          for (int i = 0; i < H; i += 4) {
            const float4 p = sc[i / 4 * BK], dp = sc[(H + i) / 4 * BK];
            pp[i / 2] = pack_bf16(p.x, p.y);
            pp[i / 2 + 1] = pack_bf16(p.z, p.w);
            dd[i / 2] = pack_bf16(scale * p.x * (dp.x - s.delta[i]),
                                  scale * p.y * (dp.y - s.delta[i + 1]));
            dd[i / 2 + 1] = pack_bf16(scale * p.z * (dp.z - s.delta[i + 2]),
                                      scale * p.w * (dp.w - s.delta[i + 3]));
          }
          mbar_wait(&s.pds_free, (n_pds & 1) ^ 1);
#pragma unroll
          for (int c = 0; c < H / 8; ++c) {
            const int off = swz(t, 8 * c);
            *reinterpret_cast<uint4*>(&s.pt[off]) =
                make_uint4(pp[4 * c], pp[4 * c + 1], pp[4 * c + 2], pp[4 * c + 3]);
            *reinterpret_cast<uint4*>(&s.dst[off]) =
                make_uint4(dd[4 * c], dd[4 * c + 1], dd[4 * c + 2], dd[4 * c + 3]);
          }
          fence_async_smem();
          __syncwarp();
          if (lane == 0) mbar_arrive(&s.pds_full);
          ++n_pds;
        }
      } else {
        // ---- dK drain ----
        u += nb;  // pass 1 blocks belong to the softmax warps
        named_sync(3, 256);
        for (int b = 0; b < nb; ++b) {
          const int slot = b * BK + t;
          const int id = slot_id(indices, index_stride, row, slot, len, kv_rows);
          for (int c = 0; c < D / TILE; ++c) {
            const uint32_t tb = lane_base + T_BUF + 128 * (u & 1);
            mbar_wait(&s.tfull[u & 1], (u >> 1) & 1);
            tc_after_sync();
            float v[4][32];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmem_ld32(tb + 32 * j, v[j]);
            tmem_wait_ld();
            tc_before_sync();
            __syncwarp();
            if (lane == 0) mbar_arrive(&s.tfree[u & 1]);
            ++u;
            // Within each 8-lane group, transpose the 8 x 8 float4 blocks of
            // every 32-column piece: afterwards lane 8g + e holds, at block r,
            // float4 e of row 8g + r, so one instruction covers 128
            // contiguous bytes of 4 rows.
#pragma unroll
            for (int sh = 4; sh > 0; sh >>= 1) {
              const bool upper = (lane & sh) != 0;
#pragma unroll
              for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                  if (e & sh) continue;
#pragma unroll
                  for (int k = 0; k < 4; ++k) {
                    float& lo = v[j][4 * e + k];
                    float& hi = v[j][4 * (e + sh) + k];
                    const float got = __shfl_xor_sync(0xffffffff, upper ? lo : hi, sh);
                    if (upper) lo = got; else hi = got;
                  }
                }
            }
            const int e = lane & 7;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
              const int src = (lane & ~7) | r;
              const int rid = __shfl_sync(0xffffffff, id, src);
              if (rid < 0) continue;
              if (partial != nullptr) {
                const int rslot = b * BK + (warp & 3) * 32 + src;
                uint2* dst = reinterpret_cast<uint2*>(
                    partial + ((int64_t)(row - row0) * index_stride + rslot) * D + c * TILE);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                  dst[(32 * j + 4 * e) / 4] = make_uint2(pack_bf16(v[j][4 * r], v[j][4 * r + 1]),
                                                         pack_bf16(v[j][4 * r + 2], v[j][4 * r + 3]));
              } else {
                float* dst = dkv + (int64_t)rid * D + c * TILE + 4 * e;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                  red_add_v4(dst + 32 * j, v[j][4 * r], v[j][4 * r + 1], v[j][4 * r + 2],
                             v[j][4 * r + 3]);
              }
            }
          }
        }
      }

      // ---- dq: inverse RoPE, fused layout, staged in s.q ----
      if (softmax) {
        mbar_wait(&s.row_done, n_row & 1);
        tc_after_sync();
        ++n_row;
        __nv_bfloat16* stage = s.q;
#pragma unroll 1
        for (int tt = 0; tt < D / TILE; ++tt) {
          const int d = tt * TILE + t;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            float v[32];
            tmem_ld32(lane_base + T_DQ + 64 * tt + 32 * half, v);
            tmem_wait_ld();
            if (d >= D - ROPE) {  // warp-uniform
              const int r = (d - (D - ROPE)) / 2;
              const float c = cs[r], sn = cs[ROPE / 2 + r];
              const bool even = (d & 1) == 0;
#pragma unroll
              for (int i = 0; i < 32; ++i) {
                const float other = __shfl_xor_sync(0xffffffff, v[i], 1);
                v[i] = even ? fmaf(v[i], c, other * sn) : fmaf(other, -sn, v[i] * c);
              }
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) {
              const int h = 32 * half + i;
              stage[(d >> 4) * (H * 16) + h * 16 + (d & 15)] = __float2bfloat16_rn(v[i]);
            }
          }
        }
        tc_before_sync();
        fence_async_smem();
        named_sync(2, 128);
        if (t == 0) {
          mbar_arrive(&s.dq_free);
          bulk_store(dq_fused + (int64_t)row * H * D, stage, H * D * 2);
          bulk_commit_wait_read();
        }
        named_sync(2, 128);
      }
    }
    if (t2 == 0) bulk_commit_wait();
  }
  tc_before_sync();
  __syncthreads();
  if (warp == 2) {
    tc_after_sync();
    tmem_dealloc(tmem, 512);
  }
}

PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    cudaDriverEntryPointQueryResult status;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                                                 reinterpret_cast<void**>(&fn), 12000,
                                                 cudaEnableDefault, &status) == cudaSuccess &&
                    status == cudaDriverEntryPointSuccess,
                "cuTensorMapEncodeTiled unavailable");
  }
  return fn;
}

}  // namespace

void segment_reduce(const torch::Tensor& starts, const torch::Tensor& order, int64_t chunk_index,
                    int64_t flat0, const torch::Tensor& partial, torch::Tensor& dkv);

// Same arguments and outputs as backward(); see mega_attention_bwd.cu.
void backward_sm100(torch::Tensor q, torch::Tensor grad, torch::Tensor positions,
                    torch::Tensor cos_sin, torch::Tensor kv, torch::Tensor indices,
                    torch::Tensor lens, torch::Tensor lse2, double scale, torch::Tensor dq,
                    torch::Tensor dkv, torch::Tensor delta, torch::Tensor starts,
                    torch::Tensor order, int64_t chunk_rows) {
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
              kv.is_contiguous() && kv.size(0) < (int64_t(1) << 31));
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

  CUtensorMap kv_map;
  uint64_t size[2] = {D, static_cast<uint64_t>(kv.size(0))};
  uint64_t stride[1] = {D * 2};
  uint32_t box[2] = {64, 1}, elem_stride[2] = {1, 1};
  TORCH_CHECK(encode_fn()(&kv_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kv.data_ptr(), size,
                          stride, box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);

  const int smem = sizeof(Smem);
  cudaFuncSetAttribute(mega_attention_bwd_sm100_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  const float scale_f = static_cast<float>(scale);
  const float scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  int device = 0, sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  const bool deterministic = starts.numel() > 0;
  const int64_t chunk = deterministic ? chunk_rows : rows;
  torch::Tensor partial;
  if (deterministic) {
    TORCH_CHECK(starts.scalar_type() == at::kLong && order.scalar_type() == at::kLong &&
                starts.numel() == (rows + chunk_rows - 1) / chunk_rows * kv.size(0) + 1);
    partial = torch::empty({chunk * indices.stride(0), D}, dkv.options().dtype(at::kBFloat16));
  }
  const int64_t padded = std::max<int64_t>(1, (indices.size(1) + BK - 1) / BK) * BK;
  auto scratch = torch::empty({std::min<int64_t>(sms, rows), padded, 2 * H}, dkv.options());
  auto stream = c10::cuda::getCurrentCUDAStream();
  for (int64_t row0 = 0, c = 0; row0 < rows; row0 += chunk, ++c) {
    const int64_t row1 = std::min(rows, row0 + chunk);
    mega_attention_bwd_sm100_kernel<<<std::min<int64_t>(sms, row1 - row0), THREADS, smem,
                                      stream>>>(
        kv_map, reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), positions.data_ptr<int>(),
        cos_sin.data_ptr<float>(), kv.size(0), indices.data_ptr<int>(), indices.stride(0),
        lens.data_ptr<int>(), lse2.data_ptr<float>(), scale_f, scale_log2,
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), dkv.data_ptr<float>(),
        delta.data_ptr<float>(), (int)row0, (int)row1, scratch.data_ptr<float>(), (int)padded,
        deterministic ? reinterpret_cast<__nv_bfloat16*>(partial.data_ptr()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (deterministic) {
      segment_reduce(starts, order, c, row0 * indices.stride(0), partial, dkv);
    }
  }
}
