// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Raw PTX helpers for sm_100a: mbarriers, TMA gather4, tensor memory and
// tcgen05.mma (kind::f16) with 128B-swizzled shared-memory operands.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm100 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- mbarrier ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

// ---- proxies and named barriers ----
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void named_arrive(int id, int threads) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// ---- TMA ----
// Four rows (r0..r3) of a 2D tensor map with box {64, 1} and 128B swizzle:
// 512 bytes land at dst as four consecutive 128-byte rows. Negative or
// out-of-range rows are zero-filled and still count toward the transaction.
__device__ __forceinline__ void tma_gather4(const CUtensorMap* map, uint64_t* bar, void* dst,
                                            int col, int4 rows) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z),
      "r"(rows.w), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_store(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst),
               "r"(smem_u32(src)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit_wait_read() {
  asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void bulk_commit_wait() {
  asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group 0;" ::: "memory");
}

// ---- tensor memory ----
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                   smem_u32(dst)),
               "r"(cols));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols));
}
__device__ __forceinline__ void tc_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
// 32 consecutive FP32 columns of this thread's lane (the warp's 32 lanes).
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float (&v)[32]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float (&v)[16]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
}

// ---- UMMA ----
// Shared-memory descriptor, 128B swizzle (atoms of 8 rows x 128 bytes).
// K-major: sbo = byte stride between 8-row groups, lbo unused.
// MN-major: sbo = byte stride between 8-row K groups, lbo = byte stride
// between 64-element MN atoms.
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return static_cast<uint64_t>((addr >> 4) & 0x3FFF) |
         (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (2ull << 61);
}
// BF16 x BF16 -> FP32 instruction descriptor.
constexpr uint32_t idesc(int m, int n, bool a_mn, bool b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | (static_cast<uint32_t>(a_mn) << 15) |
         (static_cast<uint32_t>(b_mn) << 16) | (static_cast<uint32_t>(n >> 3) << 17) |
         (static_cast<uint32_t>(m >> 4) << 24);
}
__device__ __forceinline__ void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t id,
                                     uint32_t accumulate) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(id), "r"(accumulate));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
          smem_u32(bar))
      : "memory");
}

// Element offset in a [rows][64] bf16 slab with the 128B swizzle
// (16-byte chunk index XOR row % 8), slab base 1024-byte aligned.
__device__ __forceinline__ int swz(int r, int c) {
  return r * 64 + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7);
}

}  // namespace sm100
