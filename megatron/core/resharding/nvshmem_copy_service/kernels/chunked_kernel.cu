
#include <cuda_runtime.h>

// CUDA-compatible types (no C++ standard library headers for NVRTC)
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef uint64_t uintptr_t;

// ============================================================================
// Kernel Configuration Constants (from ChunkedKernel.h)
// ============================================================================

constexpr int CHUNK_SIZE = 128 * 1024;       // 128KB per chunk
constexpr int NUM_BLOCKS = 75;               // Fixed grid size
constexpr int THREADS_PER_BLOCK = 1024;      // Fixed block size
constexpr int FLOAT4_SIZE = 16;              // 16 bytes per float4
constexpr int MAX_CHUNKS_PER_BLOCK = 512;    // Max chunks per block for shared memory

extern "C" {

/**
 * Chunked batched copy kernel implementation
 *
 * This kernel performs efficient batched memory copies using:
 * 1. Contiguous block assignment for better load balancing
 * 2. Shared memory prefetching of chunk metadata
 * 3. Vectorized float4 (16-byte) copies for aligned data
 * 4. Byte-by-byte fallback for unaligned or small data
 */
__global__ void chunked_batched_copy_kernel(
    uint8_t** src_addrs,
    uint8_t** dst_addrs,
    size_t* sizes,
    int total_chunks
) {
    // Shared memory for metadata prefetching
    __shared__ uint8_t* s_src_addrs[MAX_CHUNKS_PER_BLOCK];
    __shared__ uint8_t* s_dst_addrs[MAX_CHUNKS_PER_BLOCK];
    __shared__ size_t s_sizes[MAX_CHUNKS_PER_BLOCK];

    // Contiguous block assignment: block i processes chunks [start_chunk, end_chunk)
    int chunks_per_block = (total_chunks + gridDim.x - 1) / gridDim.x;  // Ceiling division
    int start_chunk = blockIdx.x * chunks_per_block;
    int end_chunk = start_chunk + chunks_per_block;
    if (end_chunk > total_chunks) {
        end_chunk = total_chunks;
    }
    int num_chunks_this_block = end_chunk - start_chunk;

    // Phase 1: Cooperative loading of metadata to shared memory
    // All 1024 threads cooperate to load metadata from global memory
    for (int i = threadIdx.x; i < num_chunks_this_block; i += blockDim.x) {
        int global_chunk_id = start_chunk + i;
        s_src_addrs[i] = src_addrs[global_chunk_id];
        s_dst_addrs[i] = dst_addrs[global_chunk_id];
        s_sizes[i] = sizes[global_chunk_id];
    }
    __syncthreads();

    // Phase 2: Process each chunk using metadata from shared memory
    for (int chunk_id = 0; chunk_id < num_chunks_this_block; chunk_id++) {
        uint8_t* src = s_src_addrs[chunk_id];
        uint8_t* dst = s_dst_addrs[chunk_id];
        size_t size = s_sizes[chunk_id];

        // Check if both src and dst are aligned to 16 bytes for float4 access
        uintptr_t src_addr = (uintptr_t)src;
        uintptr_t dst_addr = (uintptr_t)dst;
        bool is_aligned = ((src_addr % FLOAT4_SIZE) == 0) && ((dst_addr % FLOAT4_SIZE) == 0);

        if (is_aligned && size >= FLOAT4_SIZE) {
            // Fast path: vectorized float4 copies
            size_t aligned_size = (size / FLOAT4_SIZE) * FLOAT4_SIZE;

            // All 1024 threads cooperate on float4 copies
            #pragma unroll 4
            for (size_t offset = threadIdx.x * FLOAT4_SIZE;
                 offset < aligned_size;
                 offset += blockDim.x * FLOAT4_SIZE) {
                // Vectorized 16-byte load and store
                float4 data = *((float4*)(src + offset));
                *((float4*)(dst + offset)) = data;
            }

            // Handle remaining bytes (< 16 bytes) with byte-by-byte copy
            for (size_t offset = aligned_size + threadIdx.x;
                 offset < size;
                 offset += blockDim.x) {
                dst[offset] = src[offset];
            }
        } else {
            // Fallback path: byte-by-byte copy for unaligned addresses
            // Still use all threads for parallelism
            for (size_t offset = threadIdx.x; offset < size; offset += blockDim.x) {
                dst[offset] = src[offset];
            }
        }
    }
}

}


