# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/vllm-project/vllm.
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
"""vLLM-style Triton fused MoE kernel (BF16) for Megatron inference.

CUDA-graph compatible: all indirection table construction happens on-device
via Triton kernels with fixed-size buffers and valid_tokens gating.
"""

from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    libdevice = MagicMock()

from megatron.core.inference.moe import batch_invariant
from megatron.core.inference.moe.activations import bounded_silu_mul
from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.permute import (
    _get_num_sms,
    compute_expert_offsets,
    compute_local_tokens_per_expert,
)

# ---------------------------------------------------------------------------
# Triton kernel – BF16 grouped GEMM with indirect token addressing
# ---------------------------------------------------------------------------


def _get_default_config(M: int, E: int, top_k: int) -> dict:
    """Pick BLOCK_SIZE_*, GROUP_SIZE_M, num_warps, num_stages from M, E, top_k.

    Mirrors vLLM's ``get_default_config`` (bf16/fp16 branch) verbatim:
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py

    M here is the host-side token-count hint (``num_tokens_hint`` in
    ``vllm_fused_moe``), NOT ``hidden_states.size(0)``. The hint is the
    expected per-step token count; the worst-case buffer size would over-tune
    for prefill on every decode step.

    Two intuitions drive the choices:
      1. Small M is memory-bound (favor tall/narrow tiles, more pipeline
         stages); large M is compute-bound (favor short/wide tiles, more warps).
      2. Padding tax dominates at small M — the indirection table pads M-tiles
         per expert, so small M-tiles minimize wasted rows.
    """
    # BLOCK_SIZE_M: shrink at small M to limit per-expert padding waste.
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128

    # BLOCK_SIZE_N: small M is memory-bound on weights, narrow N keeps weight
    # traffic in check; large M has enough FMAs per weight load for wider N.
    block_n = 64 if M <= 64 else 128

    # BLOCK_SIZE_K: small M needs depth in K to keep tensor cores fed; large M
    # already has enough M*N work, so shorter K reduces accumulator stall.
    block_k = 128 if M <= 64 else 64

    # GROUP_SIZE_M: tile-grouping for L2 reuse on weight tiles. Only profitable
    # when each expert sees enough adjacent M-tiles.
    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1

    # num_warps: small M doesn't justify register pressure of more warps;
    # large M is compute-bound and feeds an MMA pipeline that wants more.
    num_warps = 4 if M <= 128 else 8

    # num_stages: extra prefetch only pays off when memory-bound (very small M).
    num_stages = 4 if M <= 32 else 3

    return {
        'BLOCK_SIZE_M': block_m,
        'BLOCK_SIZE_N': block_n,
        'BLOCK_SIZE_K': block_k,
        'GROUP_SIZE_M': group_m,
        'num_warps': num_warps,
        'num_stages': num_stages,
    }


def _config(block_m, block_n, block_k, num_warps, num_stages, group_m=1) -> dict:
    """Build a GEMM launch config from tile sizes and scheduling parameters."""
    return {
        'BLOCK_SIZE_M': block_m,
        'BLOCK_SIZE_N': block_n,
        'BLOCK_SIZE_K': block_k,
        'GROUP_SIZE_M': group_m,
        'num_warps': num_warps,
        'num_stages': num_stages,
    }


# Offline-tuned BF16 configs keyed by (SM, local experts, FC1 width, hidden size, top-k).
# Rows: (max token hint, FC1 config, FC2 config); config args: tile M/N/K, warps, stages.
# FC1 and FC2 share BLOCK_SIZE_M because they reuse the same indirection table.
# Unmatched keys or hints above the last entry use _get_default_config.
_TUNED_CONFIGS = {
    ((10, 3), 32, 1856, 2688, 6): (
        (8, _config(16, 64, 128, 4, 4), _config(16, 64, 64, 4, 5)),
        (16, _config(16, 64, 128, 4, 3), _config(16, 64, 64, 4, 3)),
        (32, _config(16, 128, 128, 4, 3), _config(16, 64, 64, 4, 3)),
        (64, _config(16, 64, 64, 4, 3), _config(16, 64, 64, 4, 5)),
        (128, _config(16, 64, 128, 4, 2), _config(16, 64, 128, 4, 3)),
        (256, _config(16, 128, 128, 4, 2), _config(16, 128, 64, 4, 5)),
        (512, _config(32, 128, 128, 4, 2), _config(32, 128, 64, 4, 4)),
        (1024, _config(64, 128, 64, 8, 4), _config(64, 128, 64, 8, 3)),
        (8192, _config(128, 256, 64, 8, 4), _config(128, 256, 64, 8, 4)),
    ),
    ((10, 3), 16, 1856, 2688, 6): (
        (16, _config(16, 64, 128, 4, 5), _config(16, 64, 64, 4, 5)),
        (32, _config(16, 64, 128, 4, 4), _config(16, 64, 64, 4, 5)),
        (128, _config(16, 64, 128, 4, 3), _config(16, 64, 64, 4, 3)),
        (256, _config(16, 128, 128, 8, 4), _config(16, 64, 64, 4, 3)),
        (512, _config(32, 128, 64, 4, 5), _config(32, 128, 64, 4, 4)),
        (1024, _config(64, 128, 64, 8, 4), _config(64, 128, 64, 4, 3)),
        (8192, _config(128, 256, 64, 8, 4), _config(128, 256, 64, 8, 4)),
    ),
}


_DEVICE_CAPABILITY = None


def _get_device_capability(device: torch.device) -> tuple[int, int]:
    """Compute capability of this process's GPU, cached like _get_num_sms."""
    global _DEVICE_CAPABILITY
    if _DEVICE_CAPABILITY is None:
        _DEVICE_CAPABILITY = torch.cuda.get_device_capability(device)
    return _DEVICE_CAPABILITY


def _get_launch_configs(
    M: int, num_local_experts: int, top_k: int, N: int, K: int, device: torch.device
) -> tuple[dict, dict]:
    """FC1 and FC2 launch configs for a token-count hint M.

    Uses a measured _TUNED_CONFIGS entry for this device and expert shape when one
    covers M, else _get_default_config for both GEMMs.
    """
    key = (_get_device_capability(device), num_local_experts, N, K, top_k)
    for max_tokens, fc1_config, fc2_config in _TUNED_CONFIGS.get(key, ()):
        if M <= max_tokens:
            return dict(fc1_config), dict(fc2_config)
    config = _get_default_config(M=M, E=num_local_experts, top_k=top_k)
    return config, dict(config)


@triton.jit
def _fused_moe_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    N,
    K,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    activation_clamp_scale,
    # Flags / constexprs
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_SQUARED_RELU: tl.constexpr,
    CLAMP_ACTIVATION: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused MoE grouped GEMM with indirect token addressing.

    Body mirrors vLLM's `fused_moe_kernel` verbatim except for the
    `FUSE_SQUARED_RELU` branch (Megatron applies relu+square in fp32 on
    the accumulator before the bf16 cast — strictly more accurate than
    upstream's separate post-FC1 activation kernel).

    Grid is sized host-side from `num_tokens_hint` (the typical-case token
    count), not the worst-case buffer length, so launch overhead at decode
    stays small. When the actual padded length exceeds the hinted grid
    size (rare prefill spikes), each CTA strides over multiple tiles via
    the outer `tl.range` loop.
    """
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    total_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_init = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for pid in tl.range(pid_init, total_tiles, grid_size):
        # GROUP_SIZE_M swizzle: pid → (pid_m, pid_n).  Mirrors upstream vLLM.
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Skip padding tiles whose expert slot was never assigned.  In
        # vLLM this also handles non-local experts via `write_zeros_to_output`;
        # our scatter excludes non-local pairs from `sorted_token_ids` entirely,
        # so `expert_id == -1` only fires on tail padding and we just skip.
        # (Triton's JIT does not support `continue`, so we gate the body.)
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts != -1:
            offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
            token_mask = offs_token < num_valid_tokens

            # `% N` keeps overflow lanes in-bounds; matching C-store mask drops
            # their contribution.  Saves a 2-D bounds check inside the K loop.
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

            a_ptrs = a_ptr + (
                offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            # Megatron-only: squared-relu fused on the fp32 accumulator before
            # the bf16 cast.  Upstream runs relu+square as a separate bf16 kernel.
            if FUSE_SQUARED_RELU:
                accumulator = tl.maximum(accumulator, 0.0)
                if CLAMP_ACTIVATION:
                    # Tanh soft clamp of the pre-activation (training's
                    # activation_func_tanh_clamp_scale), bounding the squared-relu output
                    # by s**2.  Applied after the relu rather than before, which is exact:
                    # s * tanh(x / s) is non-decreasing and maps 0 to 0, so it commutes
                    # with the relu.  Kept in fp32 through the square, matching training's
                    # fused weighted_clamped_squared_relu, which has no intermediate
                    # downcast between the clamp and the square.
                    accumulator = activation_clamp_scale * libdevice.tanh(
                        accumulator / activation_clamp_scale
                    )
                accumulator *= accumulator

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                accumulator *= moe_weight[:, None]

            accumulator = accumulator.to(tl.bfloat16)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Indirection table construction (CUDA-graph safe, fully on-device)
# ---------------------------------------------------------------------------


def _ceil_div(a, b):
    return (a + b - 1) // b


def _align_expert_bins(num_local_experts: int) -> int:
    """Histogram bins of the parallel align kernel: local experts, padded to a power of two."""
    return max(16, 1 << (num_local_experts - 1).bit_length())


class VllmFusedMoeBuffers:
    """Class-level persistent intermediate buffers for the vLLM fused-MoE kernels.

    Allocated once at engine init (from DynamicInferenceContext) so the hot
    path performs no allocations inside CUDA graph capture, and every MoE
    layer's graph shares this single buffer set. Without this, the
    intermediates are allocated inside each layer's capture and recycled by
    the shared graph mempool's free-list reuse — same net memory, but
    addresses then depend on allocator policy rather than being fixed.

    Sharing one set across all layers/graphs is safe because graph replays are
    serialized on one stream and each buffer is fully rewritten (gated by the
    indirection tables) before it is read within a single vllm_fused_moe call.

    Like the dispatcher symmetric-memory buffers, these persist for the process
    lifetime, including across engine suspend/resume: suspend deletes the CUDA
    graphs (whose captures bake in these addresses), and resume re-captures
    against the same live buffers. When not allocated (eager/standalone use,
    tests), get() falls back to torch.empty.
    """

    # name -> flat 1D tensor; requests are served as sliced views.
    _buffers: Optional[dict] = None

    @classmethod
    def allocate_buffers(
        cls,
        max_tokens: int,
        topk: int,
        fc1_output_size: int,
        hidden_size: int,
        num_local_experts: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Allocate worst-case-sized buffers. Must run outside CUDA graph capture.

        Args:
            max_tokens: max rows of the hidden states entering vllm_fused_moe
                (for the NVLS dispatcher: per_rank_worst_case_token_count * ep_size).
            topk: MoE router top-k.
            fc1_output_size: per-expert FC1 output width (2x ffn size for gated).
            hidden_size: hidden width at the MoE (moe_latent_size if set).
            num_local_experts: experts on this rank.
            device: target device (default: current CUDA device).
        """
        if device is None:
            device = torch.cuda.current_device()
        num_valid = max_tokens * topk
        # Worst case over the BLOCK_SIZE_M choices in _get_default_config
        # ({16, 32, 64, 128}): sorted ids are longest at block_m=128, the
        # per-block expert table is longest at block_m=16.
        max_sorted_worst = num_valid + 128 * (num_local_experts + 1)
        max_blocks_worst = _ceil_div(num_valid, 16) + (num_local_experts + 1)
        buffers = {
            'intermediate1': torch.empty(
                num_valid * fc1_output_size, dtype=torch.bfloat16, device=device
            ),
            'intermediate3': torch.empty(
                num_valid * hidden_size, dtype=torch.bfloat16, device=device
            ),
            'sorted_token_ids': torch.empty(max_sorted_worst, dtype=torch.int32, device=device),
            'expert_ids': torch.empty(max_blocks_worst, dtype=torch.int32, device=device),
        }
        # Only ordinary mode uses parallel alignment. Preserve BI's allocation order.
        if not batch_invariant.enabled():
            buffers['num_tokens_post_padded'] = torch.empty(1, dtype=torch.int32, device=device)
        # _moe_sum output when no external buffer (e.g. RSV) is provided.
        buffers['moe_sum_out'] = torch.empty(
            max_tokens * hidden_size, dtype=torch.float32, device=device
        )
        cls._buffers = buffers

    @classmethod
    def _delete_buffers(cls) -> None:
        """Release the buffers (tests / re-init)."""
        cls._buffers = None

    @classmethod
    def get(cls, name: str, shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
        """Return a view of the persistent buffer, or torch.empty when not allocated
        or the request does not fit (different dtype/device or larger than sized)."""
        if cls._buffers is not None:
            buf = cls._buffers.get(name)
            numel = 1
            for s in shape:
                numel *= s
            if (
                buf is not None
                and buf.dtype == dtype
                and buf.device == torch.device(device)
                and numel <= buf.numel()
            ):
                return buf[:numel].view(shape)
        return torch.empty(shape, dtype=dtype, device=device)


@triton.jit
def _init_sorted_ids_kernel(
    sorted_token_ids_ptr,
    expert_ids_ptr,
    max_sorted,
    max_blocks,
    pre_initialized_output_buffer_ptr,
    num_local_experts,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Initialize sorted_token_ids to SENTINEL and expert_ids to -1.

    Also zero the local-expert counters.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    if block_start < max_sorted or block_start < max_blocks:
        offs = block_start + tl.arange(0, BLOCK)
        tl.store(sorted_token_ids_ptr + offs, SENTINEL, mask=offs < max_sorted)
        tl.store(expert_ids_ptr + offs, -1, mask=offs < max_blocks)
        tl.store(pre_initialized_output_buffer_ptr + offs, 0, mask=offs < num_local_experts)


@triton.jit
def _scatter_token_indices_kernel(
    routing_map_ptr,
    sorted_token_ids_ptr,
    counters_ptr,
    valid_tokens_ptr,
    topk: tl.constexpr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    max_pairs,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter local-expert pair indices into the padded indirection table.

    Only local expert pairs are written; non-local pairs are skipped (the
    _moe_sum kernel handles them by checking the routing map directly).
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    valid_pairs = valid_tokens * topk
    if pid * BLOCK_SIZE >= valid_pairs:
        return
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < valid_pairs

    eids = tl.load(routing_map_ptr + offs, mask=mask, other=-1)
    lids = eids - local_expert_start
    is_local = (lids >= 0) & (lids < num_local_experts) & mask

    local_pos = tl.atomic_add(counters_ptr + lids, 1, mask=is_local)
    tl.store(sorted_token_ids_ptr + local_pos, offs, mask=is_local)


@triton.jit
def _fill_expert_block_ids_kernel(
    expert_ids_ptr,
    exclusive_offsets_ptr,
    inclusive_offsets_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fill expert_ids with expert index for each BLOCK_SIZE_M block.

    Grid: one CTA per expert (parallelised across experts).
    Inner loop uses vectorised stores of BLOCK elements at a time.
    """
    e = tl.program_id(0)
    start_block = tl.load(exclusive_offsets_ptr + e) // BLOCK_SIZE_M
    end_block = tl.load(inclusive_offsets_ptr + e) // BLOCK_SIZE_M
    num_blocks = end_block - start_block
    for off in tl.range(0, num_blocks, BLOCK):
        idxs = start_block + off + tl.arange(0, BLOCK)
        tl.store(expert_ids_ptr + idxs, e, mask=idxs < end_block)


def _moe_align_block_size_cuda_graphable(
    routing_map: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build indirection tables for the vLLM kernel, fully on-device.

    Replaces the original _moe_align_block_size which used .item() calls
    and host-side loops. All buffers are allocated at fixed max sizes so
    the function is safe for CUDA graph capture.

    Args:
        routing_map: [max_tokens, topk] expert assignments.
        block_size: BLOCK_SIZE_M for the vLLM kernel.
        num_local_experts: experts on this rank.
        local_expert_start: first global expert index on this rank.
        valid_tokens: scalar int32 CUDA tensor.

    Returns:
        sorted_token_ids: [max_sorted] int32 indirection table.
        expert_ids: [max_blocks] int32 expert per block.
        num_tokens_post_padded: [1] int32 (local expert padded count).
    """
    max_tokens, topk = routing_map.shape
    device = routing_map.device

    max_sorted = max_tokens * topk + block_size * (num_local_experts + 1)
    max_blocks = _ceil_div(max_sorted, block_size)
    sentinel = max_tokens * topk

    sorted_token_ids = VllmFusedMoeBuffers.get(
        "sorted_token_ids", (max_sorted,), torch.int32, device
    )
    expert_ids = VllmFusedMoeBuffers.get("expert_ids", (max_blocks,), torch.int32, device)

    pre_initialized_output_buffer = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    INIT_BLOCK = 1024
    # max_sorted exceeds num_local_experts, so this grid also covers the counters.
    init_grid = _ceil_div(max(max_sorted, max_blocks), INIT_BLOCK)
    _init_sorted_ids_kernel[(init_grid,)](
        sorted_token_ids,
        expert_ids,
        max_sorted,
        max_blocks,
        pre_initialized_output_buffer,
        num_local_experts,
        SENTINEL=sentinel,
        BLOCK=INIT_BLOCK,
    )

    # A separate, stream-ordered launch is necessary before atomic counting.
    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map,
        local_expert_start,
        num_local_experts,
        valid_tokens,
        persistent=True,
        pre_initialized_output_buffer=pre_initialized_output_buffer,
    )
    exclusive_offsets, inclusive_offsets = compute_expert_offsets(
        tokens_per_expert, alignment=block_size
    )

    _fill_expert_block_ids_kernel[(num_local_experts,)](
        expert_ids, exclusive_offsets, inclusive_offsets, BLOCK_SIZE_M=block_size, BLOCK=128
    )

    max_pairs = max_tokens * topk
    SCATTER_BLOCK = 256
    scatter_grid = _ceil_div(max_pairs, SCATTER_BLOCK)
    _scatter_token_indices_kernel[(scatter_grid,)](
        routing_map,
        sorted_token_ids,
        exclusive_offsets,
        valid_tokens,
        topk,
        local_expert_start,
        num_local_experts,
        max_pairs,
        BLOCK_SIZE=SCATTER_BLOCK,
    )

    num_tokens_post_padded = inclusive_offsets[-1:]
    return sorted_token_ids, expert_ids, num_tokens_post_padded


@triton.jit
def _parallel_align_block_size_kernel(
    routing_map_ptr,
    valid_tokens_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    local_expert_start,
    max_pairs,
    SENTINEL,
    topk: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_BINS: tl.constexpr,
    FIRST_HIST_CHUNK: tl.constexpr,
    HIST_CHUNK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Multi-CTA, atomics-free build of the vLLM indirection tables.

    Produces the same tables as _moe_align_block_size_cuda_graphable over the
    range the GEMM reads, [0, num_tokens_post_padded): per-expert counts padded
    to BLOCK_SIZE_M, the expert id of every BLOCK_SIZE_M block, SENTINEL in the
    padding slots, and every local (token, slot) pair index in its expert's
    segment. Entries past num_tokens_post_padded are left untouched; the GEMM
    grid never reads them.

    The valid pairs are split into CHUNK-pair chunks, and the first CTAs of the
    grid take one contiguous run of chunks each; the rest exit. Every working
    CTA counts the pairs of each expert over all valid pairs and over those
    before its run itself, so no CTA waits for another and no counter needs
    atomics. A pair's slot is its expert's segment start, plus that expert's
    pairs before the pair's chunk, plus those earlier in the chunk: each
    segment lists its pairs in increasing pair order (the multi-kernel
    builder leaves that order to its atomics; each GEMM row is computed
    independently, so results do not depend on it). The grid does not depend
    on the token-count hint, so an underestimated hint costs no parallelism.
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    bins = tl.arange(0, NUM_BINS)

    # The first routing load does not wait for valid_tokens.
    offs = tl.arange(0, FIRST_HIST_CHUNK)
    lid = tl.load(routing_map_ptr + offs, mask=offs < max_pairs, other=-1) - local_expert_start
    valid_pairs = tl.load(valid_tokens_ptr) * topk
    num_chunks = tl.cdiv(valid_pairs, CHUNK)
    num_working = tl.maximum(tl.minimum(num_ctas, num_chunks), 1)
    if pid >= num_working:
        return
    chunks_per_cta = tl.cdiv(num_chunks, num_working)
    first = pid * chunks_per_cta * CHUNK
    last = tl.minimum(first + chunks_per_cta * CHUNK, valid_pairs)

    # 1) Per-expert counts of all valid pairs, and of the pairs before `first`.
    is_local = (offs < valid_pairs) & (lid >= 0) & (lid < num_local_experts)
    lid = tl.where(is_local, lid, 0).to(tl.int32)
    counts = tl.histogram(lid, NUM_BINS, mask=is_local)
    before = tl.zeros([NUM_BINS], dtype=tl.int32)
    if pid > 0:
        before = tl.histogram(lid, NUM_BINS, mask=is_local & (offs < first))
    for start in range(FIRST_HIST_CHUNK, valid_pairs, HIST_CHUNK):
        h_offs = start + tl.arange(0, HIST_CHUNK)
        h_lid = tl.load(routing_map_ptr + h_offs, mask=h_offs < valid_pairs, other=-1)
        h_lid = h_lid - local_expert_start
        h_local = (h_offs < valid_pairs) & (h_lid >= 0) & (h_lid < num_local_experts)
        h_lid = tl.where(h_local, h_lid, 0).to(tl.int32)
        counts += tl.histogram(h_lid, NUM_BINS, mask=h_local)
        before += tl.histogram(h_lid, NUM_BINS, mask=h_local & (h_offs < first))

    # 2) Segment offsets: counts rounded up to BLOCK_SIZE_M (empty experts stay 0).
    padded = tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
    inclusive = tl.cumsum(padded, axis=0)
    exclusive = inclusive - padded
    total = tl.sum(padded, axis=0)
    if pid == 0:
        tl.store(num_tokens_post_padded_ptr, total)

    # 3) SENTINEL in each segment's padding tail (fewer than BLOCK_SIZE_M slots),
    #    experts spread over the working CTAs. Real pairs fill the other slots,
    #    so no slot is written twice.
    tail = (exclusive + counts)[:, None] + tl.arange(0, BLOCK_SIZE_M)[None, :]
    mine = (bins % num_working) == pid
    tl.store(
        sorted_token_ids_ptr + tail, SENTINEL, mask=mine[:, None] & (tail < inclusive[:, None])
    )

    # 4) Expert of each block: the number of experts whose segment ends at or
    #    before it (empty experts end where the previous one ends).
    segment_end_block = tl.where(bins < num_local_experts, inclusive // BLOCK_SIZE_M, 1 << 30)
    num_blocks = total // BLOCK_SIZE_M
    for start in range(pid * CHUNK, num_blocks, num_working * CHUNK):
        blocks = start + tl.arange(0, CHUNK)
        owner = tl.sum((blocks[:, None] >= segment_end_block[None, :]).to(tl.int32), axis=1)
        tl.store(expert_ids_ptr + blocks, owner, mask=blocks < num_blocks)

    # 5) Place this CTA's pairs.
    rows = tl.arange(0, CHUNK)
    next_slot = exclusive + before
    for start in range(first, last, CHUNK):
        p_offs = start + rows
        p_lid = tl.load(routing_map_ptr + p_offs, mask=p_offs < last, other=-1)
        p_lid = p_lid - local_expert_start
        p_local = (p_offs < last) & (p_lid >= 0) & (p_lid < num_local_experts)
        p_lid = tl.where(p_local, p_lid, -1).to(tl.int32)
        earlier_same = (p_lid[:, None] == p_lid[None, :]) & (rows[None, :] < rows[:, None])
        rank = tl.sum(earlier_same.to(tl.int32), axis=1)
        slot = tl.sum(tl.where(p_lid[:, None] == bins[None, :], next_slot[None, :], 0), axis=1)
        tl.store(sorted_token_ids_ptr + slot + rank, p_offs, mask=p_local)
        if start + CHUNK < last:
            next_slot += tl.histogram(tl.maximum(p_lid, 0), NUM_BINS, mask=p_local)


# Multi-CTA alignment limit in hinted token-expert pairs (num_tokens_hint * topk).
# Larger hints or BLOCK_SIZE_M >= 128 use the multi-kernel path.
_PARALLEL_ALIGN_MAX_PAIRS = 12288
_PARALLEL_ALIGN_CHUNK = 32
_PARALLEL_ALIGN_NUM_WARPS = 8


def _moe_align_block_size_parallel(
    routing_map: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
    num_pairs_hint: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-CTA equivalent of _moe_align_block_size_cuda_graphable.

    Same arguments and return values, plus the host-side pair-count hint that
    sizes the first histogram tile; only entries before num_tokens_post_padded
    are written (see _parallel_align_block_size_kernel).
    """
    max_tokens, topk = routing_map.shape
    device = routing_map.device
    max_pairs = max_tokens * topk
    max_sorted = max_pairs + block_size * (num_local_experts + 1)
    max_blocks = _ceil_div(max_sorted, block_size)
    sorted_token_ids = VllmFusedMoeBuffers.get(
        "sorted_token_ids", (max_sorted,), torch.int32, device
    )
    expert_ids = VllmFusedMoeBuffers.get("expert_ids", (max_blocks,), torch.int32, device)
    num_tokens_post_padded = VllmFusedMoeBuffers.get(
        "num_tokens_post_padded", (1,), torch.int32, device
    )
    num_ctas = max(1, min(_ceil_div(max_pairs, _PARALLEL_ALIGN_CHUNK), _get_num_sms(device)))
    _parallel_align_block_size_kernel[(num_ctas,)](
        routing_map,
        valid_tokens,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        local_expert_start,
        max_pairs,
        max_pairs,
        topk=topk,
        num_local_experts=num_local_experts,
        BLOCK_SIZE_M=block_size,
        NUM_BINS=_align_expert_bins(num_local_experts),
        # The hinted pair count as a power of two in [64, 1024].
        FIRST_HIST_CHUNK=min(max(triton.next_power_of_2(max(num_pairs_hint, 1)), 64), 1024),
        HIST_CHUNK=1024,
        CHUNK=_PARALLEL_ALIGN_CHUNK,
        num_warps=_PARALLEL_ALIGN_NUM_WARPS,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ---------------------------------------------------------------------------
# Kernel launcher
# ---------------------------------------------------------------------------


def _invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    grid_size: int,
    fuse_squared_relu: bool = False,
    activation_clamp_scale: Optional[float] = None,
):
    """Launch the Triton fused-MoE kernel for one GEMM pass.

    Body matches upstream vLLM `fused_moe_kernel` (1 CTA per (pid_m, pid_n)
    tile, raw pointer arithmetic with `% N` on the N axis), apart from the
    optional fused squared-relu activation in fp32 and its optional tanh
    soft clamp.

    `grid_size` is sized host-side from `num_tokens_hint` so launch overhead
    at decode is small.  When the actual padded length exceeds the hinted
    grid size, each CTA strides over additional tiles via the kernel's outer
    `tl.range`.  The full launch config (tile sizes, warps, stages) is picked
    host-side by ``_get_default_config`` from M = num_tokens_hint.
    """
    M = A.size(0)
    num_tokens = M * top_k

    _fused_moe_kernel[(grid_size,)](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        activation_clamp_scale if activation_clamp_scale is not None else 0.0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        FUSE_SQUARED_RELU=fuse_squared_relu,
        CLAMP_ACTIVATION=fuse_squared_relu and activation_clamp_scale is not None,
        top_k=top_k,
        BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=config['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=config['BLOCK_SIZE_K'],
        GROUP_SIZE_M=config['GROUP_SIZE_M'],
        num_warps=config['num_warps'],
        num_stages=config['num_stages'],
    )


# ---------------------------------------------------------------------------
# Fused topk reduction (replaces torch.sum + copy)
# ---------------------------------------------------------------------------


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    topk_weights_ptr,
    valid_tokens_ptr,
    routing_map_ptr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    K,
    topk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    APPLY_WEIGHTS: tl.constexpr = True,
    ACC_FP64: tl.constexpr = False,
):
    """Reduce topk dimension with routing weight application.

    input:  [max_tokens * topk, K] bf16
    output: [max_tokens, K] — dtype matches the output buffer (fp32 or bf16)

    For token t < valid_tokens: output[t] = sum of input[t*topk+k] * prob[t*topk+k]
    over topk slots k where the expert is local.  Non-local slots are skipped
    (their values in `input` are undefined because FC2 only processes
    local-expert blocks).
    Rows for t >= valid_tokens are not written; downstream consumers
    (e.g. reduce-scatter-v) only read the first valid_tokens rows.
    Routing weight multiplication and accumulation in fp32 for numerical accuracy.

    Persistent grid: launches BLOCK_M CTAs that stride over valid_tokens.
    CUDA-graph safe (grid is static); the loop bound is loaded device-side.
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)

    for token_id in tl.range(pid, valid_tokens, BLOCK_M):
        token_id_i64 = token_id.to(tl.int64)
        base = token_id_i64 * topk * K

        # k_idx outer / topk inner keeps the live accumulator at one BLOCK_K tile.
        # Swapping (topk outer) would need NUM_K_BLOCKS persistent accumulators
        # (~NUM_K_BLOCKS * BLOCK_K * 4 B), which spills / cuts occupancy at large K.
        for k_idx in range(NUM_K_BLOCKS):
            offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            acc = tl.zeros([BLOCK_K], dtype=tl.float64 if ACC_FP64 else tl.float32)
            for t in range(topk):
                eid = tl.load(routing_map_ptr + token_id * topk + t)
                lid = eid - local_expert_start
                if lid >= 0 and lid < num_local_experts:
                    v = tl.load(input_ptr + base + t * K + offs_k, mask=k_mask, other=0.0)
                    if APPLY_WEIGHTS:
                        w = tl.load(topk_weights_ptr + token_id * topk + t)
                        if ACC_FP64:
                            acc += v.to(tl.float64) * w.to(tl.float64)
                        else:
                            acc += v.to(tl.float32) * w
                    else:
                        if ACC_FP64:
                            acc += v.to(tl.float64)
                        else:
                            acc += v.to(tl.float32)

            tl.store(output_ptr + token_id_i64 * K + offs_k, acc.to(tl.float32), mask=k_mask)


@triton.jit
def _moe_sum_tiled_kernel(
    input_ptr,
    output_ptr,
    topk_weights_ptr,
    valid_tokens_ptr,
    routing_map_ptr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    K,
    topk: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    APPLY_WEIGHTS: tl.constexpr = True,
):
    """Reduce topk dimension with routing weight application.

    input:  [max_tokens * topk, K] bf16
    output: [max_tokens, K] — dtype matches the output buffer (fp32 or bf16)

    For token t < valid_tokens: output[t] = sum of input[t*topk+k] * prob[t*topk+k]
    over topk slots k where the expert is local.  Non-local slots are skipped
    (their values in `input` are undefined because FC2 only processes
    local-expert blocks).
    Rows for t >= valid_tokens are not written; downstream consumers
    (e.g. reduce-scatter-v) only read the first valid_tokens rows.
    Routing weight multiplication and accumulation in fp32 for numerical accuracy.

    Work items are [BLOCK_T tokens, BLOCK_K columns] tiles. The routing ids of
    a tile are loaded before its topk slot rows, so the slot loads are
    independent and many rows are in flight per CTA (a token-at-a-time loop
    with a dependent routing load per slot leaves HBM mostly idle at prefill
    sizes). Each element still accumulates its local slots in slot order,
    skipping non-local slots, so results do not depend on the tiling.

    Persistent grid: the CTAs stride over the tiles of valid_tokens.
    CUDA-graph safe (grid is static); the loop bound is loaded device-side.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    num_tiles = tl.cdiv(valid_tokens, BLOCK_T) * num_k_blocks

    for tile in tl.range(pid, num_tiles, num_programs):
        offs_t = (tile // num_k_blocks) * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_k = (tile % num_k_blocks) * BLOCK_K + tl.arange(0, BLOCK_K)
        t_mask = offs_t < valid_tokens
        k_mask = offs_k < K
        slot_base = offs_t.to(tl.int64) * topk

        acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)
        for t in tl.static_range(topk):
            eid = tl.load(routing_map_ptr + slot_base + t, mask=t_mask, other=-1)
            lid = eid - local_expert_start
            is_local = t_mask & (lid >= 0) & (lid < num_local_experts)
            v = tl.load(
                input_ptr + (slot_base + t)[:, None] * K + offs_k[None, :],
                mask=is_local[:, None] & k_mask[None, :],
                other=0.0,
            )
            if APPLY_WEIGHTS:
                w = tl.load(topk_weights_ptr + slot_base + t, mask=is_local, other=0.0)
                contribution = v.to(tl.float32) * w[:, None]
            else:
                contribution = v.to(tl.float32)
            # Skip (rather than add zero for) non-local slots, which keeps a
            # -0.0 accumulator bit-identical.
            acc = tl.where(is_local[:, None], acc + contribution, acc)

        tl.store(
            output_ptr + offs_t.to(tl.int64)[:, None] * K + offs_k[None, :],
            acc.to(tl.float32),
            mask=t_mask[:, None] & k_mask[None, :],
        )


def _moe_sum(
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    max_tokens: int,
    topk: int,
    K: int,
    valid_tokens: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    out: Optional[torch.Tensor] = None,
    apply_weights: bool = True,
    acc_fp64: bool = False,
) -> torch.Tensor:
    """Fused topk reduction: [max_tokens*topk, K] bf16 → [max_tokens, K].

    Applies routing weights and reduces over topk in a single kernel.
    Accumulates in fp32, or fp64 when `acc_fp64` is set (the output buffer
    stays fp32 either way). When `out` is None, allocates and returns an fp32
    buffer. When `out` is provided (e.g. the RSV symmetric memory tensor),
    writes directly into it — tl.store handles the cast to the buffer's dtype.
    Only writes the first valid_tokens rows; rows beyond are left untouched
    (downstream RSV reads only the valid range). Only accumulates contributions
    from local experts; non-local topk slots are skipped (their values in
    `input` are undefined).

    Args:
        apply_weights: multiply each slot by its routing probability (default).
            Pass False when the probabilities were already applied upstream
            (e.g. at the activation in batch-invariant mode).
        acc_fp64: accumulate the topk sum in fp64, making the reduction
            order-independent by precision.
    """
    if out is None:
        out = VllmFusedMoeBuffers.get("moe_sum_out", (max_tokens, K), torch.float32, input.device)
    # Preserve the existing BI launch and arithmetic, including FP64 reduction.
    if acc_fp64 or batch_invariant.enabled():
        BLOCK_K = min(triton.next_power_of_2(K), 1024)
        NUM_K_BLOCKS = _ceil_div(K, BLOCK_K)
        BLOCK_M = _get_num_sms(input.device)
        _moe_sum_kernel[(BLOCK_M,)](
            input,
            out,
            topk_weights,
            valid_tokens,
            routing_map,
            local_expert_start,
            num_local_experts,
            K,
            topk=topk,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            NUM_K_BLOCKS=NUM_K_BLOCKS,
            APPLY_WEIGHTS=apply_weights,
            ACC_FP64=acc_fp64,
        )
        return out
    BLOCK_T = 16
    BLOCK_K = min(triton.next_power_of_2(K), 128)
    # Enough CTAs to keep many slot rows in flight at prefill sizes, capped by
    # the tile count of the worst-case token count. GB300, K=2688, top-6: 8 CTAs
    # per SM vs 4 take 43 vs 67 us at 8192 tokens and match it at decode sizes.
    num_programs = min(
        8 * _get_num_sms(input.device), _ceil_div(max_tokens, BLOCK_T) * _ceil_div(K, BLOCK_K)
    )
    _moe_sum_tiled_kernel[(num_programs,)](
        input,
        out,
        topk_weights,
        valid_tokens,
        routing_map,
        local_expert_start,
        num_local_experts,
        K,
        topk=topk,
        BLOCK_T=BLOCK_T,
        BLOCK_K=BLOCK_K,
        APPLY_WEIGHTS=apply_weights,
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def vllm_fused_moe(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation_type: ActivationType,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
    routing_map: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_tokens_hint: Optional[int] = None,
    activation_clamp_scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused MoE using the vLLM Triton grouped-GEMM kernel (BF16).

    CUDA-graph compatible: indirection tables are built entirely on-device
    using fixed-size buffers gated by valid_tokens.

    Args:
        hidden_states: [max_tokens, hidden_size] BF16 input. Only the first
            valid_tokens rows are valid; the rest are ignored.
        probs: [max_tokens, topk] fp32 routing probabilities.
        fc1_weight: [num_local_experts, fc1_out, hidden_size] BF16.
        fc2_weight: [num_local_experts, hidden_size, fc1_out] BF16.
        activation_type: ActivationType enum.
        num_local_experts: experts on this rank.
        local_expert_start: first global expert index on this rank.
        valid_tokens: scalar int32 CUDA tensor with number of valid tokens.
        routing_map: [max_tokens, topk] int expert assignments.
        out: optional [max_tokens, hidden_size] output buffer (e.g. the RSV
            symmetric memory tensor). If None, an fp32 buffer is allocated.
            When provided, tl.store casts to the buffer's dtype automatically.
        num_tokens_hint: optional host-side int with the expected number of
            valid tokens (e.g. batch_size * ep_size). Used to select a better
            BLOCK_SIZE_M instead of using the worst-case buffer size.
        activation_clamp_scale: config.activation_func_tanh_clamp_scale. When set, the
            squared-ReLU pre-activation is soft-clamped with ``s * tanh(x / s)`` before
            the square, bounding the activation output by ``s ** 2``. Only supported for
            SQUARED_RELU; the gated SiTU-GLU form of the clamp is not implemented here.

    Returns:
        [max_tokens, hidden_size] output (fp32 when out=None, else out's dtype).
        tl.store handles the implicit cast when out is a different dtype.
    """
    assert (
        hidden_states.dtype == torch.bfloat16
    ), f"vllm_fused_moe requires bf16 input, got {hidden_states.dtype}"

    max_tokens = hidden_states.size(0)
    topk = routing_map.shape[1]
    effective_tokens = num_tokens_hint if num_tokens_hint is not None else max_tokens

    N = fc1_weight.size(1)
    K = fc1_weight.size(2)

    # Mirror upstream vLLM: pick the full launch configs (tile sizes, warps,
    # stages) host-side from the token-count hint, not from the worst-case
    # buffer size. Measured per-device configs may differ between FC1 and FC2.
    batch_invariant_mode = batch_invariant.enabled()
    if batch_invariant_mode:
        fc1_config = _get_default_config(M=effective_tokens, E=num_local_experts, top_k=topk)
        # Batch-invariant mode: pin only the K-reduction recipe. The kernel
        # accumulates in fp32 with no split-K, so bits depend solely on the
        # K-loop grouping (BLOCK_SIZE_K); M/N tile shapes, tile grouping and
        # pipeline depth reorder nothing in the accumulation (bf16 products
        # are exact in fp32 — only the addition order matters). Keeping the
        # M/N tiling adaptive preserves the decode-tuned configs; pinning
        # BLOCK_SIZE_K removes the one field _get_default_config varies that
        # could change the summation order across co-batch sizes.
        fc1_config['BLOCK_SIZE_K'] = 64
        fc2_config = fc1_config
    else:
        fc1_config, fc2_config = _get_launch_configs(
            effective_tokens, num_local_experts, topk, N, K, hidden_states.device
        )

    # FC1 and FC2 share the indirection tables, hence BLOCK_SIZE_M.
    block_m = fc1_config['BLOCK_SIZE_M']
    assert fc2_config['BLOCK_SIZE_M'] == block_m
    align_args = (routing_map, block_m, num_local_experts, local_expert_start, valid_tokens)
    # Keep the large-prefill tile family on one table builder. Switching at the
    # pair-count cutoff also changes the padded-count pointer's alignment (the
    # parallel builder owns a scalar; the fallback returns an offsets slice),
    # compiling another GEMM specialization during a warmed request. Decode
    # still uses the parallel builder, including underestimated token hints.
    if (
        not batch_invariant_mode
        and block_m < 128
        and effective_tokens * topk <= _PARALLEL_ALIGN_MAX_PAIRS
    ):
        tables = _moe_align_block_size_parallel(*align_args, effective_tokens * topk)
    else:
        tables = _moe_align_block_size_cuda_graphable(*align_args)
    sorted_token_ids, expert_ids, num_post_padded = tables
    num_valid = max_tokens * topk

    # Grid sized for the typical-case token count (num_tokens_hint).  When the
    # actual num_tokens_post_padded exceeds this, the kernel's outer tl.range
    # makes each CTA stride over additional tiles — correct but with reduced
    # parallelism on rare prefill spikes.  EM hint = effective_tokens*topk +
    # BLOCK_SIZE_M*num_local_experts upper-bounds the per-expert padding.
    em_hint = effective_tokens * topk + block_m * num_local_experts
    num_pid_m_hint = _ceil_div(em_hint, block_m)
    num_pid_n_fc1 = _ceil_div(N, fc1_config['BLOCK_SIZE_N'])
    num_pid_n_fc2 = _ceil_div(K, fc2_config['BLOCK_SIZE_N'])
    grid_size_fc1 = num_pid_m_hint * num_pid_n_fc1
    grid_size_fc2 = num_pid_m_hint * num_pid_n_fc2

    topk_weights_flat = probs.reshape(-1).contiguous()

    # FC1 + activation: [max_tokens, K] → [max_tokens*topk, N]. Outside
    # batch-invariant mode, SQUARED_RELU fuses into the GEMM epilogue while
    # SwiGLU runs separately. Batch-invariant mode runs both activations
    # separately so their probability application and BF16 rounding match the
    # training TEGroupedMLP path exactly.
    assert activation_type in (ActivationType.SQUARED_RELU, ActivationType.SWIGLU)
    is_swiglu = activation_type == ActivationType.SWIGLU
    assert not (is_swiglu and activation_clamp_scale is not None), (
        "activation_func_tanh_clamp_scale is only implemented for squared ReLU here; the "
        "gated form (SiTU-GLU) has no inference kernel yet."
    )
    fuse_squared_relu = not is_swiglu
    if batch_invariant_mode:
        # Apply the activation separately below to match training-side rounding.
        fuse_squared_relu = False

    intermediate1 = VllmFusedMoeBuffers.get(
        "intermediate1", (num_valid, N), hidden_states.dtype, hidden_states.device
    )
    _invoke_fused_moe_kernel(
        hidden_states,
        fc1_weight,
        intermediate1,
        topk_weights_flat,
        sorted_token_ids,
        expert_ids,
        num_post_padded,
        mul_routed_weight=False,
        top_k=topk,
        config=fc1_config,
        grid_size=grid_size_fc1,
        fuse_squared_relu=fuse_squared_relu,
        activation_clamp_scale=activation_clamp_scale,
    )
    if batch_invariant_mode:
        live_rows = (valid_tokens * topk).to(torch.int32)
        if is_swiglu:
            # Match training: routing probabilities multiply at the activation
            # (before FC2), with the training kernel's exact rounding sequence
            # (single bf16 round of fp32 silu(gate)*up*prob). The reduction
            # below then sums with unit weights. Device-bounded to the live
            # valid_tokens*topk prefix; CUDA-graph safe.
            bound_elems = valid_tokens.to(torch.int64) * (topk * (intermediate1.shape[1] // 2))
            intermediate1 = batch_invariant.weighted_silu_mul_bounded(
                intermediate1, topk_weights_flat, bound_elems
            )
        else:
            # Nemotron-style weighted squared-ReLU rounds the squared BF16
            # activation before its FP32 probability multiply, then rounds to
            # BF16 again. Reuse the training-parity kernel with the flattened
            # routing map as the live-row mask.
            intermediate1 = batch_invariant.squared_relu_with_probs(
                intermediate1,
                routing_map.reshape(-1),
                live_rows,
                topk_weights_flat,
                activation_clamp_scale,
            )
    elif is_swiglu:
        # intermediate1 is [num_valid, 2N] (gate | up); reduce to [num_valid, N] via
        # SiLU(gate) * up over the valid_tokens*topk live rows only.
        intermediate1 = bounded_silu_mul(intermediate1, (valid_tokens * topk).to(torch.int32))

    # FC2: [max_tokens*topk, N] → [max_tokens*topk, K]. Batch-invariant mode
    # already applied routing weights at the activation to match training;
    # ordinary inference applies them in the reduction kernel.
    # Only local-expert blocks are processed; non-local positions are left
    # undefined and skipped by _moe_sum (which checks the routing map).
    intermediate3 = VllmFusedMoeBuffers.get(
        "intermediate3", (num_valid, K), hidden_states.dtype, hidden_states.device
    )
    _invoke_fused_moe_kernel(
        intermediate1,
        fc2_weight,
        intermediate3,
        topk_weights_flat,
        sorted_token_ids,
        expert_ids,
        num_post_padded,
        mul_routed_weight=False,
        top_k=1,
        config=fc2_config,
        grid_size=grid_size_fc2,
    )

    # Reduce over topk: [max_tokens*topk, K] → [max_tokens, K]
    # Applies routing weights and accumulates in fp32, writes directly to
    # out (if provided), zeros rows beyond valid_tokens, and skips non-local
    # expert slots.
    apply_routing_weights = True
    accumulate_in_fp64 = False
    if batch_invariant_mode:
        # Probabilities were applied at the activation to mirror training. Use
        # training's invariant within-rank accumulation for the unweighted sum.
        apply_routing_weights = False
        accumulate_in_fp64 = True

    return _moe_sum(
        intermediate3,
        probs,
        max_tokens,
        topk,
        K,
        valid_tokens,
        routing_map,
        local_expert_start,
        num_local_experts,
        out=out,
        apply_weights=apply_routing_weights,
        acc_fp64=accumulate_in_fp64,
    )
