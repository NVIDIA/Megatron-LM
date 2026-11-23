# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Offline-tuned launch configs for the vLLM fused-MoE GEMMs in vllm_fused_moe.py."""


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
# Unmatched keys or hints above the last entry use vllm_fused_moe._get_default_config.
TUNED_CONFIGS = {
    ((10, 3), 32, 1856, 2688, 6): (
        (1, _config(16, 32, 128, 4, 5), _config(16, 32, 128, 4, 5)),
        (2, _config(16, 64, 128, 4, 5), _config(16, 32, 128, 4, 5)),
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
    ((10, 3), 128, 2688, 1024, 22): (
        (2, _config(16, 64, 64, 4, 5), _config(16, 64, 128, 4, 5)),
        (4, _config(16, 64, 64, 4, 3), _config(16, 64, 128, 4, 4)),
        (8, _config(16, 64, 128, 4, 3), _config(16, 64, 128, 4, 3)),
        (16, _config(16, 64, 128, 4, 3), _config(16, 128, 128, 4, 2)),
        (32, _config(16, 64, 128, 4, 3), _config(16, 64, 128, 4, 3)),
        (128, _config(16, 64, 128, 4, 3), _config(16, 64, 128, 4, 2)),
        (256, _config(16, 64, 128, 4, 3), _config(16, 64, 128, 4, 3)),
        (512, _config(32, 64, 128, 4, 2), _config(32, 128, 128, 4, 2)),
        (1024, _config(64, 128, 64, 4, 3), _config(64, 128, 64, 8, 4)),
        (8192, _config(128, 256, 64, 8, 4), _config(128, 256, 64, 8, 4)),
    ),
    ((10, 3), 64, 2688, 1024, 22): (
        (1, _config(16, 32, 128, 4, 4), _config(16, 32, 256, 4, 5)),
        (2, _config(16, 64, 64, 4, 5), _config(16, 64, 128, 4, 5)),
        (4, _config(16, 64, 64, 4, 4), _config(16, 64, 128, 4, 5)),
        (8, _config(16, 64, 64, 4, 3), _config(16, 64, 128, 4, 4)),
        (16, _config(16, 64, 64, 4, 5), _config(16, 64, 128, 4, 3)),
        (32, _config(16, 64, 128, 4, 3), _config(16, 64, 64, 4, 3)),
        (128, _config(16, 64, 128, 4, 3), _config(16, 64, 128, 4, 2)),
        (256, _config(16, 64, 128, 4, 3), _config(16, 128, 128, 4, 2)),
        (512, _config(32, 128, 64, 4, 4), _config(32, 128, 128, 4, 2)),
        (1024, _config(64, 256, 64, 8, 2), _config(64, 128, 64, 4, 4)),
        (8192, _config(128, 256, 64, 4, 2), _config(128, 256, 64, 8, 4)),
    ),
    ((10, 3), 128, 5120, 2048, 22): (
        (64, _config(16, 64, 128, 4, 4), _config(16, 64, 128, 4, 4)),
        (128, _config(16, 128, 128, 8, 4), _config(16, 128, 128, 8, 4)),
        (256, _config(16, 128, 128, 4, 3), _config(16, 128, 128, 4, 3)),
        (512, _config(32, 128, 128, 4, 2), _config(32, 64, 128, 4, 3)),
        (1024, _config(64, 256, 64, 8, 5), _config(64, 128, 64, 4, 3)),
        (8192, _config(128, 256, 64, 8, 4), _config(128, 256, 64, 8, 4)),
    ),
    ((10, 3), 64, 5120, 2048, 22): (
        (2, _config(16, 64, 64, 4, 5), _config(16, 64, 128, 4, 5)),
        (4, _config(16, 64, 128, 4, 4), _config(16, 64, 128, 4, 4)),
        (8, _config(16, 64, 128, 4, 3), _config(16, 128, 128, 4, 3)),
        (32, _config(16, 64, 128, 4, 4), _config(16, 64, 128, 4, 4)),
        (64, _config(16, 64, 128, 4, 4), _config(16, 64, 128, 4, 2)),
        (128, _config(16, 64, 128, 4, 3), _config(16, 64, 128, 4, 2)),
        (256, _config(16, 128, 128, 4, 3), _config(16, 64, 128, 4, 3)),
        (512, _config(32, 128, 64, 4, 4), _config(32, 128, 128, 4, 2)),
        (1024, _config(64, 256, 64, 8, 5), _config(64, 256, 64, 8, 5)),
        (8192, _config(128, 256, 64, 8, 4), _config(128, 256, 64, 8, 4)),
    ),
}
