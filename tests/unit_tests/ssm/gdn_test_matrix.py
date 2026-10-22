# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded integration matrix; kernel-size sweeps live in the kernel unit tests."""

# tp, cp, sequence_parallel, batch, packed, pre_fusion, recompute, value_heads.
# Cover every pair of topology/layout/pre-fusion/recompute/head-count values,
# and every layout x pre-fusion x recompute triple, without the 96-case product.
# Keep the packed pre-fusion CP2 case that exposed the previous CI timeout,
# and cover packed pre-fusion plus recompute on CP4 without growing the matrix.
# Preserve CP4's single-batch unpacked layout alongside its packed cases.
POST_FUSION_DISTRIBUTED_CASES = (
    (1, 2, False, 1, True, True, False, 16),
    (1, 4, False, 1, True, True, True, 16),
    (2, 2, True, 2, False, False, False, 64),
    (4, 1, True, 1, False, True, True, 64),
    (2, 2, True, 2, False, True, True, 16),
    (1, 2, False, 1, True, False, True, 64),
    (1, 4, False, 1, False, False, True, 16),
    (1, 4, False, 2, False, True, False, 64),
    (1, 2, False, 1, False, True, False, 16),
    (1, 2, False, 2, False, False, True, 16),
    (1, 4, False, 1, True, False, False, 16),
    (2, 2, True, 1, True, True, True, 16),
    (2, 2, True, 1, False, False, False, 16),
    (4, 1, True, 2, False, False, False, 16),
    (4, 1, True, 1, True, False, False, 16),
)
