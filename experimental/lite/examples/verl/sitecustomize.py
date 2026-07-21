# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process-wide compatibility hooks for the VERL MLite examples."""

import os

if os.environ.get("VERL_MLITE_SKIP_RUNTIME_PATCHES") != "1":
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()
