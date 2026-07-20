# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process-wide compatibility hooks for the VERL MLite examples."""

from verl_mlite.compat import (
    _patch_bucketed_weight_sender,
    _weight_sync_probe_enabled,
    apply_runtime_patches,
)

if _weight_sync_probe_enabled():
    # The probe is also used by the Megatron/mbridge control. Keep that path
    # limited to sender instrumentation instead of importing MLite's optional
    # Transformers compatibility layer into every Ray and vLLM process.
    _patch_bucketed_weight_sender()
else:
    apply_runtime_patches()
