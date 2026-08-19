# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process-wide compatibility hooks for the VERL MLite examples."""

import os
import sys


def _is_ray_control_plane_process() -> bool:
    """Keep Ray agents below raylet's fixed 15-second startup deadline."""
    command = " ".join(sys.argv)
    return any(
        marker in command
        for marker in (
            "/ray/dashboard/agent.py",
            "/ray/_private/runtime_env/agent/",
        )
    )


if (
    os.environ.get("VERL_MLITE_SKIP_RUNTIME_PATCHES") != "1"
    and not _is_ray_control_plane_process()
):
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()

    # Keep the proven DS4 environment process-wide: canonical DeepGEMM for
    # dense kernels, with only the validated batch-invariant masked-grouped
    # entry point overlaid.  vLLM TP workers use ``spawn`` and therefore must
    # install this in sitecustomize; a parent-process monkeypatch is not
    # inherited.
    if os.environ.get("VLLM_BATCH_INVARIANT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from rat_kernel_compat import (
            configure_deep_gemm_batch_invariant,
            enable_batched_deepgemm_batch_invariance,
            install_deep_gemm_batch_invariant_overlay,
            patch_deep_ep_buffer_kwargs,
        )

        # The proven four-layer gate uses DeepEP-align_fp8, whose Buffer does
        # not expose vLLM's newer, disabled-by-default enable_shrink keyword.
        # Spawned rollout workers must install the same compatibility shim.
        patch_deep_ep_buffer_kwargs()
        install_deep_gemm_batch_invariant_overlay()
        if configure_deep_gemm_batch_invariant():
            enable_batched_deepgemm_batch_invariance()
