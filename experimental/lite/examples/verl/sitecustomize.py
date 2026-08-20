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
