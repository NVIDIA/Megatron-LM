# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Transformer Engine's CPU offloading: the ``cpu_offload`` module and the version-aware
``get_cpu_offload_context`` wrapper (``None`` when Transformer Engine is unavailable)."""

from megatron.core.utils import is_te_min_version

try:
    from transformer_engine.pytorch.cpu_offload import (
        get_cpu_offload_context as _get_cpu_offload_context,
    )

    def get_cpu_offload_context(
        enabled,
        num_layers,
        model_layers,
        activation_offloading,
        weight_offloading,
        double_buffering,
        retain_pinned_cpu_buffers,
    ):
        """Get CPU offload context and sync function."""
        if is_te_min_version("2.10.0"):
            # TE 2.10+ supports retain_pinned_cpu_buffers
            context, sync_func = _get_cpu_offload_context(
                enabled,
                num_layers,
                model_layers,
                activation_offloading,
                weight_offloading,
                double_buffering,
                retain_pinned_cpu_buffers=retain_pinned_cpu_buffers,
            )
        elif is_te_min_version("2.5.0"):
            # TE 2.5-2.9 supports double_buffering but not retain_pinned_cpu_buffers
            context, sync_func = _get_cpu_offload_context(
                enabled,
                num_layers,
                model_layers,
                activation_offloading,
                weight_offloading,
                double_buffering,
            )
        elif is_te_min_version("1.10.0.dev0"):
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, model_layers, activation_offloading, weight_offloading
            )
        else:
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, activation_offloading, weight_offloading
            )

        return context, sync_func

except ImportError:
    get_cpu_offload_context = None  # type: ignore[assignment, misc]

try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch import cpu_offload_v1 as cpu_offload
except ImportError:
    try:
        from transformer_engine.pytorch import cpu_offload
    except ImportError:
        cpu_offload = None
