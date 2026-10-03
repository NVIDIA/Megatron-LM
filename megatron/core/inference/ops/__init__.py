# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inference-optimized operation backends.

These fill the same slots as any other backend and are selected the same way, by
``--transformer-impl inference_optimized`` or ``--op-backend``. They live here rather than
under ``megatron/core/ops`` so the inference-only implementations stay with the rest of the
inference subsystem; ``megatron.core.ops.<family>.BACKENDS`` points at them by name.
"""

from megatron.core.inference.ops.backends import LinearInference, MoeInference, NormInference

__all__ = ["LinearInference", "MoeInference", "NormInference"]
