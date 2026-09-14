# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1 compressed sparse attention with cross-layer shared state (CSA2).

Package layout:

* :mod:`roles`      pure-Python role resolution / validation (importable from config code).
* :mod:`reference`  framework-free PyTorch reference operators.
* :mod:`state`      the per-microbatch shared-state container threaded through the stack.
* :mod:`compressor` V4.1 compressor (softmax-gated pooling, ratio 1 = projection).
* :mod:`indexer`    V4.1 indexer with optional two-level candidate-block selection.
* :mod:`attention`  the core attention module and the self-attention wrapper.
"""

from megatron.core.transformer.experimental_attention_variant.csa2.roles import (
    CSA2LayerMode,
    CSA2LayerPlan,
    CSA2Plan,
    resolve_csa2_plan,
)

__all__ = ["CSA2LayerMode", "CSA2LayerPlan", "CSA2Plan", "resolve_csa2_plan"]
