# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared markers for dynamic-CP CUDA graph tests."""

import pytest

from megatron.core.transformer.cuda_graphs import te_supports_graph_memory_slots

requires_te_graph_memory_slots = pytest.mark.skipif(
    not te_supports_graph_memory_slots(),
    reason="dynamic-CP graph banks need NVIDIA/TransformerEngine#3353 (_graph_memory_slots)",
)
