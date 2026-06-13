# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Ported verbatim from DeepSeek DualPipe: https://github.com/deepseek-ai/DualPipe

from megatron.core.pipeline_parallel.dualpipe.comm import (
    set_p2p_tensor_dtype,
    set_p2p_tensor_shapes,
)
from megatron.core.pipeline_parallel.dualpipe.dualpipe import DualPipe
from megatron.core.pipeline_parallel.dualpipe.dualpipev import DualPipeV
from megatron.core.pipeline_parallel.dualpipe.utils import WeightGradStore

__all__ = [
    "DualPipe",
    "DualPipeV",
    "WeightGradStore",
    "set_p2p_tensor_shapes",
    "set_p2p_tensor_dtype",
]
