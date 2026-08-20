"""DeepSeek-V4 vLLM checkpoint compatibility surface.

The vLLM-aligned model deliberately uses the same parameter containers and HF
mapping as ``deepseek_v4.lite``.  Keep the historical import path as a thin
alias so callers and the full checkpoint regression suite exercise that one
shared implementation instead of maintaining a second mapper.
"""

from megatron.lite.model.deepseek_v4.lite import checkpoint as _shared
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    DeepseekV4WeightSpec,
    export_hf_weights,
    invalidate_bound_source_scales,
    load_hf_weights,
    save_hf_weights,
)
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    quantize_block_fp8_weight,
)

dist = _shared.dist
_pipeline_export_source_scales = _shared._export_source_scales

__all__ = [
    "DeepseekV4WeightSpec",
    "export_hf_weights",
    "invalidate_bound_source_scales",
    "load_hf_weights",
    "save_hf_weights",
]
