"""The vLLM-aligned model intentionally reuses Lite checkpoint semantics."""

from megatron.lite.model.deepseek_v4.lite import checkpoint as shared
from megatron.lite.model.deepseek_v4.vllm import checkpoint as aligned
from megatron.lite.primitive.quantization import deployment_block_fp8


def test_vllm_checkpoint_surface_is_the_lite_implementation() -> None:
    assert aligned.DeepseekV4WeightSpec is shared.DeepseekV4WeightSpec
    assert aligned.load_hf_weights is shared.load_hf_weights
    assert aligned.export_hf_weights is shared.export_hf_weights
    assert aligned.save_hf_weights is shared.save_hf_weights
    assert (
        aligned.invalidate_bound_source_scales
        is shared.invalidate_bound_source_scales
    )


def test_vllm_checkpoint_uses_the_canonical_block_fp8_quantizer() -> None:
    assert (
        aligned.quantize_block_fp8_weight
        is deployment_block_fp8.quantize_block_fp8_weight
    )
