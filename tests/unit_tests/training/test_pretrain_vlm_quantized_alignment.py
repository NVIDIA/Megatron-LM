# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""pretrain_vlm.py rejects FP8/FP4 token counts that Transformer Engine's GEMMs cannot run."""

import pytest

# The default CLIP setting: 224x224 images in 16x16 patches give 196 patches and a class token,
# and a 512-token text sequence gives 512 + 197 = 709 language model tokens.
VISION_TOKENS = 197
DECODER_TOKENS = 709


@pytest.mark.parametrize(
    "fp8_recipe, fp4_recipe, micro_batch_size, is_valid",
    [
        ("delayed", None, 2, False),
        ("delayed", None, 8, True),
        ("tensorwise", None, 8, True),
        ("blockwise", None, 8, False),
        ("blockwise", None, 16, True),
        ("mxfp8", None, 8, False),
        ("mxfp8", None, 32, True),
        (None, "nvfp4", 16, False),
        (None, "nvfp4", 32, True),
        (None, None, 1, True),
        ("custom", None, 1, True),
    ],
)
def test_check_quantized_gemm_alignment(fp8_recipe, fp4_recipe, micro_batch_size, is_valid):
    import pretrain_vlm

    def check():
        pretrain_vlm.check_quantized_gemm_alignment(
            VISION_TOKENS,
            DECODER_TOKENS,
            micro_batch_size,
            fp8_recipe=fp8_recipe,
            fp4_recipe=fp4_recipe,
        )

    if is_valid:
        check()
    else:
        with pytest.raises(ValueError, match="the vision encoder gets 197 tokens"):
            check()


def test_check_quantized_gemm_alignment_language_model():
    import pretrain_vlm

    # 208 vision tokens fit the alignment; 709 language model tokens do not.
    with pytest.raises(ValueError, match="the language model gets 709 tokens"):
        pretrain_vlm.check_quantized_gemm_alignment(208, DECODER_TOKENS, 1, fp8_recipe="delayed")
    # The language model check is skipped when its sequence is padded (None).
    pretrain_vlm.check_quantized_gemm_alignment(208, None, 1, fp8_recipe="delayed")
