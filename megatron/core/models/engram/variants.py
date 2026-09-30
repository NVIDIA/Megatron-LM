# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Official conventions of each supported n-gram memory variant.

The original DeepSeek Engram, Qwen PLE, and DeepSeek-V4.1 variants share table
allocation and EP lookup. These descriptors select their hash, projection,
normalization, convolution, and gate conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

DEEPSEEK_VARIANT_NAME = "deepseek"
QWEN_VARIANT_NAME = "qwen"
DEEPSEEK_V41_VARIANT_NAME = "deepseek_v41"


@dataclass(frozen=True)
class EngramVariant:
    """Immutable description of one n-gram memory variant.

    Args:
        name: Value accepted by ``--engram-variant``.
        uses_tokenizer_artifact: Hash constants and the raw-to-hash token map come from a
            versioned offline artifact rather than being generated in process. Variants
            without an artifact hash raw token IDs directly.
        resets_windows_at_boundary_token: Suffix windows restart at every occurrence of the
            boundary token, so n-grams never span two documents. Variants without it only
            pad the window at the start of the sequence.
        projection_bias: Dense value/key projections carry a bias term.
        zero_centered_gamma: Group RMSNorms store ``gamma - 1`` and scale by ``1 + weight``.
        boundary_token_flag: CLI flag that supplies the boundary token, used in errors.
        short_convolution: Apply the original causal convolution after gating.
        joint_key_value: Store keys and the shared value in one bias-free projection.
        fp32_gate: Evaluate the V4.1 gate and residual update in FP32.
    """

    name: str
    uses_tokenizer_artifact: bool
    resets_windows_at_boundary_token: bool
    projection_bias: bool
    zero_centered_gamma: bool
    boundary_token_flag: str
    short_convolution: bool = True
    joint_key_value: bool = False
    fp32_gate: bool = False


DEEPSEEK_VARIANT = EngramVariant(
    name=DEEPSEEK_VARIANT_NAME,
    uses_tokenizer_artifact=True,
    resets_windows_at_boundary_token=False,
    projection_bias=True,
    zero_centered_gamma=False,
    boundary_token_flag="engram_pad_token_id",
)

QWEN_VARIANT = EngramVariant(
    name=QWEN_VARIANT_NAME,
    uses_tokenizer_artifact=False,
    resets_windows_at_boundary_token=True,
    projection_bias=False,
    zero_centered_gamma=True,
    boundary_token_flag="engram_eos_token_id",
)

DEEPSEEK_V41_VARIANT = EngramVariant(
    name=DEEPSEEK_V41_VARIANT_NAME,
    uses_tokenizer_artifact=True,
    resets_windows_at_boundary_token=False,
    projection_bias=False,
    zero_centered_gamma=False,
    boundary_token_flag="engram_pad_token_id",
    short_convolution=False,
    joint_key_value=True,
    fp32_gate=True,
)

ENGRAM_VARIANTS = {
    variant.name: variant for variant in (DEEPSEEK_VARIANT, QWEN_VARIANT, DEEPSEEK_V41_VARIANT)
}


def resolve_variant(name: str) -> EngramVariant:
    """Return the variant descriptor for ``name``, or raise with the supported names."""
    try:
        return ENGRAM_VARIANTS[name]
    except KeyError:
        supported = ", ".join(repr(key) for key in ENGRAM_VARIANTS)
        raise ValueError(f"engram_variant must be one of {supported}; got {name!r}.") from None
