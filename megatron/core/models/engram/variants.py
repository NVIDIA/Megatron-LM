# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Official conventions of each supported n-gram memory variant.

Engram-style n-gram memory exists in several published forms that share the hashing, the
tables and every distributed mechanism (row-sharded tables, lookup routing, gradient
delivery), and differ in a handful of conventions and in how the retrieved memory enters the
hidden states. Those differences are described once here instead of being rediscovered through
``variant == "..."`` comparisons at each use site, so adding a variant means adding one
descriptor rather than auditing the package.
"""

from __future__ import annotations

from dataclasses import dataclass

DEEPSEEK_VARIANT_NAME = "deepseek"
DEEPSEEK_V41_VARIANT_NAME = "deepseek_v41"
QWEN_VARIANT_NAME = "qwen"

GATED_CONV_FUSION = "gated_conv"
GATED_FUSION = "gated"


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
        fusion: How the retrieved memory enters the hidden states. ``"gated_conv"`` is the
            form of the published Engram and PLE modules: a key and a value projection, one
            group RMSNorm per role, and a causal short convolution over the gated value.
            ``"gated"`` is the DeepSeek-V4.1 form: one bias-free projection for the keys and
            the value, two per-channel gate weights, gate arithmetic in FP32, no convolution,
            and a closed gate on the positions after the last document boundary of a packed row.
        projection_bias: Dense value/key projections carry a bias term (``"gated_conv"`` only).
        zero_centered_gamma: Group RMSNorms store ``gamma - 1`` and scale by ``1 + weight``
            (``"gated_conv"`` only).
        boundary_token_flag: CLI flag that supplies the boundary token, used in errors.
    """

    name: str
    uses_tokenizer_artifact: bool
    resets_windows_at_boundary_token: bool
    fusion: str
    projection_bias: bool
    zero_centered_gamma: bool
    boundary_token_flag: str

    @property
    def has_short_convolution(self) -> bool:
        """Whether the fusion runs the causal short convolution."""
        return self.fusion == GATED_CONV_FUSION


DEEPSEEK_VARIANT = EngramVariant(
    name=DEEPSEEK_VARIANT_NAME,
    uses_tokenizer_artifact=True,
    resets_windows_at_boundary_token=False,
    fusion=GATED_CONV_FUSION,
    projection_bias=True,
    zero_centered_gamma=False,
    boundary_token_flag="engram_pad_token_id",
)

# The DeepSeek-V4.1 module hashes like the ``deepseek`` variant and differs in the fusion.
DEEPSEEK_V41_VARIANT = EngramVariant(
    name=DEEPSEEK_V41_VARIANT_NAME,
    uses_tokenizer_artifact=True,
    resets_windows_at_boundary_token=False,
    fusion=GATED_FUSION,
    projection_bias=False,
    zero_centered_gamma=False,
    boundary_token_flag="engram_pad_token_id",
)

QWEN_VARIANT = EngramVariant(
    name=QWEN_VARIANT_NAME,
    uses_tokenizer_artifact=False,
    resets_windows_at_boundary_token=True,
    fusion=GATED_CONV_FUSION,
    projection_bias=False,
    zero_centered_gamma=True,
    boundary_token_flag="engram_eos_token_id",
)

ENGRAM_VARIANTS = {
    variant.name: variant for variant in (DEEPSEEK_VARIANT, DEEPSEEK_V41_VARIANT, QWEN_VARIANT)
}


def resolve_variant(name: str) -> EngramVariant:
    """Return the variant descriptor for ``name``, or raise with the supported names."""
    try:
        return ENGRAM_VARIANTS[name]
    except KeyError:
        supported = ", ".join(repr(key) for key in ENGRAM_VARIANTS)
        raise ValueError(f"engram_variant must be one of {supported}; got {name!r}.") from None
