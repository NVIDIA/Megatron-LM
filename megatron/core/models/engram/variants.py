# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Official conventions of each supported n-gram memory variant.

Engram-style n-gram memory exists in two published forms that share every distributed
mechanism (EP-sharded tables, all-to-all routing, gating, causal convolution) and differ only
in a handful of conventions. Those differences are described once here instead of being
rediscovered through ``variant == "..."`` comparisons at each use site, so adding a third
variant means adding one descriptor rather than auditing the package.
"""

from __future__ import annotations

from dataclasses import dataclass

DEEPSEEK_VARIANT_NAME = "deepseek"
QWEN_VARIANT_NAME = "qwen"


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
    """

    name: str
    uses_tokenizer_artifact: bool
    resets_windows_at_boundary_token: bool
    projection_bias: bool
    zero_centered_gamma: bool
    boundary_token_flag: str

    @property
    def supports_packed_sequences(self) -> bool:
        """Whether packed (THD) rows are meaningful for this variant.

        Packed rows concatenate several documents into one row, so they are only safe when
        the hash windows reset at the document boundaries inside that row.
        """
        return self.resets_windows_at_boundary_token


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

ENGRAM_VARIANTS = {variant.name: variant for variant in (DEEPSEEK_VARIANT, QWEN_VARIANT)}


def resolve_variant(name: str) -> EngramVariant:
    """Return the variant descriptor for ``name``, or raise with the supported names."""
    try:
        return ENGRAM_VARIANTS[name]
    except KeyError:
        supported = ", ".join(repr(key) for key in ENGRAM_VARIANTS)
        raise ValueError(f"engram_variant must be one of {supported}; got {name!r}.") from None
