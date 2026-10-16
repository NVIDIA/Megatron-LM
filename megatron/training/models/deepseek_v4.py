# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek-V4-specific training configuration normalization."""

from argparse import Namespace
from typing import Any


def normalize_dsv4_hybrid_csa_compress_ratios(
    args: Namespace, config_kwargs: dict[str, Any], pattern: str
) -> None:
    """Normalize compact DSv4 HybridModel ratios into a per-layer config list."""
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    variant = config_kwargs.get(
        "experimental_attention_variant", getattr(args, "experimental_attention_variant", None)
    )
    if variant != "dsv4_hybrid":
        return

    fixed_ratio_map = Symbols.DSV4_COMPRESS_RATIO_MAP
    ratio_symbols = set(fixed_ratio_map)
    sections = pattern.split(Symbols.MTP_SEPARATOR)
    layers = "".join(section.replace(Symbols.PIPE, "") for section in sections)
    attention_symbols = [symbol for symbol in layers if symbol in ratio_symbols]
    if not attention_symbols:
        return
    compact_len = len(attention_symbols)
    full_len = len(layers)

    def compact_to_full(provided: list[int]) -> list[int]:
        full = []
        compact_iter = iter(provided)
        for symbol in layers:
            if symbol in ratio_symbols:
                ratio = next(compact_iter)
                expected = fixed_ratio_map[symbol]
                assert ratio == expected, (
                    f"csa_compress_ratios has ratio {ratio} for hybrid symbol "
                    f"'{symbol}', expected {expected}."
                )
                full.append(ratio)
            else:
                full.append(0)
        return full

    provided_ratios = getattr(args, "csa_compress_ratios", None)
    if provided_ratios is None:
        compact_ratios = [fixed_ratio_map[symbol] for symbol in attention_symbols]
        full_ratios = compact_to_full(compact_ratios)
    else:
        provided = list(provided_ratios)
        if len(provided) == compact_len:
            full_ratios = compact_to_full(provided)
        elif len(provided) == full_len:
            for ratio, symbol in zip(provided, layers):
                if symbol in ratio_symbols:
                    expected = fixed_ratio_map[symbol]
                    assert ratio == expected, (
                        f"csa_compress_ratios has ratio {ratio} for hybrid symbol "
                        f"'{symbol}', expected {expected}."
                    )
                else:
                    assert ratio == 0, (
                        "csa_compress_ratios should not pad non-DSv4 hybrid symbol "
                        f"'{symbol}' with non-zero ratio {ratio}."
                    )
            full_ratios = provided
        else:
            raise AssertionError(
                f"csa_compress_ratios length ({len(provided)}) must equal either the "
                f"number of W/C/H attention symbols ({compact_len}) or the legacy "
                f"number of all layers in the hybrid pattern ({full_len}) for pattern "
                f"'{pattern}'."
            )

    args.csa_compress_ratios = full_ratios
    config_kwargs["csa_compress_ratios"] = list(full_ratios)
