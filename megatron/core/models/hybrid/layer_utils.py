# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

from megatron.core.transformer.transformer_config import TransformerConfig


class Symbols:
    """Symbols for different layer types and pattern separators."""

    MAMBA = "M"
    GDN = 'G'
    ATTENTION = "*"
    DS_ATTENTION = "D"
    MLA = "+"
    MLP = "-"
    MOE = 'E'
    PIPE = '|'
    MTP_SEPARATOR = "/"
    VALID_LAYERS = {MAMBA, GDN, ATTENTION, DS_ATTENTION, MLA, MLP, MOE}

    @classmethod
    def name_sorted_valid_layer_symbols(cls) -> list[str]:
        """Return valid layer symbols sorted by their public attribute names."""
        valid_layer_attrs = []
        for name, value in vars(cls).items():
            if not name.startswith('_') and value in cls.VALID_LAYERS:
                valid_layer_attrs.append((name, value))
        valid_layer_attrs.sort()
        return [value for (_, value) in valid_layer_attrs]


def normalize_tp_comm_overlap(
    config: TransformerConfig, segment: str, has_mtp: bool = False
) -> None:
    """Disable TP communication overlap unsupported by built-in hybrid layers.

    This must run before ``validate_segment_layers`` copies the stack-level config so
    every generated layer config receives the normalized value.

    Args:
        config: Stack-level config that will be copied for each layer.
        segment: Selected pipeline segment, containing only layer symbols.
        has_mtp: Whether this model instance will build an MTP block.
    """
    unsupported_features: list[str] = []
    if Symbols.MLA in segment:
        unsupported_features.append("MLA")
    if Symbols.DS_ATTENTION in segment:
        unsupported_features.append("DSA")
    if has_mtp:
        unsupported_features.append("MTP")

    if not config.tp_comm_overlap or not unsupported_features:
        return

    config.tp_comm_overlap = False
    warnings.warn(
        "TP communication overlap is not supported with hybrid "
        f"{'/'.join(unsupported_features)} layers. Disabling tp_comm_overlap.",
        stacklevel=2,
    )
