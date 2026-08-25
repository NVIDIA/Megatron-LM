# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
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


def create_layer_config(config: TransformerConfig, layer_symbol: str) -> TransformerConfig:
    """Create a layer-specific config from a normalized stack-level config.

    Args:
        config: Normalized stack-level config to copy.
        layer_symbol: Symbol identifying the layer config type to create.

    Returns:
        An independent config of the type corresponding to ``layer_symbol``.

    Raises:
        ValueError: If ``layer_symbol`` does not identify a supported hybrid layer.
    """
    if layer_symbol == Symbols.MAMBA:
        return MambaLayerConfig.from_config(config)
    if layer_symbol == Symbols.GDN:
        return GDNLayerConfig.from_config(config)
    if layer_symbol == Symbols.ATTENTION:
        return AttentionLayerConfig.from_config(config)
    if layer_symbol == Symbols.DS_ATTENTION:
        return DSALayerConfig.from_config(config)
    if layer_symbol == Symbols.MLA:
        return MLALayerConfig.from_config(config)
    if layer_symbol == Symbols.MLP:
        return MLPLayerConfig.from_config(config)
    if layer_symbol == Symbols.MOE:
        return MoELayerConfig.from_config(config)
    raise ValueError(f"Unexpected hybrid layer symbol: {layer_symbol}")


def get_layer_symbol_from_config(layer_config: TransformerConfig) -> str:
    """Return the canonical symbol for a layer config, including subclasses.

    Args:
        layer_config: Layer config whose hybrid symbol should be returned.

    Returns:
        The symbol corresponding to ``layer_config``.

    Raises:
        ValueError: If the config type is unsupported or matches multiple layer types.
    """
    matching_symbols: list[str] = []
    if isinstance(layer_config, MambaLayerConfig):
        matching_symbols.append(Symbols.MAMBA)
    if isinstance(layer_config, GDNLayerConfig):
        matching_symbols.append(Symbols.GDN)
    if isinstance(layer_config, AttentionLayerConfig):
        matching_symbols.append(Symbols.ATTENTION)
    if isinstance(layer_config, DSALayerConfig):
        matching_symbols.append(Symbols.DS_ATTENTION)
    if isinstance(layer_config, MLALayerConfig):
        matching_symbols.append(Symbols.MLA)
    if isinstance(layer_config, MLPLayerConfig):
        matching_symbols.append(Symbols.MLP)
    if isinstance(layer_config, MoELayerConfig):
        matching_symbols.append(Symbols.MOE)
    if not matching_symbols:
        raise ValueError(f"Unexpected hybrid layer config type: {type(layer_config).__name__}")
    if len(matching_symbols) > 1:
        raise ValueError(
            f"Ambiguous hybrid layer config type: {type(layer_config).__name__} "
            f"matches symbols {matching_symbols}"
        )
    return matching_symbols[0]


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
