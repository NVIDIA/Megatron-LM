# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
    ATTENTION_LAYERS = {ATTENTION, DS_ATTENTION, MLA}

    @classmethod
    def name_sorted_valid_layer_symbols(cls) -> list[str]:
        """Return valid layer symbols sorted by their public attribute names."""
        valid_layer_attrs = []
        for name, value in vars(cls).items():
            if not name.startswith('_') and value in cls.VALID_LAYERS:
                valid_layer_attrs.append((name, value))
        valid_layer_attrs.sort()
        return [value for (_, value) in valid_layer_attrs]


_LAYER_CONFIG_TYPES = (
    (Symbols.MAMBA, MambaLayerConfig),
    (Symbols.GDN, GDNLayerConfig),
    (Symbols.ATTENTION, AttentionLayerConfig),
    (Symbols.DS_ATTENTION, DSALayerConfig),
    (Symbols.MLA, MLALayerConfig),
    (Symbols.MLP, MLPLayerConfig),
    (Symbols.MOE, MoELayerConfig),
)


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
    for symbol, config_type in _LAYER_CONFIG_TYPES:
        if layer_symbol == symbol:
            return config_type.from_config(config)
    raise ValueError(f"Unexpected hybrid layer symbol: {layer_symbol}")


def get_layer_symbol_from_config(layer_config: TransformerConfig) -> str:
    """Return the canonical symbol for a layer config.

    Args:
        layer_config: Layer config whose hybrid symbol should be returned.

    Returns:
        The symbol corresponding to ``layer_config``.

    Raises:
        ValueError: If the exact config type is unsupported.
    """
    for symbol, config_type in _LAYER_CONFIG_TYPES:
        if type(layer_config) is config_type:
            return symbol
    raise ValueError(f"Unexpected hybrid layer config type: {type(layer_config).__name__}")


def validate_tp_comm_overlap(
    config: TransformerConfig, segment: str, has_mtp: bool = False
) -> None:
    """Validate TP communication overlap support for built-in hybrid layers.

    Args:
        config: Config whose TP communication overlap setting should be validated.
        segment: Layer symbols governed by ``config``.
        has_mtp: Whether this model instance will build an MTP block.

    Raises:
        ValueError: If TP communication overlap is enabled with MLA, DSA, or MTP.
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

    raise ValueError(
        "TP communication overlap is not supported with hybrid "
        f"{'/'.join(unsupported_features)} layers. Set tp_comm_overlap=False."
    )
