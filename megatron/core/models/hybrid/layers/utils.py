# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections.abc import Sequence

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
    LAYER_CONFIG_MAP = {
        MAMBA: MambaLayerConfig,
        GDN: GDNLayerConfig,
        ATTENTION: AttentionLayerConfig,
        DS_ATTENTION: DSALayerConfig,
        MLA: MLALayerConfig,
        MLP: MLPLayerConfig,
        MOE: MoELayerConfig,
    }
    ATTENTION_LAYER_CONFIGS = {AttentionLayerConfig, DSALayerConfig, MLALayerConfig}

    @classmethod
    def name_sorted_valid_layer_symbols(cls) -> list[str]:
        """Return valid layer symbols sorted by their public attribute names."""
        valid_layer_attrs = []
        for name, value in vars(cls).items():
            if not name.startswith('_') and isinstance(value, str) and is_valid_symbol(value):
                valid_layer_attrs.append((name, value))
        valid_layer_attrs.sort()
        return [value for (_, value) in valid_layer_attrs]


def is_valid_symbol(layer_symbol: str, allow_pipe: bool = False) -> bool:
    """Return whether ``layer_symbol`` identifies a supported layer or allowed pipe.

    Args:
        layer_symbol: Symbol to validate.
        allow_pipe: Whether to also accept the pipeline separator symbol.
    """
    return layer_symbol in Symbols.LAYER_CONFIG_MAP or (allow_pipe and layer_symbol == Symbols.PIPE)


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
    if not is_valid_symbol(layer_symbol):
        raise ValueError(f"Unexpected hybrid layer symbol: {layer_symbol}")
    return Symbols.LAYER_CONFIG_MAP[layer_symbol].from_config(config)


def get_layer_symbol_from_config(layer_config: TransformerConfig) -> str:
    """Return the canonical symbol for a layer config.

    Args:
        layer_config: Layer config whose hybrid symbol should be returned.

    Returns:
        The symbol corresponding to ``layer_config``.

    Raises:
        ValueError: If the exact config type is unsupported.
    """
    for symbol, config_type in Symbols.LAYER_CONFIG_MAP.items():
        if type(layer_config) is config_type:
            return symbol
    raise ValueError(f"Unexpected hybrid layer config type: {type(layer_config).__name__}")


def validate_tp_comm_overlap(
    config: TransformerConfig, layers: str | Sequence[TransformerConfig], has_mtp: bool = False
) -> None:
    """Validate TP communication overlap support for built-in hybrid layers.

    Args:
        config: Config whose TP communication overlap setting should be validated.
        layers: Layer configs governed by ``config``. A string is also accepted by the
            legacy pattern adapter.
        has_mtp: Whether this model instance will build an MTP block.

    Raises:
        ValueError: If TP communication overlap is enabled with MLA, DSA, or MTP.
    """
    unsupported_features: list[str] = []
    if isinstance(layers, str):
        overlap_enabled = config.tp_comm_overlap
        has_mla = overlap_enabled and Symbols.MLA in layers
        has_dsa = overlap_enabled and Symbols.DS_ATTENTION in layers
        has_unsupported_mtp = overlap_enabled and has_mtp
    else:
        overlap_configs = [layer_config for layer_config in layers if layer_config.tp_comm_overlap]
        layer_types = {type(layer_config) for layer_config in overlap_configs}
        has_mla = MLALayerConfig in layer_types
        has_dsa = DSALayerConfig in layer_types
        has_unsupported_mtp = has_mtp and (config.tp_comm_overlap or bool(overlap_configs))

    if has_mla:
        unsupported_features.append("MLA")
    if has_dsa:
        unsupported_features.append("DSA")
    if has_unsupported_mtp:
        unsupported_features.append("MTP")

    if not unsupported_features:
        return

    raise ValueError(
        "TP communication overlap is not supported with hybrid "
        f"{'/'.join(unsupported_features)} layers. Set tp_comm_overlap=False."
    )
