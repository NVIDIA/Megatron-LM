# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    CSALayerConfig,
)
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig


class Symbols:
    """Symbols for different layer types and pattern separators."""

    MAMBA = "M"
    GDN = 'G'
    ATTENTION = "*"
    DS_ATTENTION = "D"
    CSA = "C"  # DSv4 Compressed Sparse Attention (compress_ratio=4)
    HCA = "H"  # DSv4 Heavily Compressed Attention (compress_ratio=128)
    MLA = "+"
    WINDOW = "W"  # DSv4 sliding-window-only attention (compress_ratio=0)
    MLP = "-"
    MOE = 'E'
    PIPE = '|'
    MTP_SEPARATOR = "/"
    LAYER_CONFIG_MAP = {
        MAMBA: MambaLayerConfig,
        GDN: GDNLayerConfig,
        ATTENTION: AttentionLayerConfig,
        DS_ATTENTION: DSALayerConfig,
        CSA: CSALayerConfig,
        HCA: CSALayerConfig,
        MLA: MLALayerConfig,
        WINDOW: CSALayerConfig,
        MLP: MLPLayerConfig,
        MOE: MoELayerConfig,
    }
    DSV4_COMPRESS_RATIO_MAP = {CSA: 4, HCA: 128, WINDOW: 0}
    MLA_ATTENTION = {MLA, DS_ATTENTION, CSA, HCA, WINDOW}
    ATTENTION_LAYER_CONFIGS = {AttentionLayerConfig, DSALayerConfig, CSALayerConfig, MLALayerConfig}

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
    layer_config = Symbols.LAYER_CONFIG_MAP[layer_symbol].from_config(config)
    if type(layer_config) is CSALayerConfig:
        layer_config.compress_ratio = Symbols.DSV4_COMPRESS_RATIO_MAP[layer_symbol]
    return layer_config


def get_layer_symbol_from_config(layer_config: TransformerConfig) -> str:
    """Return the canonical symbol for a layer config.

    Args:
        layer_config: Layer config whose hybrid symbol should be returned.

    Returns:
        The symbol corresponding to ``layer_config``.

    Raises:
        ValueError: If the exact config type or DSv4 compression ratio is unsupported.
    """
    if type(layer_config) is CSALayerConfig:
        for symbol, compress_ratio in Symbols.DSV4_COMPRESS_RATIO_MAP.items():
            if layer_config.compress_ratio == compress_ratio:
                return symbol
        valid_ratios = sorted(Symbols.DSV4_COMPRESS_RATIO_MAP.values())
        raise ValueError(
            f"Unexpected CSALayerConfig compress_ratio: {layer_config.compress_ratio}. "
            f"Expected one of {valid_ratios}."
        )

    for symbol, config_type in Symbols.LAYER_CONFIG_MAP.items():
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
        ValueError: If TP communication overlap is enabled with MLA, DSA, DSv4 attention, or MTP.
    """
    unsupported_features: list[str] = []
    if Symbols.MLA in segment:
        unsupported_features.append("MLA")
    if Symbols.DS_ATTENTION in segment:
        unsupported_features.append("DSA")
    if any(symbol in segment for symbol in (Symbols.CSA, Symbols.HCA, Symbols.WINDOW)):
        unsupported_features.append("DSv4 attention")
    if has_mtp:
        unsupported_features.append("MTP")

    if not config.tp_comm_overlap or not unsupported_features:
        return

    raise ValueError(
        "TP communication overlap is not supported with hybrid "
        f"{'/'.join(unsupported_features)} layers. Set tp_comm_overlap=False."
    )
