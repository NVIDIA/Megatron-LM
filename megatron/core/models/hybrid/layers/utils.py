# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections.abc import Callable, Sequence

from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    CSALayerConfig,
)
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig


class Symbols:
    """Symbols for different layer types and pattern separators."""

    MAMBA = "M"
    GDN = 'G'
    KDA = 'K'
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
        KDA: KDALayerConfig,
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


def get_custom_layer_type(layer_config: object) -> type[MegatronModule] | None:
    """Resolve a config's direct implementation reference without a registry or imports."""
    if not isinstance(layer_config, TransformerConfig):
        return None
    layer_type = getattr(layer_config, "layer_type", None)
    if layer_type is None:
        return None
    if not isinstance(layer_type, type) or not issubclass(layer_type, MegatronModule):
        raise TypeError("layer_type must be a MegatronModule subclass.")
    return layer_type


def uses_shared_rotary_embeddings(layer_config: TransformerConfig) -> bool:
    """Return whether a layer consumes HybridModel's shared position embeddings."""
    layer_type = get_custom_layer_type(layer_config)
    if layer_type is not None:
        return getattr(layer_type, "uses_shared_rotary_embeddings", False)
    return type(layer_config) is AttentionLayerConfig


def is_moe_layer_config(layer_config: TransformerConfig) -> bool:
    """Identify configs participating in the standard MoE tracker contract."""
    layer_type = get_custom_layer_type(layer_config)
    if layer_type is not None:
        return getattr(layer_type, "is_moe_layer", False)
    return type(layer_config) is MoELayerConfig


def _validate_custom_layer_execution(
    layer_config: TransformerConfig,
    stack_config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
) -> None:
    """Enforce Core's custom-layer integration limits without changing either config.

    HybridModel checks the complete architecture before construction. HybridStack
    checks physical configs too, including standalone stacks and nested MTP stacks.
    Layer-specific requirements belong to the layer's construction code.
    """
    unsupported = set()
    for config in (stack_config, layer_config):
        if config.tensor_model_parallel_size != 1:
            unsupported.add("tensor parallelism")
        if config.context_parallel_size != 1:
            unsupported.add("context parallelism")
        for field_name in (
            "sequence_parallel",
            "tp_comm_overlap",
            "fp8",
            "fp4",
            "quant_recipe",
            "enable_mhc_connections",
            "wide_residual",
            "moe_shortcut_connection",
            "overlap_moe_expert_parallel_comm",
            "cpu_offloading",
            "fine_grained_activation_offloading",
        ):
            if getattr(config, field_name, None):
                unsupported.add(field_name)
        if config.cuda_graph_impl != "none":
            unsupported.add("CUDA graphs")
        if config.recompute_granularity == "selective":
            unsupported.add("selective recomputation")
        if is_moe_layer_config(layer_config) and config.moe_num_hash_layers:
            unsupported.add("custom hash routing")
    for group_name in ("tp", "cp"):
        group = getattr(pg_collection, group_name, None)
        if group is None:
            raise ValueError(f"Custom hybrid layers require pg_collection.{group_name}.")
        if group.size() != 1:
            unsupported.add(f"{group_name.upper()} process group size > 1")
    if layer_config.hidden_size != stack_config.hidden_size:
        raise ValueError("Custom hybrid layers must preserve the stack's hidden_size.")
    if unsupported:
        raise ValueError(
            f"{type(layer_config).__name__} uses unsupported custom-layer execution features: "
            + ", ".join(sorted(unsupported))
        )


def invoke_custom_layer(
    layer: MegatronModule,
    hidden_states: Tensor,
    *,
    attention_mask: Tensor | None = None,
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
    packed_seq_params: PackedSeqParams | None = None,
    padding_mask: Tensor | None = None,
    sequence_len_offset: Tensor | None = None,
    input_ids: Tensor | None = None,
    inference_context=None,
) -> Tensor:
    """Apply the custom-layer calling convention in eager and recomputed execution."""
    if inference_context is not None:
        raise ValueError("Custom hybrid layers do not support inference contexts yet.")
    output = layer(
        hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        sequence_len_offset=sequence_len_offset,
        input_ids=input_ids,
    )
    if not isinstance(output, Tensor) or output.shape != hidden_states.shape:
        raise ValueError("A custom hybrid layer must return a tensor with the input shape.")
    return output


def validate_shared_rotary_config(
    layer_configs: Sequence[TransformerConfig], kv_channels: int, position_embedding_type: str
) -> None:
    """Require standard attention layers to match the model's shared RoPE width.

    MLA, DSA, and CSA construct their own rotary embeddings and are not constrained here.
    """
    if position_embedding_type not in ("rope", "yarn"):
        return
    for layer_config in layer_configs:
        if uses_shared_rotary_embeddings(layer_config) and layer_config.kv_channels != kv_channels:
            raise ValueError(
                f"{type(layer_config).__name__}.kv_channels "
                f"({layer_config.kv_channels}) must match "
                f"HybridModel config.kv_channels ({kv_channels}) when using "
                f"position_embedding_type={position_embedding_type!r}: decoder and MTP "
                "attention layers use the model's shared rotary embeddings."
            )


def get_attention_variant_loss_scale_func(
    layer_configs: Sequence[TransformerConfig],
) -> Callable[[Tensor], None] | None:
    """Return the scheduler loss-scale hook required by the architecture's indexers."""
    if any(
        type(layer_config) in (DSALayerConfig, CSALayerConfig) for layer_config in layer_configs
    ):
        from megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexerLossAutoScaler,
        )

        return DSAIndexerLossAutoScaler.set_loss_scale
    return None


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
