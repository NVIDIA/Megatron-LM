# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Sequence, TypeAlias, cast, get_args

from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig


class PipelineSplit:
    """Class sentinel marking a pipeline-stage boundary in a hybrid layer config list.

    The class itself, rather than an instance, is placed in the architecture list.
    """


class MTPSplit:
    """Class sentinel marking the start of an MTP depth in a hybrid layer config list.

    The class itself, rather than an instance, is placed in the architecture list.
    """


_SupportedLayerConfig: TypeAlias = (
    MambaLayerConfig
    | GDNLayerConfig
    | AttentionLayerConfig
    | DSALayerConfig
    | MLALayerConfig
    | MLPLayerConfig
    | MoELayerConfig
)
ArchitectureEntry: TypeAlias = _SupportedLayerConfig | type[PipelineSplit] | type[MTPSplit]

_SUPPORTED_LAYER_CONFIG_TYPES = get_args(_SupportedLayerConfig)


def validate_hybrid_layer_config_families(layer_config_list: Sequence[TransformerConfig]) -> None:
    """Reject layer-config families that cannot share one HybridStack."""

    layer_config_types = {type(layer_config) for layer_config in layer_config_list}
    if AttentionLayerConfig in layer_config_types and layer_config_types.intersection(
        {DSALayerConfig, MLALayerConfig}
    ):
        raise ValueError(
            "AttentionLayerConfig cannot be combined with MLA or DSA layer configs in one model."
        )


@dataclass(frozen=True)
class ArchitectureMetadata:
    """Structural metadata derived from a flat hybrid architecture list."""

    decoder_layer_count: int
    mtp_num_depths: int
    pipeline_split_indices: tuple[int, ...]
    mtp_split_indices: tuple[int, ...]
    pipeline_segment_count: int
    inferred_vpp_size: int | None


def scan_hybrid_layer_config_list(
    layer_config_list: Sequence[ArchitectureEntry], *, pp_size: int = 1
) -> ArchitectureMetadata:
    """Validate a flat hybrid architecture and return structural metadata.

    Decoder configs precede the first :class:`MTPSplit`. Each ``MTPSplit`` starts one
    MTP prediction depth, and all depths must reuse the first depth's config objects in
    the same order. ``PipelineSplit`` markers are only valid in the decoder prefix.

    Args:
        layer_config_list: Flat sequence of supported per-layer configs and split classes.
        pp_size: Pipeline-model-parallel size used to validate explicit pipeline segments.

    Returns:
        Counts and raw marker positions without splitting or copying the architecture list.

    Raises:
        ValueError: If an entry or marker placement is invalid, an MTP depth is empty,
            or MTP depths do not reuse the same configs in the same order.
    """

    if pp_size < 1:
        raise ValueError(f"pp_size must be positive, got {pp_size}.")
    if not layer_config_list:
        raise ValueError(
            "layer_config_list must not be empty; use a leading MTPSplit for a zero-decoder model."
        )

    pipeline_split_indices: list[int] = []
    mtp_split_indices: list[int] = []
    layer_configs: list[TransformerConfig] = []
    seen_mtp = False

    for index, entry in enumerate(layer_config_list):
        if entry is PipelineSplit:
            if seen_mtp:
                raise ValueError(
                    f"PipelineSplit at index {index} is invalid after the first MTPSplit."
                )
            pipeline_split_indices.append(index)
        elif entry is MTPSplit:
            seen_mtp = True
            mtp_split_indices.append(index)
        elif type(entry) not in _SUPPORTED_LAYER_CONFIG_TYPES:
            raise ValueError(
                f"Invalid hybrid layer config entry at index {index}: {entry!r}. "
                "Expected a supported TransformerConfig, PipelineSplit, or MTPSplit."
            )
        else:
            layer_configs.append(cast(TransformerConfig, entry))

    validate_hybrid_layer_config_families(layer_configs)

    decoder_end = mtp_split_indices[0] if mtp_split_indices else len(layer_config_list)
    decoder_layer_count = sum(
        entry is not PipelineSplit for entry in layer_config_list[:decoder_end]
    )
    pipeline_segment_count = len(pipeline_split_indices) + 1
    inferred_vpp_size = None
    if pipeline_split_indices:
        segment_boundaries = (-1, *pipeline_split_indices, decoder_end)
        for segment_index, (left, right) in enumerate(
            zip(segment_boundaries, segment_boundaries[1:]), start=1
        ):
            if right == left + 1:
                raise ValueError(
                    f"Pipeline segment {segment_index} is empty; PipelineSplit must separate "
                    "non-empty decoder segments."
                )
        if pipeline_segment_count % pp_size != 0:
            raise ValueError(
                f"The {pipeline_segment_count} PipelineSplit segments are not evenly divisible "
                f"by pp_size={pp_size}."
            )
        inferred_vpp_size = pipeline_segment_count // pp_size

    if mtp_split_indices:
        first_start = mtp_split_indices[0] + 1
        first_end = mtp_split_indices[1] if len(mtp_split_indices) > 1 else len(layer_config_list)
        first_head = layer_config_list[first_start:first_end]
        if not first_head:
            raise ValueError("MTP depth 1 is empty; MTPSplit must be followed by layer configs.")

        for depth_index, split_index in enumerate(mtp_split_indices[1:], start=2):
            depth_end = (
                mtp_split_indices[depth_index]
                if depth_index < len(mtp_split_indices)
                else len(layer_config_list)
            )
            head = layer_config_list[split_index + 1 : depth_end]
            if not head:
                raise ValueError(
                    f"MTP depth {depth_index} is empty; MTPSplit must be followed by layer configs."
                )
            if len(head) != len(first_head) or any(
                config is not first_config
                for config, first_config in zip(head, first_head, strict=True)
            ):
                raise ValueError(
                    "All MTP depths must reuse the same layer config objects in the same order."
                )

    return ArchitectureMetadata(
        decoder_layer_count=decoder_layer_count,
        mtp_num_depths=len(mtp_split_indices),
        pipeline_split_indices=tuple(pipeline_split_indices),
        mtp_split_indices=tuple(mtp_split_indices),
        pipeline_segment_count=pipeline_segment_count,
        inferred_vpp_size=inferred_vpp_size,
    )


__all__ = [
    "ArchitectureEntry",
    "ArchitectureMetadata",
    "MTPSplit",
    "PipelineSplit",
    "scan_hybrid_layer_config_list",
    "validate_hybrid_layer_config_families",
]
