# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Sequence, TypeAlias

import torch

from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size, log_on_each_pipeline_stage

logger = logging.getLogger(__name__)


class PipelineSplit:
    """Class sentinel marking a pipeline-stage boundary in a hybrid layer config list.

    The class itself, rather than an instance, is placed in the architecture list.
    """


class MTPSplit:
    """Class sentinel marking the start of an MTP depth in a hybrid layer config list.

    The class itself, rather than an instance, is placed in the architecture list.
    """


ArchitectureEntry: TypeAlias = TransformerConfig | type[PipelineSplit] | type[MTPSplit]


@dataclass(frozen=True)
class ArchitectureMetadata:
    """Structural metadata derived from a flat hybrid architecture list."""

    decoder_layer_count: int
    pipeline_split_indices: tuple[int, ...]
    mtp_split_indices: tuple[int, ...]
    inferred_vpp_size: int | None

    @property
    def mtp_num_depths(self) -> int:
        """Number of logical MTP depths."""

        return len(self.mtp_split_indices)

    @property
    def pipeline_segment_count(self) -> int:
        """Number of explicitly defined decoder pipeline segments."""

        return len(self.pipeline_split_indices) + 1


def scan_hybrid_layer_config_list(
    layer_config_list: Sequence[ArchitectureEntry], pp_size: int = 1
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
    layer_config_types: set[type[TransformerConfig]] = set()
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
        elif type(entry) not in layer_utils.Symbols.LAYER_CONFIG_MAP.values():
            raise ValueError(
                f"Invalid hybrid layer config entry at index {index}: {entry!r}. "
                "Expected a supported TransformerConfig, PipelineSplit, or MTPSplit."
            )
        else:
            layer_config_types.add(type(entry))

    layer_utils.validate_layer_config_types(layer_config_types)

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
        pipeline_split_indices=tuple(pipeline_split_indices),
        mtp_split_indices=tuple(mtp_split_indices),
        inferred_vpp_size=inferred_vpp_size,
    )


def _normalize_vp_stage(vp_stage: int | None, vp_size: int) -> int:
    """Validate and normalize the virtual-pipeline stage index."""

    if vp_stage is None:
        if vp_size > 1:
            raise ValueError(
                "vp_stage must be provided when virtual_pipeline_model_parallel_size is set."
            )
        return 0
    if not 0 <= vp_stage < vp_size:
        raise ValueError(f"vp_stage must be in [0, {vp_size}), got {vp_stage}.")
    return vp_stage


def select_pipeline_config_segment(
    decoder_entries: Sequence[TransformerConfig | type[PipelineSplit]],
    config: TransformerConfig,
    pp_group: torch.distributed.ProcessGroup | None,
    vp_stage: int | None,
    tp_group: torch.distributed.ProcessGroup | None = None,
    dp_cp_group: torch.distributed.ProcessGroup | None = None,
) -> tuple[list[TransformerConfig], int]:
    """Select this PP/VPP rank's decoder configs from a list architecture."""

    pp_rank = get_pg_rank(pp_group)
    pp_size = get_pg_size(pp_group)
    if config.pipeline_model_parallel_size != pp_size:
        raise ValueError(
            f"config.pipeline_model_parallel_size is {config.pipeline_model_parallel_size}, "
            f"but the model pipeline process group has size {pp_size}."
        )
    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size < 1
    ):
        raise ValueError(
            "config.virtual_pipeline_model_parallel_size must be positive when set, got "
            f"{config.virtual_pipeline_model_parallel_size}."
        )
    if any(entry is MTPSplit for entry in decoder_entries):
        raise ValueError("decoder_entries must not contain MTPSplit markers.")

    architecture_metadata = (
        scan_hybrid_layer_config_list(decoder_entries, pp_size=pp_size) if decoder_entries else None
    )
    global_layer_config_list = [
        entry for entry in decoder_entries if isinstance(entry, TransformerConfig)
    ]
    decoder_layer_count = (
        architecture_metadata.decoder_layer_count if architecture_metadata is not None else 0
    )
    if decoder_layer_count != config.num_layers:
        raise ValueError(
            f"layer_config_list defines {decoder_layer_count} decoder layers, "
            f"but config.num_layers is {config.num_layers}."
        )

    topology_conflicts = []
    if config.pipeline_model_parallel_layout is not None:
        topology_conflicts.append("pipeline_model_parallel_layout")
    has_pipeline_splits = bool(
        architecture_metadata is not None and architecture_metadata.pipeline_split_indices
    )
    if has_pipeline_splits:
        if config.num_layers_in_first_pipeline_stage is not None:
            topology_conflicts.append("num_layers_in_first_pipeline_stage")
        if config.num_layers_in_last_pipeline_stage is not None:
            topology_conflicts.append("num_layers_in_last_pipeline_stage")
    if config.account_for_embedding_in_pipeline_split:
        topology_conflicts.append("account_for_embedding_in_pipeline_split")
    if config.account_for_loss_in_pipeline_split:
        topology_conflicts.append("account_for_loss_in_pipeline_split")
    if topology_conflicts:
        raise ValueError(
            "layer_config_list cannot be combined with pipeline topology controls: "
            + ", ".join(topology_conflicts)
            + "."
        )

    if has_pipeline_splits:
        assert architecture_metadata is not None
        segments: list[list[TransformerConfig]] = [[]]
        for entry in decoder_entries:
            if entry is PipelineSplit:
                segments.append([])
            else:
                assert isinstance(entry, TransformerConfig)
                segments[-1].append(entry)

        assert architecture_metadata.inferred_vpp_size is not None
        vp_size = architecture_metadata.inferred_vpp_size
        configured_vpp_size = config.virtual_pipeline_model_parallel_size or 1
        if configured_vpp_size != vp_size:
            raise ValueError(
                f"PipelineSplit infers virtual pipeline size {vp_size}, but "
                f"config.virtual_pipeline_model_parallel_size is "
                f"{config.virtual_pipeline_model_parallel_size}."
            )
        vp_rank = _normalize_vp_stage(vp_stage, vp_size)
        segment_index = vp_rank * pp_size + pp_rank
        layer_offset = sum(len(segment) for segment in segments[:segment_index])
        selected = segments[segment_index]
        segment_log = f", segment_index={segment_index}/{len(segments)}"
    else:
        vp_size = config.virtual_pipeline_model_parallel_size or 1
        vp_rank = _normalize_vp_stage(vp_stage, vp_size)
        if pp_size == 1 and vp_size > 1:
            if len(global_layer_config_list) % vp_size != 0:
                raise ValueError(
                    f"The {len(global_layer_config_list)} decoder configs in layer_config_list "
                    f"must be evenly divisible across VPP={vp_size}."
                )
            chunk_layer_counts = [len(global_layer_config_list) // vp_size] * vp_size
        else:
            from megatron.core.transformer.transformer_block import get_num_layers_to_build

            try:
                chunk_layer_counts = [
                    get_num_layers_to_build(config, vp_stage=chunk_vp_rank, pp_rank=chunk_pp_rank)
                    for chunk_vp_rank in range(vp_size)
                    for chunk_pp_rank in range(pp_size)
                ]
            except AssertionError as error:
                raise ValueError(str(error)) from error

        if any(layer_count < 0 for layer_count in chunk_layer_counts):
            raise ValueError(
                "Pipeline allocation produced a negative decoder layer count; check "
                "num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage."
            )
        if sum(chunk_layer_counts) != len(global_layer_config_list):
            raise ValueError(
                f"Pipeline allocation owns {sum(chunk_layer_counts)} decoder layers, but "
                f"layer_config_list defines {len(global_layer_config_list)}."
            )
        segment_index = vp_rank * pp_size + pp_rank
        num_layers_to_build = chunk_layer_counts[segment_index]
        layer_offset = sum(chunk_layer_counts[:segment_index])
        selected = global_layer_config_list[layer_offset : layer_offset + num_layers_to_build]
        segment_log = ""

    log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        f"HybridModel: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_rank}, "
        f"num_layers={len(selected)}, layer_offset={layer_offset}"
        f"{segment_log}",
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )
    return selected, layer_offset
