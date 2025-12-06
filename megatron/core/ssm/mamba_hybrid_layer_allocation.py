# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Dict, List, Tuple

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_world_size,
)
from megatron.core.utils import log_on_each_pipeline_stage


logger = logging.getLogger(__name__)


class Symbols:
    """Symbols for different layer types."""

    MAMBA = "M"
    ATTENTION = "*"
    MLP = "-"
    MOE = 'E'
    PIPE = '|'
    VALID = {MAMBA, ATTENTION, MLP, MOE, PIPE}


def get_hybrid_total_layer_count(layer_pattern: str) -> int:
    """Returns the total number of layers in the hybrid layer pattern"""
    pattern = layer_pattern.replace(Symbols.PIPE, '')
    return len(pattern)


def get_hybrid_total_pipeline_segment_count(layer_pattern: str) -> int:
    """Returns the number of pipeline segments in a layer pattern"""
    return layer_pattern.count(Symbols.PIPE) + 1


def allocate_layers(layer_pattern: str, vp_stage: int = 0) -> tuple[list, int]:
    """
    Allocates layers for the current (virtual) pipeline parallel segment

    Returns:
        list: list of symbols, one for each layer
        int: layer number offset for this model segment
    """
    assert layer_pattern is not None, '--hybrid-layer-pattern must be specified for hybrid models'
    segments = layer_pattern.split(Symbols.PIPE)

    # TODO(duncan): Handle parallel_state.is_inside_encoder() later (for merge to main)
    pp_rank = get_pipeline_model_parallel_rank()
    pp_size = get_pipeline_model_parallel_world_size()
    vp_size = get_virtual_pipeline_model_parallel_world_size()
    if vp_size is not None:
        vp_relative_rank = vp_stage  # get_virtual_pipeline_model_parallel_rank()
    else:
        vp_relative_rank = 0
    vp_global_rank = vp_relative_rank * pp_size + pp_rank

    layer_offset = sum(len(segments[i]) for i in range(vp_global_rank))
    segment = segments[vp_global_rank]
    layer_type_list = list(segment)

    for l in layer_type_list:
        if l not in Symbols.VALID:
            raise ValueError(f"In hybrid layer pattern, '{l}' is not " f"one of {Symbols.VALID}")

    log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        f"pp_rank: {pp_rank}, vp_relative_rank: {vp_relative_rank}, layer pattern: {segment}",
    )

    return layer_type_list, layer_offset
