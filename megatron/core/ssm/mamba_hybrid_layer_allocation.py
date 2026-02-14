# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from megatron.core.utils import log_on_each_pipeline_stage

logger = logging.getLogger(__name__)


class Symbols:
    """Symbols for different layer types and pattern separators."""

    MAMBA = "M"
    ATTENTION = "*"
    MLP = "-"
    MOE = 'E'
    PIPE = '|'
    MTP_SEPARATOR = "/"
    VALID_LAYERS = {MAMBA, ATTENTION, MLP, MOE}


@dataclass
class ParsedHybridPattern:
    """Result of parsing a unified hybrid pattern string.

    A unified pattern encodes both the main decoder pattern and the MTP pattern
    in a single string using "/" as a separator. The main pattern may also
    contain "|" pipe symbols to define pipeline stage boundaries for flexible
    virtual pipeline parallelism (fVPP).

    Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."

    Examples:
        - "M*M*" -> main="M*M*", mtp=None, depths=0 (no MTP)
        - "M*M*/MM/MM" -> main="M*M*", mtp="MM", depths=2
        - "MMMM/*M/*M/*M" -> main="MMMM", mtp="*M", depths=3
        - "M-M-|M-M*-/MM/MM" -> main="M-M-|M-M*-" (2 PP stages), mtp="MM", depths=2

    The "/" symbol introduces MTP patterns. Each repeated pattern after the main
    decoder represents one MTP prediction depth.

    The "|" symbol in the main pattern defines pipeline stage boundaries.

    Attributes:
        main_pattern: The main decoder layer pattern (e.g., "M*M*" or "M-M-|M-M*-")
        mtp_pattern: The MTP layer pattern per depth (e.g., "MM"), or None if no MTP
        mtp_num_depths: Number of MTP prediction depths (0 if no MTP)
    """

    main_pattern: Optional[str]
    mtp_pattern: Optional[str]
    mtp_num_depths: int


def get_hybrid_total_layer_count(pattern: str) -> int:
    """Returns the total number of main decoder layers in a hybrid layer pattern.

    Extracts the main pattern (before the first MTP separator '/'), strips
    pipeline stage separators '|', and returns the character count.

    Args:
        pattern: Full hybrid layer pattern, possibly including MTP and pipe separators.

    Returns:
        Total number of layers in the main decoder pattern.
    """
    main_pattern = pattern.split(Symbols.MTP_SEPARATOR)[0]
    return len(main_pattern.replace(Symbols.PIPE, ''))


def get_hybrid_total_pipeline_segment_count(pattern: str) -> int:
    """Returns the number of pipeline segments in a hybrid layer pattern.

    Extracts the main pattern (before the first MTP separator '/') and counts
    the number of segments delimited by '|'.

    Args:
        pattern: Full hybrid layer pattern, possibly including MTP and pipe separators.

    Returns:
        Number of pipeline segments (pipe count + 1).
    """
    main_pattern = pattern.split(Symbols.MTP_SEPARATOR)[0]
    return main_pattern.count(Symbols.PIPE) + 1


def get_hybrid_layer_counts(pattern: str) -> Tuple[int, int, int, int]:
    """Count layers by type across the full hybrid pattern (main + MTP).

    Parses the pattern to extract main and MTP components, then counts
    each layer type. Main pattern '|' separators are skipped. MTP layers
    are counted once per MTP depth.

    Args:
        pattern: Full hybrid layer pattern string.

    Returns:
        Tuple of (num_attention, num_mamba, num_mlp, num_moe) layer counts.

    Examples:
        >>> get_hybrid_layer_counts("M*M*")
        (2, 2, 0, 0)

        >>> get_hybrid_layer_counts("M-M-|M-M*-/MM/MM")
        (1, 8, 4, 0)
    """
    parsed = parse_hybrid_pattern(pattern)
    counts = {Symbols.ATTENTION: 0, Symbols.MAMBA: 0, Symbols.MLP: 0, Symbols.MOE: 0}

    # Count main decoder layers (skip '|' pipe separators)
    if parsed.main_pattern:
        for char in parsed.main_pattern:
            if char in counts:
                counts[char] += 1

    # Count MTP layers (pattern repeated mtp_num_depths times)
    if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
        for char in parsed.mtp_pattern:
            if char in counts:
                counts[char] += parsed.mtp_num_depths

    return (
        counts[Symbols.ATTENTION],
        counts[Symbols.MAMBA],
        counts[Symbols.MLP],
        counts[Symbols.MOE],
    )


def parse_hybrid_pattern(pattern: Optional[str]) -> ParsedHybridPattern:
    """Parse a unified hybrid pattern string into main and MTP components.

    The pattern uses "/" as a separator between the main decoder pattern and
    MTP patterns. Each MTP pattern after the separator represents one prediction
    depth. The main pattern may contain "|" pipe symbols for pipeline stage
    boundaries.

    Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."

    Args:
        pattern: Unified pattern string, e.g., "M*M*/MM/MM" or just "M*M*"

    Returns:
        ParsedHybridPattern with main_pattern, mtp_pattern, and mtp_num_depths

    Raises:
        ValueError: If MTP patterns are inconsistent (all must be identical)
        ValueError: If pattern contains invalid layer symbols

    Examples:
        >>> parse_hybrid_pattern("M*M*")
        ParsedHybridPattern(main_pattern="M*M*", mtp_pattern=None, mtp_num_depths=0)

        >>> parse_hybrid_pattern("M*M*/MM/MM")
        ParsedHybridPattern(main_pattern="M*M*", mtp_pattern="MM", mtp_num_depths=2)

        >>> parse_hybrid_pattern("MMMM/*M/*M/*M")
        ParsedHybridPattern(main_pattern="MMMM", mtp_pattern="*M", mtp_num_depths=3)

        >>> parse_hybrid_pattern("M-M-|M-M*-/MM/MM")
        ParsedHybridPattern(main_pattern="M-M-|M-M*-", mtp_pattern="MM", mtp_num_depths=2)
    """
    if pattern is None:
        return ParsedHybridPattern(main_pattern=None, mtp_pattern=None, mtp_num_depths=0)

    parts = pattern.split(Symbols.MTP_SEPARATOR)

    if len(parts) == 1:
        # No MTP separator found - pattern is main decoder only
        main_pattern = parts[0]
        _validate_pattern(main_pattern, "main", allow_pipe=True)
        return ParsedHybridPattern(main_pattern=main_pattern, mtp_pattern=None, mtp_num_depths=0)

    # First part is main decoder pattern
    main_pattern = parts[0]
    if main_pattern:
        _validate_pattern(main_pattern, "main", allow_pipe=True)

    # Remaining parts are MTP patterns (one per depth)
    mtp_parts = parts[1:]

    if not mtp_parts or all(p == "" for p in mtp_parts):
        # No MTP patterns after separator
        return ParsedHybridPattern(
            main_pattern=main_pattern if main_pattern else None, mtp_pattern=None, mtp_num_depths=0
        )

    # Validate all MTP patterns are identical
    mtp_pattern = mtp_parts[0]
    for i, part in enumerate(mtp_parts[1:], start=2):
        if part != mtp_pattern:
            raise ValueError(
                f"All MTP patterns must be identical. "
                f"Pattern 1 is '{mtp_pattern}', but pattern {i} is '{part}'. "
                f"Full pattern: '{pattern}'"
            )

    _validate_pattern(mtp_pattern, "MTP", allow_pipe=False)

    return ParsedHybridPattern(
        main_pattern=main_pattern if main_pattern else None,
        mtp_pattern=mtp_pattern,
        mtp_num_depths=len(mtp_parts),
    )


def _validate_pattern(pattern: str, pattern_name: str, allow_pipe: bool = False) -> None:
    """Validate that a pattern contains only valid layer symbols.

    Args:
        pattern: Layer pattern string to validate
        pattern_name: Name of pattern for error messages (e.g., "main" or "MTP")
        allow_pipe: Whether to allow the pipe '|' separator (for main patterns)

    Raises:
        ValueError: If pattern contains invalid symbols
    """
    valid_chars = Symbols.VALID_LAYERS | {Symbols.PIPE} if allow_pipe else Symbols.VALID_LAYERS
    for char in pattern:
        if char not in valid_chars:
            raise ValueError(
                f"In {pattern_name} pattern, '{char}' is not a valid layer symbol. "
                f"Valid symbols are: {valid_chars}"
            )


def validate_segment_layers(segment: str) -> list:
    """Validate and convert a single pipeline segment pattern to a layer type list.

    This is used after the main pattern has been split by '|' into segments.
    Each segment should contain only valid layer symbols (no '|').

    Args:
        segment: A single pipeline segment pattern string (e.g., "M-M*-")

    Returns:
        List of layer type characters.

    Raises:
        ValueError: If segment contains invalid layer symbols.
    """
    layer_type_list = list(segment)
    for layer_char in layer_type_list:
        if layer_char not in Symbols.VALID_LAYERS:
            raise ValueError(
                f"In hybrid layer pattern segment, '{layer_char}' is not "
                f"one of {Symbols.VALID_LAYERS}"
            )
    return layer_type_list


def select_pipeline_segment(
    main_pattern: str, pp_group: Optional[torch.distributed.ProcessGroup], vp_stage: Optional[int]
) -> Tuple[List[str], int]:
    """Select and validate the pipeline segment for the given PP rank and VP stage.

    Splits the main pattern by '|' into pipeline segments, determines which
    segment belongs to this rank based on PP rank and VP stage, validates the
    segment's layer symbols, and logs the assignment.

    Args:
        main_pattern: Main decoder pattern (may contain '|' separators).
            Empty string is allowed (produces one empty segment).
        pp_group: Pipeline parallel process group, or None if not using PP.
        vp_stage: Virtual pipeline stage, or None if not using VPP.

    Returns:
        Tuple of (layer_type_list, layer_offset) where layer_type_list is
        the list of layer type characters for this segment, and layer_offset
        is the sum of layer counts from all preceding segments.

    Raises:
        ValueError: If the segment contains invalid layer symbols.
        IndexError: If the computed segment index is out of range.
    """
    segments = main_pattern.split(Symbols.PIPE) if main_pattern else ['']

    pp_rank = torch.distributed.get_rank(pp_group) if pp_group is not None else 0
    pp_size = torch.distributed.get_world_size(pp_group) if pp_group is not None else 1
    vp_rel = vp_stage if vp_stage is not None else 0
    segment_index = vp_rel * pp_size + pp_rank

    layer_offset = sum(len(segments[i]) for i in range(segment_index))
    my_segment = segments[segment_index]

    layer_type_list = validate_segment_layers(my_segment)

    log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        f"MambaModel: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_rel}, "
        f"segment_index={segment_index}/{len(segments)}, "
        f"layers='{my_segment}' ({len(layer_type_list)} layers), "
        f"layer_offset={layer_offset}",
    )

    return layer_type_list, layer_offset


def get_layer_maps_from_layer_type_list(
    layer_type_list: List[str],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Returns maps from global layer index to the corresponding layer index
    for each layer type in [Attention, Mamba, MLP, MoE] given a layer type list.
    """
    layer_types = [Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE]
    layer_maps = {layer_type: {} for layer_type in layer_types}
    for global_layer_idx, layer_type in enumerate(layer_type_list):
        layer_map = layer_maps[layer_type]
        local_layer_idx = len(layer_map)
        layer_map[global_layer_idx] = local_layer_idx
    return [layer_maps[layer_type] for layer_type in layer_types]
