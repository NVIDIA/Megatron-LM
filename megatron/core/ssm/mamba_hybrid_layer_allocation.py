# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from megatron.core.utils import log_on_each_pipeline_stage, log_single_rank

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


def pattern_from_ratios(
    num_layers: int, attention_ratio: float = 0.0, mlp_ratio: float = 0.0
) -> str:
    """Convert deprecated ratio arguments to a layer pattern string.

    Generates an evenly-spaced hybrid layer pattern from target attention and MLP
    ratios. This exists for backward compatibility with code that uses the deprecated
    hybrid_attention_ratio and hybrid_mlp_ratio parameters.

    Args:
        num_layers: Total number of layers.
        attention_ratio: Target ratio of attention layers to total layers.
        mlp_ratio: Target ratio of MLP layers to total layers.

    Returns:
        A layer pattern string (e.g., "MMM*MMM*MM").
    """
    assert num_layers > 0
    assert 0.0 <= attention_ratio <= 1.0
    assert 0.0 <= mlp_ratio <= 1.0
    assert attention_ratio + mlp_ratio <= 1.0

    # Allocate attention layers (evenly spaced, starting and ending with mamba)
    attention_count = round(num_layers * attention_ratio)
    mamba_count = num_layers - attention_count
    sections = attention_count + 1
    section_len = mamba_count / sections

    layer_types = [Symbols.MAMBA] * num_layers
    x = section_len
    for i in range(num_layers):
        if x < 0.5:
            layer_types[i] = Symbols.ATTENTION
            x += section_len
        else:
            x -= 1

    # Allocate MLP layers (evenly distributed, not replacing attention)
    mlp_count = round(num_layers * mlp_ratio)
    if mlp_count > 0:
        mamba_count -= mlp_count
        ratio = mamba_count / mlp_count
        x = ratio
        for i in range(num_layers):
            if layer_types[i] == Symbols.MAMBA:
                if x < 0.5:
                    layer_types[i] = Symbols.MLP
                    x += ratio
                else:
                    x -= 1

    return ''.join(layer_types)


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
    _validate_pattern(main_pattern, "main", allow_pipe=True)
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


def get_hybrid_layer_counts(pattern: str) -> Dict[str, int]:
    """Count layers by type across the full hybrid pattern (main + MTP).

    Parses the pattern to extract main and MTP components, then counts
    each layer type. Main pattern '|' separators are skipped. MTP layers
    are counted once per MTP depth.

    Args:
        pattern: Full hybrid layer pattern string.

    Returns:
        Dictionary mapping layer symbol to count. Keys are Symbols.ATTENTION,
        Symbols.MAMBA, Symbols.MLP, and Symbols.MOE.

    Examples:
        >>> get_hybrid_layer_counts("M*M*")
        {'*': 2, 'M': 2, '-': 0, 'E': 0}

        >>> get_hybrid_layer_counts("M-M-|M-M*-/MM/MM")
        {'*': 1, 'M': 8, '-': 4, 'E': 0}
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

    return counts


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


def validate_segment_layers(segment: str) -> List[str]:
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
    main_pattern: str,
    pp_group: Optional[torch.distributed.ProcessGroup],
    vp_stage: Optional[int],
    first_stage_layers: Optional[int] = None,
    last_stage_layers: Optional[int] = None,
) -> Tuple[List[str], int]:
    """Select and validate the pipeline segment for the given PP rank and VP stage.

    When the main pattern contains '|' pipe separators, splits by '|' into
    pipeline segments and selects the segment for the current PP rank / VP stage.

    When the pattern has no pipes but pp_size > 1, falls back to runtime layer
    slicing (for backwards compatibility), supporting both even and uneven PP splits
    via first_stage_layers / last_stage_layers.

    Args:
        main_pattern: Main decoder pattern (may contain '|' separators).
            Empty string is allowed (produces one empty segment).
        pp_group: Pipeline parallel process group, or None if not using PP.
        vp_stage: Virtual pipeline stage, or None if not using VPP.
        first_stage_layers: Number of layers on the first pipeline stage for
            uneven PP. Only valid when the pattern has no pipe separators.
        last_stage_layers: Number of layers on the last pipeline stage for
            uneven PP. Only valid when the pattern has no pipe separators.

    Returns:
        Tuple of (layer_type_list, layer_offset) where layer_type_list is
        the list of layer type characters for this segment, and layer_offset
        is the sum of layer counts from all preceding segments.

    Raises:
        ValueError: If the segment contains invalid layer symbols, if
            first/last_stage_layers are used with pipe separators, if VPP is
            requested without pipe separators, or if layer counts are not
            evenly divisible across pipeline stages.
    """
    segments = main_pattern.split(Symbols.PIPE) if main_pattern else ['']

    pp_rank = torch.distributed.get_rank(pp_group) if pp_group is not None else 0
    pp_size = torch.distributed.get_world_size(pp_group) if pp_group is not None else 1

    if len(segments) > 1 and (first_stage_layers is not None or last_stage_layers is not None):
        raise ValueError(
            "Cannot specify num_layers_in_first_pipeline_stage or "
            "num_layers_in_last_pipeline_stage when hybrid_layer_pattern "
            "contains pipe ('|') separators. The pipeline layout is already "
            "explicitly defined by the pipe separators."
        )

    if len(segments) == 1 and pp_size > 1:
        if vp_stage is not None:
            raise ValueError(
                "Virtual pipeline parallelism (vp_stage != None) is not supported "
                "when hybrid_layer_pattern has no pipe ('|') separators. "
                "Add '|' separators to define explicit pipeline/virtual-pipeline "
                "stage boundaries."
            )
        log_single_rank(
            logger,
            logging.WARNING,
            "DEPRECATION: Using hybrid_layer_pattern without pipe ('|') separators "
            "with pipeline_model_parallel_size > 1 is deprecated. Please add '|' "
            "separators to explicitly define pipeline stage boundaries. "
            "Example: 'M*M*M*M*' with pp_size=2 should become 'M*M*|M*M*'.",
        )
        full_pattern = segments[0]
        layer_type_list = validate_segment_layers(full_pattern)
        num_layers = len(layer_type_list)

        if first_stage_layers is not None or last_stage_layers is not None:
            first = first_stage_layers or 0
            last = last_stage_layers or 0
            middle_num_layers = num_layers - first - last
            middle_stages = pp_size - sum(
                1 for x in (first_stage_layers, last_stage_layers) if x is not None
            )
            if middle_stages > 0:
                if middle_num_layers % middle_stages != 0:
                    raise ValueError(
                        f"Middle layers ({middle_num_layers}) must be evenly divisible "
                        f"by middle pipeline stages ({middle_stages})."
                    )
                layers_per_middle = middle_num_layers // middle_stages
            else:
                layers_per_middle = 0

            is_first = first_stage_layers is not None and pp_rank == 0
            is_last = last_stage_layers is not None and pp_rank == pp_size - 1

            if is_first:
                offset = 0
                count = first
            elif is_last:
                offset = num_layers - last
                count = last
            else:
                middle_rank = pp_rank if first_stage_layers is None else pp_rank - 1
                offset = middle_rank * layers_per_middle + first
                count = layers_per_middle
        else:
            if num_layers % pp_size != 0:
                raise ValueError(
                    f"Number of layers ({num_layers}) must be evenly divisible "
                    f"by pipeline-model-parallel-size ({pp_size}) when no pipe "
                    f"separators are specified in the pattern."
                )
            layers_per_rank = num_layers // pp_size
            offset = pp_rank * layers_per_rank
            count = layers_per_rank

        selected = layer_type_list[offset : offset + count]
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f"MambaModel: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_stage}, "
            f"layers='{''.join(selected)}' ({len(selected)} layers), "
            f"layer_offset={offset} (auto-split)",
        )
        return selected, offset

    # Pipe-based segment selection
    if len(segments) > 1 and len(segments) % pp_size != 0:
        raise ValueError(
            f"The number of pipe-delimited segments ({len(segments)}) in "
            f"hybrid_layer_pattern must be evenly divisible by "
            f"pipeline_model_parallel_size ({pp_size})."
        )

    vp_rel = vp_stage if vp_stage is not None else 0
    segment_index = vp_rel * pp_size + pp_rank

    if segment_index >= len(segments):
        raise ValueError(
            f"Pipeline segment index {segment_index} (pp_rank={pp_rank}, "
            f"vp_stage={vp_rel}) is out of range for {len(segments)} segments. "
            f"The pattern does not define enough pipe-delimited segments for "
            f"the current PP/VPP configuration."
        )

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
