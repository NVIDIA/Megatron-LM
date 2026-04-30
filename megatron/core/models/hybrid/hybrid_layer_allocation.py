# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from megatron.core.utils import log_on_each_pipeline_stage, log_single_rank

logger = logging.getLogger(__name__)


class Symbols:
    """Symbols for different layer types and pattern separators."""

    MAMBA = "M"
    GDN = 'G'
    ATTENTION = "*"
    DS_ATTENTION = "D"
    MLP = "-"
    MOE = 'E'
    PIPE = '|'
    MTP_SEPARATOR = "/"
    GROUP_START = "["
    GROUP_END = "]"
    VALID_LAYERS = {MAMBA, GDN, ATTENTION, DS_ATTENTION, MLP, MOE}

    @classmethod
    def name_sorted_valid_layer_symbols(cls) -> list[str]:
        """Return the valid layer symbols sorted lexicographically by their public attribute
        name.
        """
        valid_layer_attrs = []
        for name, value in vars(cls).items():
            if not name.startswith('_') and value in cls.VALID_LAYERS:
                valid_layer_attrs.append((name, value))
        valid_layer_attrs.sort()
        return [value for (_, value) in valid_layer_attrs]


LayerPatternItem = Union[str, Tuple[str, ...]]


def is_layer_group(layer_type: LayerPatternItem) -> bool:
    """Return whether a parsed layer item is a bracketed group."""
    return isinstance(layer_type, tuple)


def flatten_layer_type_list(layer_type_list: List[LayerPatternItem]) -> List[str]:
    """Flatten bracketed layer groups into their physical layer symbols."""
    flattened = []
    for layer_type in layer_type_list:
        if is_layer_group(layer_type):
            flattened.extend(layer_type)
        else:
            flattened.append(layer_type)
    return flattened


def get_layer_type_physical_count(layer_type: LayerPatternItem) -> int:
    """Return the number of physical layers represented by a parsed layer item."""
    return len(layer_type) if is_layer_group(layer_type) else 1


def get_layer_type_logical_count(layer_type: LayerPatternItem) -> int:
    """Return the number of logical layers represented by a parsed layer item."""
    return 1


def get_layer_type_list_physical_count(layer_type_list: List[LayerPatternItem]) -> int:
    """Return the number of physical layers represented by a parsed layer list."""
    return sum(get_layer_type_physical_count(layer_type) for layer_type in layer_type_list)


def get_layer_type_list_logical_count(layer_type_list: List[LayerPatternItem]) -> int:
    """Return the number of logical layers represented by a parsed layer list."""
    return sum(get_layer_type_logical_count(layer_type) for layer_type in layer_type_list)


def layer_type_item_to_str(layer_type: LayerPatternItem) -> str:
    """Render one parsed layer item back to pattern syntax."""
    if is_layer_group(layer_type):
        return f"{Symbols.GROUP_START}{''.join(layer_type)}{Symbols.GROUP_END}"
    return layer_type


def layer_type_list_to_str(layer_type_list: List[LayerPatternItem]) -> str:
    """Render a parsed layer list back to pattern syntax."""
    return ''.join(layer_type_item_to_str(layer_type) for layer_type in layer_type_list)


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
    return sum(
        get_layer_type_list_physical_count(validate_segment_layers(segment))
        for segment in main_pattern.split(Symbols.PIPE)
    )


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
        Dictionary mapping layer symbol to count. Keys are all valid layer symbols
            (Symbols.VALID_LAYERS).

    Examples:
        >>> get_hybrid_layer_counts("M*M*")
        {'*': 2, 'G': 0, 'D': 0, 'M': 2, '-': 0, 'E': 0}

        >>> get_hybrid_layer_counts("M-M-|M-M*-/MM/MM")
        {'*': 1, 'G': 0, 'D': 0, 'M': 8, '-': 4, 'E': 0}
    """
    parsed = parse_hybrid_pattern(pattern)
    counts = {symbol: 0 for symbol in Symbols.name_sorted_valid_layer_symbols()}

    # Count main decoder layers (skip '|' pipe separators)
    if parsed.main_pattern:
        for segment in parsed.main_pattern.split(Symbols.PIPE):
            for char in flatten_layer_type_list(validate_segment_layers(segment)):
                counts[char] += 1

    # Count MTP layers (pattern repeated mtp_num_depths times)
    if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
        for char in flatten_layer_type_list(validate_segment_layers(parsed.mtp_pattern)):
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
    valid_chars = (
        Symbols.VALID_LAYERS
        | {Symbols.GROUP_START, Symbols.GROUP_END}
        | ({Symbols.PIPE} if allow_pipe else set())
    )
    if not allow_pipe and Symbols.PIPE in pattern:
        raise ValueError(
            f"In {pattern_name} pattern, '{Symbols.PIPE}' is not a valid layer symbol. "
            f"Valid symbols are: {valid_chars}"
        )
    flat_layers = []
    for segment in pattern.split(Symbols.PIPE):
        flat_layers.extend(
            flatten_layer_type_list(
                _parse_segment_layers(segment, pattern_name, valid_chars=valid_chars)
            )
        )

    # Disallow Attention + MLA/DSA hybridity.
    if Symbols.ATTENTION in flat_layers and Symbols.DS_ATTENTION in flat_layers:
        raise ValueError("Not supported to have both Attention and MLA/DSA in one model")


def _parse_segment_layers(
    segment: str, pattern_name: str, valid_chars: Optional[set[str]] = None
) -> List[LayerPatternItem]:
    """Parse a pipe-free pattern segment into symbols and bracketed groups."""
    if valid_chars is None:
        valid_chars = Symbols.VALID_LAYERS | {Symbols.GROUP_START, Symbols.GROUP_END}

    layer_type_list: List[LayerPatternItem] = []
    flat_layers = []
    i = 0
    while i < len(segment):
        layer_char = segment[i]
        if layer_char == Symbols.GROUP_START:
            group_end = segment.find(Symbols.GROUP_END, i + 1)
            if group_end == -1:
                raise ValueError(
                    f"In {pattern_name} pattern, '[' starts a layer group without a matching ']'."
                )
            group = segment[i + 1 : group_end]
            if group == "":
                raise ValueError(f"In {pattern_name} pattern, layer groups cannot be empty.")
            if Symbols.GROUP_START in group or Symbols.GROUP_END in group:
                raise ValueError(
                    f"In {pattern_name} pattern, nested layer groups are not supported."
                )
            for group_char in group:
                if group_char not in Symbols.VALID_LAYERS:
                    raise ValueError(
                        f"In {pattern_name} pattern, '{group_char}' is not a valid layer symbol. "
                        f"Valid symbols are: {valid_chars}"
                    )
            if Symbols.MOE in group[:-1]:
                raise ValueError(
                    f"In {pattern_name} pattern, MoE layer '{Symbols.MOE}' must be the last "
                    f"symbol inside a layer group."
                )
            group_tuple = tuple(group)
            layer_type_list.append(group_tuple)
            flat_layers.extend(group_tuple)
            i = group_end + 1
            continue
        if layer_char == Symbols.GROUP_END:
            raise ValueError(
                f"In {pattern_name} pattern, ']' closes a layer group that was not opened."
            )
        if layer_char not in Symbols.VALID_LAYERS:
            raise ValueError(
                f"In {pattern_name} pattern, '{layer_char}' is not a valid layer symbol. "
                f"Valid symbols are: {valid_chars}"
            )
        layer_type_list.append(layer_char)
        flat_layers.append(layer_char)
        i += 1

    if Symbols.ATTENTION in flat_layers and Symbols.DS_ATTENTION in flat_layers:
        raise ValueError("Not supported to have both Attention and MLA/DSA in one model")

    return layer_type_list


def validate_segment_layers(segment: str) -> List[LayerPatternItem]:
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
    return _parse_segment_layers(segment, "hybrid layer pattern segment")


def _slice_layer_type_list_by_physical_range(
    layer_type_list: List[LayerPatternItem], offset: int, count: int
) -> List[LayerPatternItem]:
    """Slice parsed layer items by physical layer range without splitting groups."""
    selected = []
    cursor = 0
    end = offset + count
    for layer_type in layer_type_list:
        item_count = get_layer_type_physical_count(layer_type)
        item_end = cursor + item_count
        if item_end <= offset:
            cursor = item_end
            continue
        if cursor >= end:
            break
        if cursor < offset or item_end > end:
            raise ValueError(
                "Pipeline splitting would split a bracketed hybrid layer group. "
                "Add pipe ('|') separators around bracketed groups to define valid boundaries."
            )
        selected.append(layer_type)
        cursor = item_end
    return selected


def _get_logical_offset_from_physical_offset(
    layer_type_list: List[LayerPatternItem], offset: int
) -> int:
    """Return the logical item count before a physical-layer offset."""
    logical_offset = 0
    cursor = 0
    for layer_type in layer_type_list:
        item_count = get_layer_type_physical_count(layer_type)
        item_end = cursor + item_count
        if item_end <= offset:
            logical_offset += get_layer_type_logical_count(layer_type)
            cursor = item_end
            continue
        if cursor == offset:
            return logical_offset
        raise ValueError(
            "Pipeline splitting would split a bracketed hybrid layer group. "
            "Add pipe ('|') separators around bracketed groups to define valid boundaries."
        )
    if cursor == offset:
        return logical_offset
    raise ValueError(f"Physical layer offset {offset} is out of range for hybrid layer pattern.")


def select_pipeline_segment_with_logical_offset(
    main_pattern: str,
    pp_group: Optional[torch.distributed.ProcessGroup],
    vp_stage: Optional[int],
    first_stage_layers: Optional[int] = None,
    last_stage_layers: Optional[int] = None,
) -> Tuple[List[LayerPatternItem], int, int]:
    """Select a pipeline segment and return physical and logical offsets."""
    layer_type_list, layer_offset = select_pipeline_segment(
        main_pattern,
        pp_group,
        vp_stage,
        first_stage_layers=first_stage_layers,
        last_stage_layers=last_stage_layers,
    )

    segments = main_pattern.split(Symbols.PIPE) if main_pattern else ['']
    if len(segments) == 1:
        full_layer_type_list = validate_segment_layers(segments[0])
        logical_layer_offset = _get_logical_offset_from_physical_offset(
            full_layer_type_list, layer_offset
        )
    else:
        pp_rank = torch.distributed.get_rank(pp_group) if pp_group is not None else 0
        pp_size = torch.distributed.get_world_size(pp_group) if pp_group is not None else 1
        vp_rel = vp_stage if vp_stage is not None else 0
        segment_index = vp_rel * pp_size + pp_rank
        logical_layer_offset = sum(
            get_layer_type_list_logical_count(validate_segment_layers(segments[i]))
            for i in range(segment_index)
        )

    return layer_type_list, layer_offset, logical_layer_offset


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
        num_layers = get_layer_type_list_physical_count(layer_type_list)

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

        selected = _slice_layer_type_list_by_physical_range(layer_type_list, offset, count)
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f"HybridModel: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_stage}, "
            f"layers='{layer_type_list_to_str(selected)}' "
            f"({get_layer_type_list_physical_count(selected)} layers), "
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

    layer_offset = sum(
        get_layer_type_list_physical_count(validate_segment_layers(segments[i]))
        for i in range(segment_index)
    )
    my_segment = segments[segment_index]

    layer_type_list = validate_segment_layers(my_segment)

    log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        f"HybridModel: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_rel}, "
        f"segment_index={segment_index}/{len(segments)}, "
        f"layers='{my_segment}' ({get_layer_type_list_physical_count(layer_type_list)} layers), "
        f"layer_offset={layer_offset}",
    )

    return layer_type_list, layer_offset


def get_layer_maps_from_layer_type_list(
    layer_type_list: list[LayerPatternItem],
) -> dict[str, dict[int, int]]:
    """
    Returns maps from global layer index to the corresponding layer index
    for each valid layer type (those in Symbols.VALID_LAYERS) given a layer type list.
    """
    layer_types = [symbol for symbol in Symbols.name_sorted_valid_layer_symbols()]
    layer_maps = {layer_type: {} for layer_type in layer_types}
    for global_layer_idx, layer_type in enumerate(flatten_layer_type_list(layer_type_list)):
        layer_map = layer_maps[layer_type]
        local_layer_idx = len(layer_map)
        layer_map[global_layer_idx] = local_layer_idx
    return layer_maps
