# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TypeAlias

import torch

from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import log_on_each_pipeline_stage, log_single_rank

Symbols = layer_utils.Symbols

logger = logging.getLogger(__name__)


class PipelineSplit:
    """Bare class sentinel that separates pipeline stages in a hybrid config list.

    Add ``PipelineSplit`` itself to the list. Instances of this class are not valid entries.
    """


class MTPSplit:
    """Bare class sentinel that begins an MTP depth in a hybrid config list.

    Add ``MTPSplit`` itself to the list. Instances of this class are not valid entries.
    """


HybridLayerConfigListEntry: TypeAlias = TransformerConfig | type[PipelineSplit] | type[MTPSplit]
MainLayerConfigListEntry: TypeAlias = TransformerConfig | type[PipelineSplit]


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


@dataclass(frozen=True)
class ParsedHybridConfigList:
    """Validated view of a hybrid layer config list.

    The returned tuples contain the original objects. Cloning is intentionally deferred until a
    physical decoder or MTP layer is selected for construction.

    Attributes:
        main_layer_config_list: Decoder configs and any ``PipelineSplit`` sentinels.
        mtp_layer_config_list: Source config objects for one MTP depth, or ``None`` without MTP.
        mtp_num_depths: Number of ``MTPSplit`` sections in the source list.
    """

    main_layer_config_list: tuple[MainLayerConfigListEntry, ...]
    mtp_layer_config_list: tuple[TransformerConfig, ...] | None
    mtp_num_depths: int


def _validate_layer_config_sequence(
    layer_configs: Sequence[TransformerConfig], section_name: str
) -> None:
    """Validate exact config types and attention-family compatibility."""
    config_types = set()
    supported_types = set(Symbols.LAYER_CONFIG_MAP.values())
    for index, layer_config in enumerate(layer_configs):
        if type(layer_config) not in supported_types:
            if isinstance(layer_config, (PipelineSplit, MTPSplit)):
                raise ValueError(
                    f"{section_name} entry {index} is a sentinel instance. "
                    "Use the bare PipelineSplit or MTPSplit class instead."
                )
            raise ValueError(
                f"{section_name} entry {index} has unsupported hybrid layer config type "
                f"{type(layer_config).__name__}. Exact built-in layer config types are required."
            )
        config_types.add(type(layer_config))

    attention_type = Symbols.LAYER_CONFIG_MAP[Symbols.ATTENTION]
    dsa_type = Symbols.LAYER_CONFIG_MAP[Symbols.DS_ATTENTION]
    mla_type = Symbols.LAYER_CONFIG_MAP[Symbols.MLA]
    if attention_type in config_types and (dsa_type in config_types or mla_type in config_types):
        raise ValueError(f"Not supported to have both Attention and MLA/DSA in the {section_name}")


def parse_hybrid_layer_config_list(
    hybrid_layer_config_list: Sequence[HybridLayerConfigListEntry],
    expected_num_layers: int | None = None,
    expected_mtp_num_layers: int | None = None,
) -> ParsedHybridConfigList:
    """Parse and validate a first-class hybrid layer config list.

    ``PipelineSplit`` separates main-decoder pipeline segments. ``MTPSplit`` begins one MTP
    prediction depth, and every MTP depth must contain the same source objects in the same order.
    The input is snapshotted as tuples but its configs are not copied or converted to symbols.

    Args:
        hybrid_layer_config_list: Layer configs plus optional bare split sentinel classes.
        expected_num_layers: If provided, required number of main-decoder configs.
        expected_mtp_num_layers: If provided (including zero), required number of MTP depths.

    Returns:
        A validated, identity-preserving parsed representation.

    Raises:
        ValueError: If an entry, marker placement, layer count, or MTP template is invalid.
    """
    source_entries = tuple(hybrid_layer_config_list)
    main_entries: list[MainLayerConfigListEntry] = []
    mtp_sections: list[list[TransformerConfig]] = []
    current_mtp_section: list[TransformerConfig] | None = None

    for index, entry in enumerate(source_entries):
        if entry is PipelineSplit:
            if current_mtp_section is not None:
                raise ValueError(
                    f"PipelineSplit at index {index} is not allowed after the first MTPSplit"
                )
            main_entries.append(PipelineSplit)
            continue

        if entry is MTPSplit:
            if current_mtp_section is not None:
                if not current_mtp_section:
                    raise ValueError("Each MTPSplit must begin a non-empty MTP layer config list")
                mtp_sections.append(current_mtp_section)
            current_mtp_section = []
            continue

        if current_mtp_section is None:
            main_entries.append(entry)
        else:
            current_mtp_section.append(entry)

    if current_mtp_section is not None:
        if not current_mtp_section:
            raise ValueError("Each MTPSplit must begin a non-empty MTP layer config list")
        mtp_sections.append(current_mtp_section)

    main_configs = [entry for entry in main_entries if entry is not PipelineSplit]
    _validate_layer_config_sequence(main_configs, "main decoder")
    for depth, mtp_section in enumerate(mtp_sections, start=1):
        _validate_layer_config_sequence(mtp_section, f"MTP depth {depth}")

    if expected_num_layers is not None and len(main_configs) != expected_num_layers:
        raise ValueError(
            "hybrid_layer_config_list defines "
            f"{len(main_configs)} main decoder layers, but num_layers is {expected_num_layers}"
        )

    mtp_num_depths = len(mtp_sections)
    if expected_mtp_num_layers is not None and mtp_num_depths != expected_mtp_num_layers:
        raise ValueError(
            "hybrid_layer_config_list defines "
            f"{mtp_num_depths} MTP depths, but mtp_num_layers is {expected_mtp_num_layers}"
        )

    mtp_layer_config_list: tuple[TransformerConfig, ...] | None = None
    if mtp_sections:
        first_mtp_section = mtp_sections[0]
        for depth, mtp_section in enumerate(mtp_sections[1:], start=2):
            if len(mtp_section) != len(first_mtp_section) or any(
                actual is not expected
                for actual, expected in zip(mtp_section, first_mtp_section, strict=True)
            ):
                raise ValueError(
                    "All MTP layer config lists must contain the same source objects in the same "
                    f"order; MTP depth {depth} differs from depth 1"
                )
        mtp_layer_config_list = tuple(first_mtp_section)

    return ParsedHybridConfigList(
        main_layer_config_list=tuple(main_entries),
        mtp_layer_config_list=mtp_layer_config_list,
        mtp_num_depths=mtp_num_depths,
    )


def clone_hybrid_layer_config_list(
    layer_config_list: Sequence[TransformerConfig],
) -> list[TransformerConfig]:
    """Clone every physical layer-config occurrence without invoking ``__post_init__``."""
    _validate_layer_config_sequence(layer_config_list, "hybrid layer config list")
    return [
        layer_utils.normalize_hybrid_layer_config(type(layer_config).from_config(layer_config))
        for layer_config in layer_config_list
    ]


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
    _validate_pattern(main_pattern, allow_pipe=True)
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
        Dictionary mapping layer symbol to count. Keys are all valid layer symbols
            (the keys of ``Symbols.LAYER_CONFIG_MAP``).

    Examples:
        >>> get_hybrid_layer_counts("M*M*")
        {'*': 2, 'D': 0, 'G': 0, 'M': 2, '+': 0, '-': 0, 'E': 0}

        >>> get_hybrid_layer_counts("M-M-|M-M*-/MM/MM")
        {'*': 1, 'D': 0, 'G': 0, 'M': 8, '+': 0, '-': 4, 'E': 0}
    """
    parsed = parse_hybrid_pattern(pattern)
    counts = {symbol: 0 for symbol in Symbols.name_sorted_valid_layer_symbols()}

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
        _validate_pattern(main_pattern, allow_pipe=True)
        return ParsedHybridPattern(main_pattern=main_pattern, mtp_pattern=None, mtp_num_depths=0)

    # First part is main decoder pattern
    main_pattern = parts[0]
    if main_pattern:
        _validate_pattern(main_pattern, allow_pipe=True)

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

    _validate_pattern(mtp_pattern)

    return ParsedHybridPattern(
        main_pattern=main_pattern if main_pattern else None,
        mtp_pattern=mtp_pattern,
        mtp_num_depths=len(mtp_parts),
    )


def _validate_pattern(pattern: str, allow_pipe: bool = False) -> None:
    """Validate that a pattern contains only valid layer symbols.

    Args:
        pattern: Layer pattern string to validate
        allow_pipe: Whether to allow the pipe '|' separator (for main patterns)

    Raises:
        ValueError: If pattern contains invalid symbols
    """
    for char in pattern:
        if not layer_utils.is_valid_symbol(char, allow_pipe=allow_pipe):
            raise ValueError(
                f"'{char}' is not a valid layer symbol. "
                f"Valid symbols are: {Symbols.LAYER_CONFIG_MAP.keys()}"
            )

    # Disallow Attention + MLA/DSA hybridity.
    if Symbols.ATTENTION in pattern and (Symbols.DS_ATTENTION in pattern or Symbols.MLA in pattern):
        raise ValueError("Not supported to have both Attention and MLA/DSA in one model")


def validate_segment_layers(segment: str, config: TransformerConfig) -> List[TransformerConfig]:
    """Validate and convert a single pipeline segment pattern to layer configs.

    This is used after the main pattern has been split by '|' into segments.
    Each segment should contain only valid layer symbols (no '|').

    Each layer config is copied from the source config without running ``__post_init__``
    a second time.

    Args:
        segment: A single pipeline segment pattern string (e.g., "M-M*-")
        config: Normalized stack-level config to copy for each layer.

    Returns:
        List of independent per-layer configs.

    Raises:
        ValueError: If segment contains invalid layer symbols.
    """
    _validate_pattern(segment)

    layer_configs: list[TransformerConfig] = []
    for layer_symbol in segment:
        layer_configs.append(layer_utils.create_layer_config(config, layer_symbol))

    return layer_configs


def select_pipeline_segment(
    main_pattern: str,
    config: TransformerConfig,
    pp_group: Optional[torch.distributed.ProcessGroup],
    vp_stage: Optional[int],
    first_stage_layers: Optional[int] = None,
    last_stage_layers: Optional[int] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[List[TransformerConfig], int]:
    """Select and validate the pipeline segment for the given PP rank and VP stage.

    When the main pattern contains '|' pipe separators, splits by '|' into
    pipeline segments and selects the segment for the current PP rank / VP stage.

    When the pattern has no pipes but pp_size > 1, falls back to runtime layer
    slicing (for backwards compatibility), supporting both even and uneven PP splits
    via first_stage_layers / last_stage_layers.

    Args:
        main_pattern: Main decoder pattern (may contain '|' separators).
            Empty string is allowed (produces one empty segment).
        config: Normalized stack-level config to copy for each selected layer.
        pp_group: Pipeline parallel process group, or None if not using PP.
        vp_stage: Virtual pipeline stage, or None if not using VPP.
        first_stage_layers: Number of layers on the first pipeline stage for
            uneven PP. Only valid when the pattern has no pipe separators.
        last_stage_layers: Number of layers on the last pipeline stage for
            uneven PP. Only valid when the pattern has no pipe separators.
        tp_group: Optional tensor-parallel process group used for per-stage logging.
        dp_cp_group: Optional data/context-parallel process group used for per-stage logging.

    Returns:
        Tuple of (layer_config_list, layer_offset) where layer_config_list is
        the list of independent configs for this segment, and layer_offset
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
        _validate_pattern(full_pattern)
        num_layers = len(full_pattern)

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

        selected_pattern = full_pattern[offset : offset + count]
        layer_utils.validate_tp_comm_overlap(config, selected_pattern)
        selected = validate_segment_layers(selected_pattern, config)
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f"HybridModel: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_stage}, "
            f"layers='{selected_pattern}' ({len(selected)} layers), "
            f"layer_offset={offset} (auto-split)",
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
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

    layer_utils.validate_tp_comm_overlap(config, my_segment)
    layer_config_list = validate_segment_layers(my_segment, config)

    log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        f"HybridModel: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_rel}, "
        f"segment_index={segment_index}/{len(segments)}, "
        f"layers='{my_segment}' ({len(layer_config_list)} layers), "
        f"layer_offset={layer_offset}",
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )

    return layer_config_list, layer_offset


def select_pipeline_segment_from_config_list(
    main_layer_config_list: Sequence[MainLayerConfigListEntry],
    pp_group: torch.distributed.ProcessGroup | None,
    vp_stage: int | None,
    virtual_pipeline_model_parallel_size: int | None = None,
    first_stage_layers: int | None = None,
    last_stage_layers: int | None = None,
    tp_group: torch.distributed.ProcessGroup | None = None,
    dp_cp_group: torch.distributed.ProcessGroup | None = None,
) -> Tuple[List[TransformerConfig], int]:
    """Select and clone this rank's decoder configs from a first-class config list.

    The PP/VPP segment ordering is VPP-major, matching ``select_pipeline_segment`` and
    ``PipelineParallelLayerLayout``. Without ``PipelineSplit``, PP uses the existing contiguous
    even/uneven slicing rules and VPP is unsupported. With explicit markers, the number of
    segments must exactly equal the configured PP x VPP topology.

    Args:
        main_layer_config_list: Parsed main-decoder configs and optional ``PipelineSplit`` classes.
        pp_group: Pipeline-parallel process group, or ``None`` for one PP rank.
        vp_stage: Current virtual-pipeline stage, or ``None`` without VPP.
        virtual_pipeline_model_parallel_size: Configured VPP size, or ``None`` without VPP.
        first_stage_layers: Uneven first-stage layer count for marker-free lists.
        last_stage_layers: Uneven last-stage layer count for marker-free lists.
        tp_group: Optional tensor-parallel process group used for logging.
        dp_cp_group: Optional data/context-parallel process group used for logging.

    Returns:
        Independent physical config clones for this segment and its global decoder-layer offset.
    """
    source_entries = tuple(main_layer_config_list)
    for index, entry in enumerate(source_entries):
        if entry is MTPSplit:
            raise ValueError(
                f"MTPSplit at index {index} is not valid in a parsed main layer config list"
            )
    source_configs = [entry for entry in source_entries if entry is not PipelineSplit]
    _validate_layer_config_sequence(source_configs, "main decoder")

    pp_rank = torch.distributed.get_rank(pp_group) if pp_group is not None else 0
    pp_size = torch.distributed.get_world_size(pp_group) if pp_group is not None else 1
    if (
        virtual_pipeline_model_parallel_size is not None
        and virtual_pipeline_model_parallel_size < 1
    ):
        raise ValueError("virtual_pipeline_model_parallel_size must be at least 1")

    has_pipeline_splits = any(entry is PipelineSplit for entry in source_entries)
    if has_pipeline_splits:
        if first_stage_layers is not None or last_stage_layers is not None:
            raise ValueError(
                "Cannot specify num_layers_in_first_pipeline_stage or "
                "num_layers_in_last_pipeline_stage when hybrid_layer_config_list contains "
                "PipelineSplit markers. The pipeline layout is already explicitly defined."
            )

        segments: list[list[TransformerConfig]] = [[]]
        for entry in source_entries:
            if entry is PipelineSplit:
                segments.append([])
            else:
                segments[-1].append(entry)

        vp_size = virtual_pipeline_model_parallel_size or 1
        expected_segment_count = pp_size * vp_size
        if len(segments) != expected_segment_count:
            raise ValueError(
                f"hybrid_layer_config_list defines {len(segments)} PipelineSplit-delimited "
                f"segments, but the configured PP x VPP topology requires "
                f"{expected_segment_count} ({pp_size} x {vp_size})"
            )
        if virtual_pipeline_model_parallel_size is None:
            if vp_stage is not None:
                raise ValueError(
                    "vp_stage must be None when virtual_pipeline_model_parallel_size is None"
                )
            vp_rel = 0
        else:
            if vp_stage is None:
                raise ValueError(
                    "vp_stage must be provided when virtual_pipeline_model_parallel_size is set"
                )
            if not 0 <= vp_stage < virtual_pipeline_model_parallel_size:
                raise ValueError(
                    f"vp_stage {vp_stage} is outside the configured virtual pipeline range "
                    f"[0, {virtual_pipeline_model_parallel_size})"
                )
            vp_rel = vp_stage

        segment_index = vp_rel * pp_size + pp_rank
        layer_offset = sum(len(segment) for segment in segments[:segment_index])
        selected_source_configs = segments[segment_index]
        layer_config_list = clone_hybrid_layer_config_list(selected_source_configs)
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f"HybridModel config list: pp_rank={pp_rank}/{pp_size}, vp_stage={vp_rel}, "
            f"segment_index={segment_index}/{len(segments)}, "
            f"layers={len(layer_config_list)}, layer_offset={layer_offset}",
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )
        return layer_config_list, layer_offset

    if virtual_pipeline_model_parallel_size is not None or vp_stage is not None:
        raise ValueError(
            "Virtual pipeline parallelism is not supported when hybrid_layer_config_list has no "
            "PipelineSplit markers"
        )

    num_layers = len(source_configs)
    if pp_size > 1:
        if first_stage_layers is not None or last_stage_layers is not None:
            first = first_stage_layers or 0
            last = last_stage_layers or 0
            middle_num_layers = num_layers - first - last
            middle_stages = pp_size - sum(
                1 for count in (first_stage_layers, last_stage_layers) if count is not None
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
                    f"by pipeline-model-parallel-size ({pp_size}) when no PipelineSplit "
                    "markers are specified in hybrid_layer_config_list."
                )
            count = num_layers // pp_size
            offset = pp_rank * count
    else:
        offset = 0
        count = num_layers

    selected_source_configs = source_configs[offset : offset + count]
    layer_config_list = clone_hybrid_layer_config_list(selected_source_configs)
    log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        f"HybridModel config list: pp_rank={pp_rank}/{pp_size}, vp_stage=None, "
        f"layers={len(layer_config_list)}, layer_offset={offset} (auto-split)",
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )
    return layer_config_list, offset


def get_layer_maps_from_layer_type_list(layer_type_list: list[str]) -> dict[str, dict[int, int]]:
    """
    Returns maps from global layer index to the corresponding layer index
    for each valid layer type (the keys of Symbols.LAYER_CONFIG_MAP) given a layer type list.
    """
    layer_types = [symbol for symbol in Symbols.name_sorted_valid_layer_symbols()]
    layer_maps = {layer_type: {} for layer_type in layer_types}
    for global_layer_idx, layer_type in enumerate(layer_type_list):
        layer_map = layer_maps[layer_type]
        local_layer_idx = len(layer_map)
        layer_map[global_layer_idx] = local_layer_idx
    return layer_maps


def get_layer_type_list_from_layer_config_list(
    layer_config_list: Sequence[TransformerConfig],
) -> list[str]:
    """Return the layer symbols corresponding to a sequence of layer configs.

    This compatibility projection keeps ``layer_config_list`` as the source of truth while
    supporting callers that still read ``HybridStack.layer_type_list``.

    Args:
        layer_config_list: Per-layer configs in layer order.

    Returns:
        The canonical layer symbol for each config.
    """
    return [
        layer_utils.get_layer_symbol_from_config(layer_config) for layer_config in layer_config_list
    ]
