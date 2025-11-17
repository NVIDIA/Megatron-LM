# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Dict, List, Tuple

if __name__ != "__main__":
    from megatron.core.utils import log_single_rank
else:
    from typing import Any

    import torch

    def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
        """Logs a message to the given rank."""
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == rank:
                logger.log(*args, **kwargs)
        else:
            logger.log(*args, **kwargs)


logger = logging.getLogger(__name__)


class Symbols:
    """Symbols for different layer types."""

    MAMBA = "M"
    ATTENTION = "*"
    MLP = "-"
    VALID = {MAMBA, ATTENTION, MLP}


def _allocate_auto(
    total_layers_count: int, target_attention_ratio: float, target_mlp_ratio: float
) -> list:
    # First, allocate attention (evenly spaced, starting and ending with mamba)
    attention_layers_count: int = round(total_layers_count * target_attention_ratio)
    mamba_layers_count: int = total_layers_count - attention_layers_count
    mamba_sections_count: int = attention_layers_count + 1
    mamba_section_length: float = mamba_layers_count / mamba_sections_count

    layer_type_list = [Symbols.MAMBA] * total_layers_count
    x: float = mamba_section_length
    for l in range(total_layers_count):
        if x < 0.5:
            layer_type_list[l] = Symbols.ATTENTION
            x += mamba_section_length
        else:
            x -= 1

    # Next, allocate mlp
    # (evenly distributed, but right-justified, not replacing attention)
    mlp_layers_count: int = round(total_layers_count * target_mlp_ratio)
    if mlp_layers_count > 0:
        mamba_layers_count -= mlp_layers_count
        mamba_to_mlp_ratio: float = mamba_layers_count / mlp_layers_count

        x: float = mamba_to_mlp_ratio
        for l in range(total_layers_count):
            if layer_type_list[l] == Symbols.MAMBA:
                if x < 0.5:
                    layer_type_list[l] = Symbols.MLP
                    x += mamba_to_mlp_ratio
                else:
                    x -= 1

    return layer_type_list


def _allocate_override(total_layers_count: int, override_pattern: str) -> list:
    layer_type_list = list(override_pattern)
    override_pattern_length = len(layer_type_list)
    if override_pattern_length != total_layers_count:
        raise ValueError(
            "The hybrid override pattern is the wrong "
            f"length: got {override_pattern_length}, expected "
            f"{total_layers_count}"
        )
    for l in layer_type_list:
        if l not in Symbols.VALID:
            raise ValueError(f"In hybrid override pattern, '{l}' is not one of {Symbols.VALID}")

    return layer_type_list


def _layer_counts_match(a: list, b: list) -> bool:
    for s in Symbols.VALID:
        if a.count(s) != b.count(s):
            return False
    return True


def allocate_layers(
    total_layers_count: int,
    target_attention_ratio: float,
    target_mlp_ratio: float,
    override_pattern: str = None,
) -> list:
    """Allocates layers according to the requested distribution of layer types."""
    assert total_layers_count > 0
    assert target_attention_ratio >= 0.0 and target_attention_ratio <= 1.0
    assert target_mlp_ratio >= 0.0 and target_mlp_ratio <= 1.0
    assert target_attention_ratio + target_mlp_ratio <= 1.0
    # Note: target_mamba_ratio = 1.0 - target_attention_ratio - target_mlp_ratio

    layer_type_list = _allocate_auto(total_layers_count, target_attention_ratio, target_mlp_ratio)

    if override_pattern is not None:
        layer_type_list_override = _allocate_override(total_layers_count, override_pattern)
        log_single_rank(logger, logging.INFO, "Using hybrid override pattern")
        if (target_attention_ratio > 0.0 or target_mlp_ratio > 0.0) and not _layer_counts_match(
            layer_type_list_override, layer_type_list
        ):
            raise ValueError(
                "The number of each type of layer in the override "
                "pattern must match the number in the overridden "
                "pattern."
            )
        if layer_type_list_override == layer_type_list:
            log_single_rank(
                logger, logging.INFO, "The override pattern matches the overridden pattern"
            )
        else:
            log_single_rank(logger, logging.INFO, "Warning: overriding pattern A with pattern B")
            log_single_rank(logger, logging.INFO, f"A: {''.join(layer_type_list)}")
            log_single_rank(logger, logging.INFO, f"B: {''.join(layer_type_list_override)}")
        layer_type_list = layer_type_list_override

    if target_attention_ratio > 0.0 or target_mlp_ratio > 0.0 or override_pattern is not None:
        actual_attention_layers_count = layer_type_list.count(Symbols.ATTENTION)
        actual_attention_ratio = actual_attention_layers_count / total_layers_count
        actual_mlp_layers_count = layer_type_list.count(Symbols.MLP)
        actual_mlp_ratio = actual_mlp_layers_count / total_layers_count
        allocation_string = "".join(layer_type_list)
        log_single_rank(
            logger,
            logging.INFO,
            f"Hybrid allocation ({Symbols.MAMBA} is mamba, "
            f"{Symbols.ATTENTION} is attention, "
            f"{Symbols.MLP} is mlp):",
        )
        log_single_rank(logger, logging.INFO, allocation_string)
        log_single_rank(
            logger,
            logging.INFO,
            f"{actual_attention_layers_count} attention layers in "
            f"{total_layers_count} total layers.",
        )
        log_single_rank(
            logger,
            logging.INFO,
            f"Target attention ratio: {target_attention_ratio:.2f}. "
            f"Actual attention ratio: {actual_attention_ratio:.2f}.",
        )
        log_single_rank(
            logger,
            logging.INFO,
            f"{actual_mlp_layers_count} mlp layers in " f"{total_layers_count} total layers.",
        )
        log_single_rank(
            logger,
            logging.INFO,
            f"Target mlp ratio: {target_mlp_ratio:.2f}. "
            f"Actual mlp ratio: {actual_mlp_ratio:.2f}.",
        )
    return layer_type_list


def get_layer_maps_from_layer_type_list(
    layer_type_list: List[str],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Returns maps from global layer index to the corresponding layer index
    for each layer type in [Attention, Mamba, MLP] given a layer type list.
    """
    layer_types = [Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP]
    layer_maps = {layer_type: {} for layer_type in layer_types}
    for global_layer_idx, layer_type in enumerate(layer_type_list):
        layer_map = layer_maps[layer_type]
        local_layer_idx = len(layer_map)
        layer_map[global_layer_idx] = local_layer_idx
    return [layer_maps[layer_type] for layer_type in layer_types]


if __name__ == "__main__":
    test_cases = [
        # (10, 0.2, 0.0),
        # (48, 0.0, 0.0), # will not print anything
        # (48, 0.1, 0.0),
        # 48, 0.3, 0.0),
        # (48, 0.5, 0.0),
        # (48, 0.6, 0.0),
        # (48, 0.7, 0.0),
        # (10, 0.0, 0.1),
        # (10, 0.0, 0.3),
        # (10, 0.0, 0.5),
        # (10, 0.1, 0.1),
        # (10, 0.2, 0.2),
        # (10, 0.3, 0.3),
        # (10, 0.5, 0.5),
        # (48, 0.2, 0.3),
        # (48, 0.5, 0.2),
        # (48, 0.5, 0.2, "MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-"),
        # (48, 0.25, 0.25, "MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-"),
        # (48, 0.25, 0.25, "MM-*MM-*MM*-MM*-MM*-MM*-M*M-M*M-M*M-M*M-*MM-*MM-"),
        # (48, 0.0, 0.2, "MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-"),
        # (48, 0.2, 0.0, "MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-"),
        # (48, 0.0, 0.0, "MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-MM*-"),
        # (48, 0.5, 0.5),
        # (10, 0.3, 0.2, "MMM*-*M*M-"),
        # (10, 0.3, 0.2, "MM*M-*M*M-"),
        (9, 0.0, 0.0, "M*-M*-M*-"),
        (9, 0.0, 0.0, "MMMMMMMMM"),
    ]
    for t in test_cases:
        logging.info("")
        allocate_layers(*t)
