# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from enum import Enum
from typing import List, TypeAlias

import numpy

from ..utils import log_single_rank

logger = logging.getLogger(__name__)

Blend: TypeAlias = (
    tuple[list[str], list[float] | None] | tuple[list[str], list[float] | None, list[float]]
)
"""Dataset prefixes, optional weights, and optional per-source train probabilities."""


class Split(Enum):
    """Dataset split identifiers used by the data builders."""

    train = 0
    valid = 1
    test = 2


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""
    import os
    import subprocess

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]
    if subprocess.run(command).returncode != 0:
        import sys

        log_single_rank(logger, logging.ERROR, "Failed to compile the C++ dataset helper functions")
        sys.exit(1)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """

    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def _try_float(value: str | float | int) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if numpy.isfinite(parsed) else None


def _parse_weighted_blend(blend: list[str | float | int], tuple_size: int) -> Blend | None:
    """Parse a complete weighted twople or triple blend, returning None on any mismatch."""
    if not blend or len(blend) % tuple_size != 0:
        return None

    weights = []
    prefixes = []
    probabilities = []
    for offset in range(0, len(blend), tuple_size):
        weight = _try_float(blend[offset])
        raw_prefix = blend[offset + 1]
        if not isinstance(raw_prefix, str):
            return None
        prefix = raw_prefix.strip()
        probability = _try_float(blend[offset + 2]) if tuple_size == 3 else None

        if weight is None or weight < 0.0 or not prefix or _try_float(prefix) is not None:
            return None
        if tuple_size == 3 and (probability is None or not 0.0 < probability <= 1.0):
            return None

        weights.append(weight)
        prefixes.append(prefix)
        if probability is not None:
            probabilities.append(probability)

    if not any(weight > 0.0 for weight in weights):
        return None

    if tuple_size == 3:
        return prefixes, weights, probabilities
    return prefixes, weights


def get_blend_from_list(
    blend: list[str | float | int] | None, *, allow_per_source_downsampling: bool = False
) -> Blend | None:
    # pylint: disable=line-too-long
    """Get the blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend
    from the blend list

    Args:
        blend (list[str | float | int] | None): The blend list, which can be either
            (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or
            (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"], or
            (3) a flattened, zipped list of weights, prefixes, and per-source train downsample probabilities, e.g. ["30", "path/to/dataset_1_prefix", "0.5", "70", "path/to/dataset_2_prefix", "0.25"]
        allow_per_source_downsampling (bool): Whether to accept form (3). Defaults to False.

    Returns:
        Blend | None: Prefixes, optional weights, and optional per-source train probabilities.
    """
    # pylint: enable=line-too-long
    if blend is None:
        return None

    if not blend:
        raise ValueError("blend must contain at least one dataset prefix")
    if len(blend) == 1 and isinstance(blend[0], str) and blend[0].strip():
        return [blend[0].strip()], None

    triple_blend = _parse_weighted_blend(blend, 3)
    twople_blend = _parse_weighted_blend(blend, 2)

    if triple_blend is not None:
        if not allow_per_source_downsampling:
            raise ValueError("per-source downsampling is not enabled for this blend")
        return triple_blend
    if twople_blend is not None:
        # Preserve the original twople representation for legacy weighted blends.
        return twople_blend

    # Preserve the legacy unweighted list form, but only when every item is unambiguously a
    # non-empty, non-numeric prefix. Numeric tokens indicate a malformed weighted blend.
    prefixes = [prefix.strip() if isinstance(prefix, str) else "" for prefix in blend]
    if all(prefix and _try_float(prefix) is None for prefix in prefixes):
        return prefixes, None

    raise ValueError(
        "blend must be entirely unweighted prefixes, (weight, prefix) twoples, or "
        "(weight, prefix, downsample_probability) triples"
    )
