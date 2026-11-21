# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline layouts as lists of stages of unit characters.

A unit is one of ``E`` (embedding), ``t`` (decoder layer), ``m`` (MTP layer) and ``L`` (loss,
including the output layer), the symbols of ``--pipeline-model-parallel-layout``. A layout is a
list of stages in model order (virtual-stage major), each a list of unit characters. Parsing and
validation are delegated to :class:`PipelineParallelLayerLayout`, so the planner follows the same
rules as training.
"""

from itertools import combinations
from typing import Iterator, List

from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

Stages = List[List[str]]

UNITS = ("E", "t", "m", "L")
_CHAR_TO_NAME = {"E": "embedding", "t": "decoder", "m": "mtp", "L": "loss"}
_TYPE_TO_CHAR = {
    LayerType.embedding: "E",
    LayerType.decoder: "t",
    LayerType.mtp: "m",
    LayerType.loss: "L",
}


def parse(layout_str: str) -> Stages:
    """Parse a ``--pipeline-model-parallel-layout`` string into stages of unit characters.

    Args:
        layout_str: Layout string, for example ``"Et*3|(tt|)*29,m|L"``.

    Returns:
        The stages in model order.
    """
    parsed = PipelineParallelLayerLayout.parse_str_to_list(layout_str)
    return [[_TYPE_TO_CHAR[layer] for layer in stage] for stage in parsed]


def _compact(stage: List[str]) -> str:
    parts, i = [], 0
    while i < len(stage):
        j = i
        while j < len(stage) and stage[j] == stage[i]:
            j += 1
        parts.append(stage[i] if j - i == 1 else f"{stage[i]}*{j - i}")
        i = j
    return "".join(parts)


def to_string(stages: Stages) -> str:
    """Return the layout string for ``stages``, with runs written as ``x*n``.

    Args:
        stages: Stages of unit characters.

    Returns:
        A string accepted by ``--pipeline-model-parallel-layout``.
    """
    return "|".join(_compact(stage) for stage in stages)


def validate(stages: Stages, pp_size: int, num_layers: int, mtp_num_layers: int = 0) -> None:
    """Validate ``stages`` with Megatron's own layout checks.

    Args:
        stages: Stages of unit characters.
        pp_size: Pipeline-model-parallel size.
        num_layers: Number of decoder layers.
        mtp_num_layers: Number of MTP layers.

    Raises:
        AssertionError: If Megatron would reject the layout.
    """
    named = [[_CHAR_TO_NAME[unit] for unit in stage] for stage in stages]
    PipelineParallelLayerLayout(named, pp_size).validate_layer_layout(
        num_layers=num_layers, mtp_num_layers=mtp_num_layers
    )


def is_valid(stages: Stages, pp_size: int, num_layers: int, mtp_num_layers: int = 0) -> bool:
    """Return whether Megatron accepts ``stages``.

    Args:
        stages: Stages of unit characters.
        pp_size: Pipeline-model-parallel size.
        num_layers: Number of decoder layers.
        mtp_num_layers: Number of MTP layers.

    Returns:
        True if :func:`validate` passes.
    """
    try:
        validate(stages, pp_size, num_layers, mtp_num_layers)
    except AssertionError:
        return False
    return True


def megatron_default(num_layers: int, pp_size: int, mtp_num_layers: int = 0) -> Stages:
    """Return the layout Megatron builds without ``--pipeline-model-parallel-layout``.

    Decoder layers are split evenly, the embedding goes on the first stage, and the MTP layers
    and the loss go on the last stage.

    Args:
        num_layers: Number of decoder layers, divisible by ``pp_size``.
        pp_size: Pipeline-model-parallel size.
        mtp_num_layers: Number of MTP layers.

    Returns:
        The default stages.
    """
    if num_layers % pp_size:
        raise ValueError("The default split needs num_layers divisible by pp_size.")
    per_stage = num_layers // pp_size
    stages = [["t"] * per_stage for _ in range(pp_size)]
    stages[0].insert(0, "E")
    stages[-1].extend(["m"] * mtp_num_layers + ["L"])
    return stages


def count_based(num_layers: int, pp_size: int, mtp_num_layers: int = 0) -> Stages:
    """Split ``[E, t*N, m*K, L]`` into ``pp_size`` contiguous chunks of equal unit count.

    The first ``remainder`` stages get one extra unit. This treats the embedding, each MTP layer
    and the loss as one decoder layer each.

    Args:
        num_layers: Number of decoder layers.
        pp_size: Pipeline-model-parallel size.
        mtp_num_layers: Number of MTP layers.

    Returns:
        The count-based stages.
    """
    units = ["E"] + ["t"] * num_layers + ["m"] * mtp_num_layers + ["L"]
    base, remainder = divmod(len(units), pp_size)
    stages, pos = [], 0
    for stage_idx in range(pp_size):
        size = base + (1 if stage_idx < remainder else 0)
        stages.append(units[pos : pos + size])
        pos += size
    return stages


def enumerate_layouts(
    num_layers: int, pp_size: int, mtp_num_layers: int = 0, vpp_size: int = 1
) -> Iterator[Stages]:
    """Yield every contiguous, non-empty split of ``[E, t*N, m*K, L]`` that Megatron accepts.

    MTP layers are kept together because Megatron requires them in one virtual stage.

    Args:
        num_layers: Number of decoder layers.
        pp_size: Pipeline-model-parallel size.
        mtp_num_layers: Number of MTP layers.
        vpp_size: Virtual stages per pipeline rank.

    Yields:
        Valid stages, ``pp_size * vpp_size`` of them.
    """
    sequence = ["E"] + ["t"] * num_layers + (["M"] if mtp_num_layers else []) + ["L"]
    num_stages = pp_size * vpp_size
    for cuts in combinations(range(1, len(sequence)), num_stages - 1):
        bounds = (0,) + cuts + (len(sequence),)
        stages = []
        for start, end in zip(bounds, bounds[1:]):
            stage = []
            for unit in sequence[start:end]:
                stage.extend(["m"] * mtp_num_layers if unit == "M" else [unit])
            stages.append(stage)
        if is_valid(stages, pp_size, num_layers, mtp_num_layers):
            yield stages
