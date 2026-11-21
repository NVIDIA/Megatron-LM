# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from DynamicUSP at commit
# 1ff97a4ae0beb5e088fadb9e112265bdd194b24d (src/hot_switch_solver: gbs/config.py, weights.py,
# objectives/pipeline.py, models.py and validation.py), BSD-3-Clause license.
# See ../DYNAMIC_USP_NOTICE.txt for the upstream copyright and license text.

"""Optional ``solver`` section of a DynamicUSP coefficient file.

The section holds the coefficients of the scheduler that are not group times::

    "solver": {
        "sample_weight": {"A": ..., "B": ..., "C": ...},
        "tier_time_factors": {"8": 1.0, "16": 1.1, "32": 1.3},
        "pipeline": {
            "per_mb_overhead_ms": ..., "compute_fwd_share": ..., "comm_fwd_share": ...
        },
        "stage2": {"enabled": false, "keep_threshold": 8.4}
    }

The README of the package (Original profile) has the meaning and the default of every key;
the defaults are DynamicUSP's, with Stage 2 (``runtime.stage2``) off. Every part is
optional, and unknown keys are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..runtime.hot_switch import validate_tier_time_factors
from ..runtime.stage2 import DEFAULT_KEEP_THRESHOLD, _finite

# DynamicUSP's published measured fit of the forward cost of one sample in a group of eight
# ranks. It is the default of the original profile and is not derived from the packaged
# usp_calib_v1 table.
DEFAULT_SAMPLE_WEIGHT = (7706.03421944, 6528.86747592, 351.14154099)
_KEYS = ("sample_weight", "tier_time_factors", "pipeline", "stage2")
_STAGE2_KEYS = ("enabled", "keep_threshold")
_WEIGHT_KEYS = ("A", "B", "C")
_PIPELINE_KEYS = ("per_mb_overhead_ms", "compute_fwd_share", "comm_fwd_share")
# Megatron's two pipeline schedules.
PIPELINE_SCHEDULES = ("interleaved", "non_interleaved")


def pipeline_schedule(virtual_stages: int) -> str:
    """Name of the pipeline schedule of a run with more than one pipeline stage.

    ``virtual_stages`` is the number of model chunks on a pipeline stage as a planner
    specification gives it: two or more for a run with virtual pipeline stages, for which
    Megatron runs the interleaved schedule, and 1 for a run without, for which it runs the
    non-interleaved one.
    """
    return PIPELINE_SCHEDULES[0] if virtual_stages > 1 else PIPELINE_SCHEDULES[1]


@dataclass(frozen=True)
class SolverSection:
    """Validated content of the section; the defaults are those of an absent section."""

    sample_weight: tuple | None = None
    tier_time_factors: dict | None = None
    per_mb_overhead_ms: float = 0.0
    # Forward shares of compute and of communication time: validated and recorded. With one
    # pipeline stage they change no score.
    compute_fwd_share: float = 1.0 / 3.9
    comm_fwd_share: float = 0.5
    # The three terms above that the file does not give: for these the defaults apply.
    pipeline_default_terms: tuple = _PIPELINE_KEYS
    stage2_enabled: bool = False
    stage2_keep_threshold: float = DEFAULT_KEEP_THRESHOLD


def _object(value, label, keys, required=False):
    """``value`` as a JSON object with no key outside ``keys``; ``required``: with all of them."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    unknown = sorted(set(value) - set(keys), key=str)
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in {label}; allowed keys: {list(keys)}")
    missing = [key for key in keys if key not in value]
    if required and missing:
        raise ValueError(f"{label} requires the keys {list(keys)}; missing: {missing}")
    return value


def _number(value, label, low=-math.inf, high=math.inf, kind="a finite number"):
    if not _finite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be {kind}")
    return float(value)


def _stage2(value, offered) -> dict:
    """Validated ``solver.stage2``; rejects a switch that the calibration cannot honour."""
    stage2 = _object(value, "solver.stage2", _STAGE2_KEYS)
    values = {}
    if "enabled" in stage2:
        if type(stage2["enabled"]) is not bool:
            raise ValueError("solver.stage2.enabled must be true or false")
        values["stage2_enabled"] = stage2["enabled"]
    if "keep_threshold" in stage2:
        values["stage2_keep_threshold"] = _number(
            stage2["keep_threshold"],
            "solver.stage2.keep_threshold",
            low=math.ulp(0.0),
            kind="a finite positive number",
        )
    enabled = values.get("stage2_enabled", False)
    # Without Stage 2 loads the object may only say "off": a threshold alone asks for more.
    if not offered and (enabled or "enabled" not in stage2 and "keep_threshold" in stage2):
        raise ValueError(
            "solver.stage2 is not offered for this calibration yet (it defines no Stage 2 "
            'sample loads): leave the object out or set "enabled" to false'
        )
    return values


def parse_solver_section(
    table, cp_sizes, *, pipeline_field=None, stage2_offered=True
) -> SolverSection:
    """Validate ``table["solver"]``; a table without the section gets the defaults.

    ``cp_sizes`` are the CP sizes of the cost model: ``tier_time_factors`` names no other
    size. A calibration that has its own per-microbatch overhead passes the name of that
    field as ``pipeline_field``, and ``solver.pipeline`` is then rejected. A calibration
    without Stage 2 sample loads passes ``stage2_offered=False`` (``_stage2``).
    """
    if "solver" not in table:
        return SolverSection()
    section = _object(table["solver"], "the solver section of the coefficient file", _KEYS)
    values: dict = {}
    if "sample_weight" in section:
        weight = _object(
            section["sample_weight"], "solver.sample_weight", _WEIGHT_KEYS, required=True
        )
        values["sample_weight"] = tuple(
            _number(weight[key], f"solver.sample_weight.{key}") for key in _WEIGHT_KEYS
        )
    if "tier_time_factors" in section:
        try:
            factors = validate_tier_time_factors(section["tier_time_factors"])
        except ValueError as error:
            raise ValueError(f"solver.{error}") from error
        other = sorted(set(factors) - set(cp_sizes))
        if other:
            raise ValueError(
                "solver.tier_time_factors keys must be CP sizes of the cost model "
                f"{list(cp_sizes)}; got {other}"
            )
        values["tier_time_factors"] = factors
    if "pipeline" in section:
        if pipeline_field is not None:
            raise ValueError(
                "solver.pipeline is not accepted by this calibration: its per-microbatch "
                f"overhead is {pipeline_field}"
            )
        pipeline = _object(section["pipeline"], "solver.pipeline", _PIPELINE_KEYS)
        for key, high, kind in (
            ("per_mb_overhead_ms", math.inf, "a finite nonnegative number"),
            ("compute_fwd_share", 1.0, "a number in [0, 1]"),
            ("comm_fwd_share", 1.0, "a number in [0, 1]"),
        ):
            if key in pipeline:
                values[key] = _number(pipeline[key], f"solver.pipeline.{key}", 0.0, high, kind)
        values["pipeline_default_terms"] = tuple(
            key for key in _PIPELINE_KEYS if key not in pipeline
        )
    if "stage2" in section:
        values.update(_stage2(section["stage2"], stage2_offered))
    return SolverSection(**values)


def sample_weight(length, coefficients) -> float:
    """Stage 1 weight of one sample of ``length`` tokens, floored at one."""
    a, b, c = coefficients
    weight = a * (length * length / 1e10) + b * (length / 1e5) + c
    if not math.isfinite(weight):
        raise ValueError("solver.sample_weight gives a Stage 1 weight that is not finite")
    return max(1.0, float(weight))
