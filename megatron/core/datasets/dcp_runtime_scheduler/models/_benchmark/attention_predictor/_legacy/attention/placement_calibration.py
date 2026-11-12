# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Explicit, scope-checked whole-call fits for measured physical ring layouts."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


def branch_key(cp, partition, phase):
    return f"{cp}:" + ",".join(map(str, partition)) + f":{phase}"


GEOMETRY_FEATURES = ("critical_rank_work_m", "round_critical_work_m",
    "critical_rank_sqrt_work", "round_critical_sqrt_work", "valid_tokens_k",
    "sequence_count", "max_length_k")


def features(lengths, cp, storage, base_ms, *, include_geometry=False):
    capacity = cp * storage
    result = dict(constant=1.0, base_ms=base_ms, total=sum(lengths)/capacity,
                squares=sum(n*n for n in lengths)/(capacity*capacity),
                count=float(len(lengths)))
    if include_geometry:
        from .backward_model.backward_attention_predictor import extract_backward_features
        geometry = extract_backward_features(lengths, cp, storage, 2*cp)
        result.update({name: geometry[name] for name in GEOMETRY_FEATURES})
    return result


@lru_cache(maxsize=8)
def load_calibration(path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1 or payload.get("method") not in (
            "whole_call_residual", "paired_placement_residual"):
        raise ValueError("unsupported placement calibration schema")
    if payload.get("target_statistic") != "wall_p10_ms":
        raise ValueError("placement calibration must match the frozen wall_p10_ms timing target")
    if not payload.get("branches") or not payload.get("software_provenance"):
        raise ValueError("placement calibration needs branches and software provenance")
    allowed = set(features([1], 1, 1, 1)) | set(GEOMETRY_FEATURES)
    for key, branch in payload["branches"].items():
        coefficients = branch["coefficients"]
        if not coefficients or set(coefficients)-allowed:
            raise ValueError("unknown or empty placement calibration features")
        if not all(math.isfinite(float(v)) for v in coefficients.values()):
            raise ValueError("nonfinite placement calibration coefficient")
        if branch.get("calibration_cases", 0) < 1 or branch.get("holdout_cases", 0) < 1:
            raise ValueError("placement calibration needs a disjoint holdout")
        if type(branch.get("validation_passed")) is not bool:
            raise ValueError("placement calibration requires a boolean validation result")
        floor = branch.get("time_floor_ms")
        if payload["method"] == "paired_placement_residual":
            if floor is not None:
                raise ValueError("paired placement correction cannot change the absolute timing floor")
            cp = int(key.split(":")[0])
            reference = tuple(rank // int(payload["scope"]["domain_size"]) for rank in range(cp))
            if key == branch_key(cp, reference, key.rsplit(":", 1)[1]) and any(coefficients.values()):
                raise ValueError("paired placement reference must preserve the frozen baseline exactly")
            if "paired_holdout_error" not in branch or type(branch.get("placement_validation_passed")) is not bool:
                raise ValueError("paired placement needs separate placement and absolute validation")
        if floor is not None and (not math.isfinite(float(floor)) or float(floor) < 0):
            raise ValueError("invalid placement time floor")
        if (set(coefficients) | set(branch["feature_bounds"])) & set(GEOMETRY_FEATURES):
            if branch["sequence_alignment"] <= 0:
                raise ValueError("invalid geometry feature sequence alignment")
        for name, bounds in branch["feature_bounds"].items():
            if name not in allowed or len(bounds) != 2 or not all(math.isfinite(float(x)) for x in bounds):
                raise ValueError("invalid placement feature bounds")
            if bounds[0] > bounds[1]:
                raise ValueError("reversed placement feature bounds")
    return payload, hashlib.sha256(raw).hexdigest()


def validate_scope(model):
    from .estimator import verify_calibration
    payload, digest = load_calibration(model.placement_calibration)
    if payload.get("base_calibration_sha256") != verify_calibration():
        raise ValueError("placement correction was fitted against different frozen coefficients")
    scope = payload["scope"]
    expected = dict(hardware=model.hardware, domain_size=model.domain_size or
                    {"h100": 8, "gb200": 72}[model.hardware],
                    geometry=[model.query_heads, model.kv_heads, model.head_dim],
                    storage_per_rank=model.physical_tokens_per_rank,
                    model_family=model.model_family, cp1_branch=model.cp1_branch,
                    cp_implementation=model.cp_implementation,
                    transport_backend=model.transport_backend)
    if any(scope.get(key) != value for key, value in expected.items()):
        raise ValueError("placement calibration does not match estimator hardware/geometry/stack scope")
    return payload, digest


def apply_calibration(model, lengths, cp, placement, alignment, phase, original):
    payload, digest = validate_scope(model)
    key = branch_key(cp, placement.partition, phase)
    if key not in payload["branches"]:
        raise ValueError(f"placement calibration has no measured branch for {key}")
    branch = payload["branches"][key]
    if branch["sequence_alignment"] != alignment:
        raise ValueError("placement calibration sequence alignment mismatch")
    need_geometry = bool((set(branch["coefficients"]) | set(branch["feature_bounds"])) & set(GEOMETRY_FEATURES))
    if need_geometry and alignment != 2*cp:
        raise ValueError("placement geometry features currently require native 2*CP alignment")
    values = features(lengths, cp, model.physical_tokens_per_rank, original["time_ms"],
                      include_geometry=need_geometry)
    correction = sum(float(weight)*values[name] for name, weight in branch["coefficients"].items())
    time_ms = original["time_ms"] + correction
    if branch.get("time_floor_ms") is not None:
        time_ms = max(float(branch["time_floor_ms"]), time_ms)
        correction = time_ms-original["time_ms"]
    if not math.isfinite(time_ms) or time_ms <= 0:
        raise ValueError("placement calibration produced a nonpositive or nonfinite time")
    outside = [name for name, (lo, hi) in branch["feature_bounds"].items()
               if values[name] < lo - 1e-9 or values[name] > hi + 1e-9]
    # Reference fits are fitted independently too; this is not a measurement of
    # just NCCL latency and must never be described as a native transport model.
    return dict(original, time_ms=time_ms, provisional=True,
        placement_calibration=dict(sha256=digest, branch=key, correction_ms=correction,
            target_statistic=payload["target_statistic"],
            time_floor_ms=branch.get("time_floor_ms"),
            reference_frozen_ms=original["time_ms"], method=payload["method"],
            calibration_cases=branch["calibration_cases"], holdout_cases=branch["holdout_cases"],
            holdout_error=branch["holdout_error"], validation_passed=branch["validation_passed"],
            paired_holdout_error=branch.get("paired_holdout_error"),
            placement_validation_passed=branch.get("placement_validation_passed"),
            outside_features=outside, software_provenance=payload["software_provenance"]))
