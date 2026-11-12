# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Read and validate the single-file runtime configuration without training dependencies."""

import hashlib
import json
import math
from pathlib import Path

# ``type`` selects a whole scheduler: one cost model bound to one solver. This table is
# the only place that defines the binding and the keys each type accepts.
SCHEDULERS = {
    "dcp_benchmark": {
        "solver": "lightweight",
        "required": ("hardware", "non_attention"),
        "optional": (),
    },
    # ``hardware`` selects a packaged measured calibration instead of the original profile.
    "dynamic_usp": {
        "solver": "dynamic_usp",
        "required": (),
        "optional": ("hardware", "calibration"),
    },
}
COMMON_KEYS = ("max_cp_size", "async", "trace_dir")
NON_ATTENTION_KEYS = ("forward_ms_per_layer", "backward_ms_per_layer")
# Largest accepted non-attention time. Real values are far below it; one above it is a
# unit mistake (for example nanoseconds), which the timeline arithmetic cannot represent.
MAX_NON_ATTENTION_MS = 1e5


def _unique_object(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Duplicate configuration key: {key}")
        values[key] = value
    return values


def _reject_constant(value):
    raise ValueError(f"Configuration values must be finite: {value}")


def _milliseconds(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(
            f"{label} must be a finite nonnegative number of milliseconds per layer, per "
            "microbatch, per rank"
        )
    if value > MAX_NON_ATTENTION_MS:
        raise ValueError(
            f"{label} is {value}; the unit is milliseconds per layer, per microbatch, per "
            f"rank, and at most {MAX_NON_ATTENTION_MS:g} is accepted"
        )
    return value


def _expanded_path(text, label):
    """Expand ``~`` in a path from the file; an unknown user is a configuration error."""
    try:
        return Path(text).expanduser()
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{label} {text!r} cannot be resolved: {error}") from error


def _per_layer_times(value, label):
    """One number for every layer, or one number per layer type named by the cost model."""
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{label} must be a number or a nonempty layer-type mapping")
        return {kind: _milliseconds(item, f"{label}.{kind}") for kind, item in value.items()}
    return _milliseconds(value, label)


def allowed_keys(scheduler_type):
    """Top-level keys accepted for one scheduler type, in documentation order."""
    entry = SCHEDULERS[scheduler_type]
    return ("type",) + entry["required"] + entry["optional"] + COMMON_KEYS


def validate_runtime_config(data, config_path="<runtime configuration>"):
    """Check the complete user-facing schema and return normalized settings.

    ``config_path`` labels errors, anchors a relative ``calibration`` path and is recorded
    as the provenance of the user's non-attention measurements.
    """
    where = f"Runtime configuration {str(config_path)!r}"
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a JSON object")
    scheduler_type = data.get("type")
    if not isinstance(scheduler_type, str) or scheduler_type not in SCHEDULERS:
        raise ValueError(
            f"{where}: 'type' must be one of {sorted(SCHEDULERS)}, got {scheduler_type!r}"
        )
    allowed = allowed_keys(scheduler_type)
    unknown = sorted(set(data) - set(allowed), key=str)
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown} for type {scheduler_type!r}; "
            f"allowed keys: {list(allowed)}"
        )
    missing = [key for key in SCHEDULERS[scheduler_type]["required"] if key not in data]
    if missing:
        raise ValueError(f"{where}: type {scheduler_type!r} requires key(s) {missing}")

    maximum = data.get("max_cp_size")
    if "max_cp_size" in data and (type(maximum) is not int or maximum < 1):
        raise ValueError(f"{where}: max_cp_size must be a positive integer, got {maximum!r}")
    asynchronous = data.get("async", True)
    if type(asynchronous) is not bool:
        raise ValueError(f"{where}: async must be true or false, got {asynchronous!r}")
    trace_dir = data.get("trace_dir")
    if "trace_dir" in data:
        if not isinstance(trace_dir, str) or not trace_dir.strip():
            raise ValueError(f"{where}: trace_dir must be a nonempty directory path")
        # A relative directory stays relative to the working directory of the run.
        trace_dir = str(_expanded_path(trace_dir, f"{where}: trace_dir"))

    options = {}
    if "hardware" in data:
        hardware = data["hardware"]
        if not isinstance(hardware, str) or not hardware.strip():
            raise ValueError(f"{where}: hardware must be a nonempty string, got {hardware!r}")
        options["hardware"] = hardware
    if "non_attention" in data:
        non_attention = data["non_attention"]
        if not isinstance(non_attention, dict):
            raise ValueError(
                f"{where}: non_attention must be an object with the keys {list(NON_ATTENTION_KEYS)}"
            )
        unknown = sorted(set(non_attention) - set(NON_ATTENTION_KEYS), key=str)
        missing = [key for key in NON_ATTENTION_KEYS if key not in non_attention]
        if unknown or missing:
            raise ValueError(
                f"{where}: non_attention takes exactly the keys {list(NON_ATTENTION_KEYS)}"
                + (f"; unknown: {unknown}" if unknown else "")
                + (f"; missing: {missing}" if missing else "")
            )
        options["non_attention"] = {
            key: _per_layer_times(non_attention[key], f"{where}: non_attention.{key}")
            for key in NON_ATTENTION_KEYS
        }
        # Provenance is recorded, never requested from the user.
        options["non_attention"]["source"] = str(config_path)
    if "calibration" in data:
        calibration = data["calibration"]
        if not isinstance(calibration, str) or not calibration.strip():
            raise ValueError(f"{where}: calibration must be a nonempty table path")
        calibration_path = _expanded_path(calibration, f"{where}: calibration")
        if not calibration_path.is_absolute():
            calibration_path = Path(config_path).parent / calibration_path
        try:
            options["calibration"] = str(calibration_path.resolve())
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"{where}: calibration {calibration!r} cannot be resolved: {error}"
            ) from error
    return {
        "type": scheduler_type,
        "options": options,
        "max_cp_size": maximum,
        "async": asynchronous,
        "trace_dir": trace_dir,
    }


def load_runtime_config(path):
    """Read ``--dynamic-cp-runtime-config`` and validate its schema.

    The returned settings do not depend on the training configuration; binding them to
    the model and parallel geometry is ``dcp_schedule_prefetch.runtime_spec``.
    """
    try:
        config_path = Path(path).expanduser().resolve()
        raw = config_path.read_bytes()
        data = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (OSError, TypeError, UnicodeError, ValueError, RuntimeError, RecursionError) as error:
        # RuntimeError: "~user" without such a user. RecursionError: values nested too deeply.
        raise ValueError(f"Cannot read runtime configuration {path!r}: {error}") from error
    settings = validate_runtime_config(data, config_path)
    settings["config_path"] = str(config_path)
    settings["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return settings
