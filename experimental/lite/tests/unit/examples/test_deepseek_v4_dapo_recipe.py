# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "verl"
    / "scripts"
    / "validate_deepseek_v4_dapo.py"
)
SPEC = importlib.util.spec_from_file_location("validate_deepseek_v4_dapo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


@pytest.mark.parametrize(
    ("scale_format", "expected"),
    [(None, "float32"), ("float32", "float32"), ("ue8m0", "ue8m0")],
)
def test_rollout_fp8_scale_format_follows_source_checkpoint(
    tmp_path, scale_format, expected
):
    quantization_config = {"quant_method": "fp8"}
    if scale_format is not None:
        quantization_config["scale_fmt"] = scale_format
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"quantization_config": quantization_config}), encoding="utf-8"
    )

    assert VALIDATOR.resolve_rollout_scale_format(config, 8) == expected


def test_rollout_mxfp4_scale_format_is_ue8m0(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    assert VALIDATOR.resolve_rollout_scale_format(config, 4) == "ue8m0"
