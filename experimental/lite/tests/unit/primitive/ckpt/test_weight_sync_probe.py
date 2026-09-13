# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
import importlib.util
import sys
import types

import torch

if importlib.util.find_spec("safetensors") is None:
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = None
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = None
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors_torch

from megatron.lite.primitive.ckpt import hf_weights
from megatron.lite.primitive.ckpt.weight_sync_probe import (
    get_weight_sync_probe,
    weight_sync_probe_session,
)


def _report_from_stdout(stdout: str) -> dict:
    line = next(
        line
        for line in stdout.splitlines()
        if line.startswith("MLITE_WEIGHT_SYNC_PROBE ")
    )
    return json.loads(line.split(" ", 1)[1])


def test_probe_is_silent_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv("MLITE_WEIGHT_SYNC_PROBE", raising=False)

    with weight_sync_probe_session("mlite"):
        with get_weight_sync_probe().measure("mapping") as sample:
            sample.nbytes = 16

    assert capsys.readouterr().out == ""


def test_probe_reports_nested_stage_totals(monkeypatch, capsys):
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "1")

    with weight_sync_probe_session("mlite"):
        with get_weight_sync_probe().measure("fsdp_gather") as sample:
            sample.nbytes = 32
        with get_weight_sync_probe().measure("fsdp_gather") as sample:
            sample.nbytes = 64

    report = _report_from_stdout(capsys.readouterr().out)
    assert report["backend"] == "mlite"
    assert report["stages"]["fsdp_gather"]["calls"] == 2
    assert report["stages"]["fsdp_gather"]["bytes"] == 96
    assert report["stages"]["fsdp_gather"]["wall_s"] >= 0.0
    assert report["total_wall_s"] >= report["stages"]["fsdp_gather"]["wall_s"]


def test_cpu_staging_only_counts_device_to_host(monkeypatch, capsys):
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "true")
    tensor = torch.arange(4, dtype=torch.float32)

    with weight_sync_probe_session("mlite"):
        result = hf_weights._to_cpu(tensor)

    assert result is tensor
    report = _report_from_stdout(capsys.readouterr().out)
    assert "d2h" not in report["stages"]


def test_mapping_counts_output_payload(monkeypatch, capsys):
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "1")
    tensor = torch.ones(2, dtype=torch.bfloat16)

    class Spec:
        @staticmethod
        def native_to_hf(name, value):
            return [(f"hf.{name}", value), (f"hf.{name}.copy", value.clone())]

    with weight_sync_probe_session("mlite"):
        mapped = hf_weights._native_to_hf(Spec(), "weight", tensor)

    assert [name for name, _ in mapped] == ["hf.weight", "hf.weight.copy"]
    report = _report_from_stdout(capsys.readouterr().out)
    assert report["stages"]["mapping"] == {
        "calls": 1,
        "bytes": 2 * tensor.nbytes,
        "wall_s": report["stages"]["mapping"]["wall_s"],
    }
