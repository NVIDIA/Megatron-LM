# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import subprocess
from types import SimpleNamespace


def test_extract_series_accepts_precision_metadata(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="/workspace/mlm\n")
    )
    compare_golden_values = importlib.import_module(
        "tests.test_utils.python_scripts.compare_golden_values_kl"
    )
    metric = {"value_precision": "full", "values": {"1": 1.23456789}}

    assert compare_golden_values._extract_series(metric) == {1: 1.23456789}


def test_discovers_golden_values_in_package_and_external_scenarios(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=str(tmp_path))
    )
    compare_golden_values = importlib.import_module(
        "tests.test_utils.python_scripts.compare_golden_values_kl"
    )
    monkeypatch.setattr(compare_golden_values, "REPO_ROOT", tmp_path)
    tracked = "tests/functional_tests/core/models/gpt/scenario/golden_values_dev_dgx_h100.json"
    untracked = "tests/functional_tests/nemo_scenario/golden_values_dev_dgx_h100.json"
    fixture = "tests/functional_tests/core/models/gpt/scenario/requests.json"
    for path in (tracked, untracked, fixture):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")

    def git_output(command, **kwargs):
        assert command[-1] == "tests/functional_tests"
        output = f"{tracked}\n{fixture}" if command[1] == "diff" else untracked
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", git_output)

    assert compare_golden_values.list_modified_golden_files() == [
        tmp_path / tracked,
        tmp_path / untracked,
    ]
