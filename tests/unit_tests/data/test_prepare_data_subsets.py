# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Synthetic, CPU-only tests for token-budget indexed-sequence selection."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from tools.prepare_data_subsets import prepare_subsets


def _source(root: Path, name: str, lengths: list[int], dtype=np.int32, multimodal=False) -> str:
    prefix = str(root / name)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=dtype, multimodal=multimodal)
    # Several indexed sequences deliberately share one document.
    builder.add_document(np.arange(sum(lengths)), lengths)
    builder.finalize(prefix + ".idx")
    return prefix


def _blend(root: Path, train: list, **kwargs) -> Path:
    path = root / "blend.json"
    path.write_text(json.dumps({"train": train, "valid": None, "test": None, **kwargs}))
    return path


@pytest.mark.parametrize("dtype", [np.uint16, np.int32])
@pytest.mark.parametrize("weights", [["0.4000", "0.6000"], [0.4, 0.6]])
def test_native_lengths_yield_nested_minimal_prefixes_and_preserve_blend(tmp_path, dtype, weights):
    lengths = [[0, 9, 3, 20, 1, 11], [3, 0, 25, 8, 17, 2]]
    prefixes = [_source(tmp_path, f"source-{i}", values, dtype) for i, values in enumerate(lengths)]
    train = [weights[0], prefixes[0], weights[1], prefixes[1]]
    blend = _blend(tmp_path, train)
    before = blend.read_bytes()
    output = tmp_path / "subsets"
    # Calibration must never read token payloads through the native dataset interface.
    with patch.object(IndexedDataset, "get", side_effect=AssertionError("token read")):
        with patch.object(IndexedDataset, "__getitem__", side_effect=AssertionError("token read")):
            summary = prepare_subsets(blend, output, [70, 10, 35], seed=1234)
    assert blend.read_bytes() == before
    assert summary["selection_unit"] == "indexed_sequence"
    assert summary["numpy_version"] == np.__version__
    assert json.loads((output / "summary.json").read_text()) == summary
    for i, source in enumerate(summary["sources"]):
        assert source["token_dtype"] == np.dtype(dtype).name
        assert source["available_sequences"] == len(lengths[i])
        order = np.random.RandomState(1234).permutation(np.arange(len(lengths[i]), dtype=np.int32))
        selected_before = set()
        for selected in source["subsets"]:
            retained = math.ceil(float(selected["probability"]) * len(lengths[i]))
            ids = order[:retained]
            assert retained == selected["selected_sequences"]
            assert selected_before <= set(ids)
            selected_before = set(ids)
            total = sum(lengths[i][j] for j in ids)
            assert total == selected["selected_tokens"] >= selected["token_quota"]
            assert sum(lengths[i][j] for j in ids[:-1]) < selected["token_quota"]
    for row in summary["subsets"]:
        result = json.loads((output / row["file"]).read_text())
        assert result["train"][::3] == train[::2]
        assert result["train"][1::3] == train[1::2]
        parsed = get_blend_from_list(result["train"], allow_per_source_downsampling=True)
        assert parsed[0] == prefixes
        assert parsed[1] == [float(weight) for weight in weights]
        assert result["valid"] is result["test"] is None
        assert result["downsampling"]["seed"] == 1234
        assert result["downsampling"]["selected_tokens"] == row["selected_tokens"]
        assert row["selected_tokens"] >= row["requested_tokens"]
    assert any(row["selected_tokens"] > row["requested_tokens"] for row in summary["subsets"])
    again = prepare_subsets(blend, tmp_path / "repeat", [10, 35, 70], seed=1234)
    assert again == summary


@pytest.mark.parametrize("budgets", [[], [0], [-1], [1, 1], [True], [1.5], [100]])
def test_invalid_or_unattainable_budgets_leave_no_output(tmp_path, budgets):
    prefix = _source(tmp_path, "source", [2, 7])
    blend = _blend(tmp_path, ["1.0", prefix])
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare_subsets(blend, output, budgets)
    assert not output.exists()


@pytest.mark.parametrize("weight", ["nan", "inf", "-0.1", "0", True, None, "1/2"])
def test_invalid_weights_leave_no_output(tmp_path, weight):
    prefix = _source(tmp_path, "source", [4])
    blend = _blend(tmp_path, [weight, prefix])
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare_subsets(blend, output, [1])
    assert not output.exists()


@pytest.mark.parametrize("seed", [-1, 2**32, True, 0.5])
def test_invalid_seed_leaves_no_output(tmp_path, seed):
    prefix = _source(tmp_path, "source", [4])
    blend = _blend(tmp_path, [1, prefix])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Seed"):
        prepare_subsets(blend, output, [1], seed)
    assert not output.exists()


@pytest.mark.parametrize("lengths", [[], [0, 0]])
def test_empty_sources_leave_no_output(tmp_path, lengths):
    prefix = _source(tmp_path, "empty", lengths)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare_subsets(_blend(tmp_path, [1, prefix]), output, [1])
    assert not output.exists()


def test_multimodal_source_is_rejected(tmp_path):
    prefix = _source(tmp_path, "multimodal", [3, 7], multimodal=True)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Multimodal"):
        prepare_subsets(_blend(tmp_path, [1, prefix]), output, [5])
    assert not output.exists()


def test_duplicate_sources_and_validation_input_are_rejected(tmp_path):
    prefix = _source(tmp_path, "source", [4])
    output = tmp_path / "output"
    for train, valid in [([1, prefix, 1, prefix], None), ([1, prefix], [prefix])]:
        with pytest.raises(ValueError):
            prepare_subsets(_blend(tmp_path, train, valid=valid), output, [1])
        assert not output.exists()


def test_existing_output_directory_is_never_modified(tmp_path):
    prefix = _source(tmp_path, "source", [4, 9])
    blend = _blend(tmp_path, [1, prefix])
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_subsets(blend, output, [1])
    assert list(output.iterdir()) == []
    output.rmdir()
    prepare_subsets(blend, output, [1])
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_subsets(blend, output, [9])
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_relative_prefix_strings_are_preserved(tmp_path, monkeypatch):
    _source(tmp_path, "source", [1, 8])
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    prepare_subsets(_blend(tmp_path, [1, "./source"]), output, [2])
    assert json.loads((output / "subset_2.json").read_text())["train"][1] == "./source"


def test_module_cli_from_repository_root(tmp_path):
    prefix = _source(tmp_path, "source", [3, 9])
    blend = _blend(tmp_path, [1, prefix])
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.prepare_data_subsets",
            "--input-blend",
            str(blend),
            "--output-dir",
            str(output),
            "--token-budgets",
            "2",
            "8",
            "--seed",
            "1234",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "PYTHONPATH": ""},
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert [row["requested_tokens"] for row in summary["subsets"]] == [2, 8]
    assert (output / "subset_2.json").exists()
    assert (output / "subset_8.json").exists()


@pytest.mark.parametrize("case", ["empty", "odd", "triples", "nonobject"])
def test_invalid_blend_shape_leaves_no_output(tmp_path, case):
    prefix = _source(tmp_path, "source", [4])
    inputs = {"empty": [], "odd": [1], "triples": [1, prefix, 0.5]}
    blend = _blend(tmp_path, inputs.get(case, []))
    if case == "nonobject":
        blend.write_text("[]")
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare_subsets(blend, output, [1])
    assert not output.exists()


@pytest.mark.parametrize("name", ["123", " source", "source "])
def test_ambiguous_source_spellings_are_rejected_before_output(tmp_path, monkeypatch, name):
    _source(tmp_path, name, [4])
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Numeric|whitespace"):
        prepare_subsets(_blend(tmp_path, [1, name]), output, [1])
    assert not output.exists()


def test_numeric_filename_accepts_explicit_relative_prefix(tmp_path, monkeypatch):
    _source(tmp_path, "123", [4])
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    prepare_subsets(_blend(tmp_path, [1, "./123"]), output, [1])
    result = json.loads((output / "subset_1.json").read_text())
    parsed = get_blend_from_list(result["train"], allow_per_source_downsampling=True)
    assert parsed[0] == ["./123"]
