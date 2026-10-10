# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.mark.parametrize('output_prefix', ['merged', './merged'])
def test_accepts_output_prefix_in_current_directory(tmp_path, monkeypatch, output_prefix):
    # Dataset storage is outside this argument-parsing test; keep it CPU-only.
    dataset_module = ModuleType('megatron.core.datasets.indexed_dataset')
    for symbol in ['IndexedDataset', 'IndexedDatasetBuilder', 'get_bin_path', 'get_idx_path']:
        setattr(dataset_module, symbol, None)
    monkeypatch.setitem(sys.modules, dataset_module.__name__, dataset_module)
    source = Path(__file__).resolve().parents[3] / 'tools' / 'merge_datasets.py'
    module_spec = importlib.util.spec_from_file_location('merge_datasets', source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, 'argv', ['merge_datasets', '--input', str(tmp_path), '--output-prefix', output_prefix]
    )

    assert module.get_args().output_prefix == output_prefix
