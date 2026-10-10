# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import json

import pytest

from tests.functional_tests.python_test_utils.test_inference_regular_pipeline import (
    test_inference_pipeline as validate_pipeline,
)


def run_validation(tmp_path, current):
    golden = tmp_path / 'golden.json'
    actual = tmp_path / 'actual.json'
    config = tmp_path / 'config.yaml'
    golden.write_text(json.dumps({'a': {'generated_tokens': [1]}, 'b': {'generated_tokens': [2]}}))
    actual.write_text(json.dumps(current))
    config.write_text('METRICS: [generated_tokens]\n')
    validate_pipeline(str(golden), str(actual), str(config))


def test_present_request_subset_is_validated(tmp_path, monkeypatch):
    monkeypatch.delenv('ENABLE_LIGHTWEIGHT_MODE', raising=False)
    run_validation(tmp_path, {'a': {'generated_tokens': [1]}})


def test_subset_token_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv('ENABLE_LIGHTWEIGHT_MODE', raising=False)
    with pytest.raises(AssertionError, match='Token mismatch'):
        run_validation(tmp_path, {'a': {'generated_tokens': [3]}})


def test_empty_request_subset_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv('ENABLE_LIGHTWEIGHT_MODE', raising=False)
    with pytest.raises(AssertionError, match='No current requests'):
        run_validation(tmp_path, {})
