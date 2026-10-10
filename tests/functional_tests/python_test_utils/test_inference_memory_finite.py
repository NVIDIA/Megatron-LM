# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import json

import pytest

from tests.functional_tests.python_test_utils.test_inference_regular_pipeline import (
    test_inference_pipeline as validate_pipeline,
)


@pytest.mark.parametrize(
    'golden,current',
    [(1000, float('nan')), (1000, [1000, float('nan')]), (float('inf'), float('inf'))],
)
def test_nonfinite_memory_is_rejected(tmp_path, monkeypatch, golden, current):
    monkeypatch.delenv('ENABLE_LIGHTWEIGHT_MODE', raising=False)
    paths = [tmp_path / name for name in ['golden.json', 'actual.json', 'config.yaml']]
    paths[0].write_text(json.dumps({'mem-max-allocated-bytes': golden}))
    paths[1].write_text(json.dumps({'mem-max-allocated-bytes': current}))
    paths[2].write_text('METRICS: [mem-max-allocated-bytes]\n')
    with pytest.raises(AssertionError, match='non-finite'):
        validate_pipeline(*(str(path) for path in paths))


def test_finite_samples_ignore_warmup():
    from tests.functional_tests.python_test_utils.test_inference_regular_pipeline import (
        _median_as_float,
    )

    assert _median_as_float([float('nan'), 1000, 1020]) == 1010
