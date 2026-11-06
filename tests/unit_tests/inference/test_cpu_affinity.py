# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.cpu_affinity import _check_threads


def test_check_threads(caplog):
    masks = {1: {0}, 3: {0, 5}}  # tid 2 exited after enumeration and is skipped.

    def get_affinity(tid):
        if tid not in masks:
            raise ProcessLookupError(tid)
        return masks[tid]

    _check_threads([1, 2, 3], 1, {0}, get_affinity, warn_for_helper_threads=True)
    assert "tid=3 allowed=[0,5]" in caplog.text
    with pytest.raises(RuntimeError, match="tid=3"):
        _check_threads([1, 2, 3], 1, {0}, get_affinity, warn_for_helper_threads=False)
    with pytest.raises(RuntimeError, match="caller"):
        _check_threads([3], 3, {0}, get_affinity, warn_for_helper_threads=True)
