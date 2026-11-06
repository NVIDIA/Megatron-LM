# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference import cpu_affinity as affinity


def test_nvml_uuid_adds_gpu_prefix():
    # torch reports bare UUIDs; NVML lookups need the "GPU-" prefix.
    assert affinity._nvml_uuid("109c5465-7b59") == "GPU-109c5465-7b59"


def test_find_violations():
    masks = {1: {2}, 3: {3, 4}}  # tid 2 exited before inspection and is skipped.

    def get_affinity(tid):
        if tid not in masks:
            raise ProcessLookupError(tid)
        return masks[tid]

    assert affinity._find_violations([1, 2, 3], 1, {2, 3}, get_affinity) == {3: {3, 4}}


def test_caller_violation_raises_while_helper_violation_warns(caplog):
    with pytest.raises(RuntimeError, match="caller"):
        affinity._report_violations({1: {0, 5}}, 1, {0}, "ctx", warn_for_helper_threads=True)
    affinity._report_violations({7: {0, 5}}, 1, {0}, "ctx", warn_for_helper_threads=True)
    assert "tid=7" in caplog.text
