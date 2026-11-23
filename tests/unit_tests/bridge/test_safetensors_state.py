# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Repeated reads must be independent without caching a whole first pass."""

import pytest
import torch
from safetensors.torch import save_file

from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


@pytest.mark.parametrize("reuse_handles", ["0", "1"])
def test_cache_only_reused_weights_and_preserve_original_values(
    tmp_path, monkeypatch, reuse_handles
):
    monkeypatch.setenv("MBRIDGE_STATE_REUSE_HANDLES", reuse_handles)
    monkeypatch.setenv("MBRIDGE_STATE_CACHE_SMALL_TENSORS", "1")
    expected = torch.arange(8, dtype=torch.float32)
    save_file({"weight": expected, "scale": expected + 1}, tmp_path / "model.safetensors")
    source = SafeTensorsStateSource(tmp_path)
    first = source.load_tensors(["weight", "scale"])
    assert not source._small_tensor_cache
    first["weight"].zero_()
    second = source.load_tensors(["weight"])
    torch.testing.assert_close(second["weight"], expected)
    assert set(source._small_tensor_cache) == {"weight"}
    second["weight"].zero_()
    torch.testing.assert_close(source.load_tensors(["weight"])["weight"], expected)
