# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.inference import triton_pdl

PDL_KWARGS = {"USE_PDL": True, "launch_pdl": True}


def test_pdl_kwargs_only_under_inference_mode(monkeypatch):
    """With PDL supported, kernels get the PDL launch arguments only under inference mode."""
    monkeypatch.setattr(triton_pdl, "_enabled", True)
    assert triton_pdl.pdl_kwargs() == {}
    with torch.inference_mode():
        assert triton_pdl.pdl_kwargs() == PDL_KWARGS
        with torch.inference_mode(False):
            assert triton_pdl.pdl_kwargs() == {}
        assert triton_pdl.pdl_kwargs() == PDL_KWARGS
    assert triton_pdl.pdl_kwargs() == {}


def test_pdl_kwargs_empty_when_unsupported(monkeypatch):
    """Without PDL support, inference mode does not turn it on."""
    monkeypatch.setattr(triton_pdl, "_enabled", False)
    with torch.inference_mode():
        assert triton_pdl.pdl_kwargs() == {}
