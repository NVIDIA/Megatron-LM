# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.sampling import flashinfer_sampling as flashinfer_sampling_module


@pytest.mark.parametrize("return_static_graph_outputs", [False, True])
def test_flashinfer_sampling_configures_graph_output_ownership(
    monkeypatch, return_static_graph_outputs
):
    managers = []

    class _GraphManager:
        def __init__(self, _config, _sampling, **kwargs):
            self.kwargs = kwargs
            managers.append(self)

    monkeypatch.setattr(flashinfer_sampling_module, "CudaGraphManager", _GraphManager)
    sampling = flashinfer_sampling_module.FlashInferSampling(
        vocab_size=128,
        rng=torch.Generator(),
        config=SimpleNamespace(cuda_graph_impl="local"),
        enable_cuda_graph=True,
        return_static_graph_outputs=return_static_graph_outputs,
    )

    assert sampling._sample_graph_manager is managers[0]
    assert sampling._speculative_graph_manager is managers[1]
    assert [manager.kwargs["function_name"] for manager in managers] == [
        "sample_kernel",
        "sample_speculative",
    ]
    assert all(
        manager.kwargs["clone_outputs"] is not return_static_graph_outputs for manager in managers
    )


def test_flashinfer_sampling_without_graphs_has_no_managers():
    sampling = flashinfer_sampling_module.FlashInferSampling(
        vocab_size=128, rng=torch.Generator(), enable_cuda_graph=False
    )

    assert sampling._sample_graph_manager is None
    assert sampling._speculative_graph_manager is None
