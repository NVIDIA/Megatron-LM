# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay native mHC graphs under side-stream contention, including every mapping VJP."""

import pytest
import torch

from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.mhc_mapping_graph import _MHCMappingsGraph
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_module_replays_bit_exact, seeded


class _AllMappings(torch.nn.Module):
    """Keep all three outputs differentiable through the harness's single-output VJP."""

    def __init__(self, module: HyperConnectionModule) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        """Concatenate mappings so the harness exercises the captured backward graph."""
        return torch.cat([value.reshape(-1) for value in self.module.compute_mappings(x)])


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("fp32_mappings", [False, True])
def test_mhc_mapping_graph_replays_bit_exact(batch, fp32_mappings):
    """Every output, input gradient, and parameter gradient is byte-identical under contention."""
    seeded()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_sinkhorn_iterations=20,
        mhc_keep_mappings_in_fp32=fp32_mappings,
        mhc_norm_eps_inside_sqrt=fp32_mappings,
    )
    module = HyperConnectionModule(config, 1).cuda()
    x = torch.randn(512, batch, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    module.enable_mapping_cuda_graphs(x)
    assert isinstance(module._mapping_cuda_graph, _MHCMappingsGraph)
    assert_module_replays_bit_exact(
        _AllMappings(module),
        (x,),
        replays=3,
        contention=True,
        what="native mHC mapping CUDA graphs",
    )
    assert module._mapping_cuda_graph.replays["forward"] == 3
    assert module._mapping_cuda_graph.replays["backward"] == 3
