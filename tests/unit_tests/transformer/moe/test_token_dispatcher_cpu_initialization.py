# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


def test_alltoall_dispatcher_cpu_only_initialization_does_not_access_cuda(monkeypatch):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_permute_fusion=True,
        use_cpu_initialization=True,
    )
    process_groups = SimpleNamespace(ep=None, expt_tp=None, tp_ep=None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda: pytest.fail("CPU-only dispatcher construction must not create a CUDA stream"),
    )
    monkeypatch.setattr(MoEAlltoAllTokenDispatcher, "cuda_dtoh_stream", None)

    dispatcher = MoEAlltoAllTokenDispatcher(
        num_local_experts=4,
        local_expert_indices=[0, 1, 2, 3],
        config=config,
        pg_collection=process_groups,
    )

    assert dispatcher.permute_idx_device == "cpu"
    assert dispatcher.sort_input_by_local_experts.device.type == "cpu"
    assert dispatcher.restore_output_by_local_experts.device.type == "cpu"
    assert dispatcher.cuda_dtoh_stream is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alltoall_dispatcher_cpu_weight_initialization_preserves_cuda_runtime_resources(
    monkeypatch,
):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_permute_fusion=True,
        use_cpu_initialization=True,
    )
    process_groups = SimpleNamespace(ep=None, expt_tp=None, tp_ep=None)
    monkeypatch.setattr(MoEAlltoAllTokenDispatcher, "cuda_dtoh_stream", None)

    dispatcher = MoEAlltoAllTokenDispatcher(
        num_local_experts=4,
        local_expert_indices=[0, 1, 2, 3],
        config=config,
        pg_collection=process_groups,
    )

    assert dispatcher.permute_idx_device.type == "cuda"
    assert dispatcher.sort_input_by_local_experts.device.type == "cuda"
    assert dispatcher.restore_output_by_local_experts.device.type == "cuda"
    assert dispatcher.cuda_dtoh_stream is not None
