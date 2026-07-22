# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import is_dataclass

import pytest

from megatron.core.transformer.nvshmem_cp_attention import (
    NvshmemCpWorkspace,
    configure_nvshmem_cp_backend,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _valid_config(**overrides):
    kwargs = {
        "num_layers": 4,
        "hidden_size": 2560,
        "num_attention_heads": 20,
        "num_query_groups": 20,
        "kv_channels": 128,
        "context_parallel_size": 4,
        "tensor_model_parallel_size": 1,
        "cp_comm_type": "p2p",
        "context_parallel_attention_backend": "nvshmem",
        "use_cpu_initialization": True,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_nvshmem_cp_backend_accepts_validated_contract():
    config = _valid_config()

    assert config.context_parallel_attention_backend == "nvshmem"


def test_nvshmem_cp_workspace_has_generated_constructor():
    assert is_dataclass(NvshmemCpWorkspace)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context_parallel_size": 2}, "context_parallel_size=4"),
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size=1"),
        ({"cp_comm_type": "all_gather"}, "p2p"),
        ({"kv_channels": 64}, "head_dim=128"),
        ({"num_query_groups": 4}, "supports MHA only"),
    ],
)
def test_nvshmem_cp_backend_rejects_unsupported_contract(overrides, message):
    with pytest.raises(AssertionError, match=message):
        _valid_config(**overrides)


def test_nvshmem_cp_backend_profile_does_not_override_runtime_policy(monkeypatch):
    monkeypatch.delenv("MEGATRON_NVSHMEM_CP_SELF_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("NVSHMEM_DISABLE_CUDA_VMM", raising=False)

    profile = configure_nvshmem_cp_backend()

    assert profile["MEGATRON_NVSHMEM_CP_SELF_ATTENTION_BACKEND"] == "symmetric_qkv_v0"
    assert "NVSHMEM_DISABLE_CUDA_VMM" not in profile
