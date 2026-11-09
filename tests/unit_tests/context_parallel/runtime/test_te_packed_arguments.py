# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MCore runtime process groups never leak into Transformer Engine packed arguments."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_dynamic_data_context_parallel_groups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture
def parallel_attention():
    Utils.initialize_model_parallel(dynamic_context_parallel=True)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
    )
    yield SelfAttention(
        config,
        get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    )
    Utils.destroy_model_parallel()


def _packed_params():
    bounds = torch.tensor([0, 6, 19, 22, 32], dtype=torch.int32, device='cuda')
    return PackedSeqParams(
        cu_seqlens_q=bounds,
        cu_seqlens_kv=bounds,
        max_seqlen_q=13,
        max_seqlen_kv=13,
        qkv_format='thd',
    )


@pytest.mark.parametrize("runtime_cp_size", [None, 1, 2])
def test_te_does_not_forward_runtime_groups(parallel_attention, monkeypatch, runtime_cp_size):
    import transformer_engine.pytorch as te_pytorch

    core_attention = parallel_attention.core_attention.cuda()
    packed_seq_params = _packed_params()
    runtime_group = (
        get_dynamic_data_context_parallel_groups(group_size=runtime_cp_size)
        if runtime_cp_size is not None
        else None
    )
    tp_cp_group = runtime_group if runtime_group is not None else object()
    packed_seq_params.cp_group = runtime_group
    packed_seq_params.tp_cp_group = tp_cp_group
    packed_seq_params.local_cp_size = runtime_cp_size
    captured = []

    def backend_forward(self, query, key, value, attention_mask, **kwargs):
        captured.append(kwargs)
        assert not {"cp_group", "local_cp_size", "tp_cp_group"}.intersection(kwargs)
        return query.flatten(start_dim=1)

    monkeypatch.setattr(te_pytorch.DotProductAttention, "forward", backend_forward)
    query = torch.zeros(32, 4, 16, dtype=torch.bfloat16, device="cuda")
    core_attention(
        query,
        torch.zeros_like(query),
        torch.zeros_like(query),
        None,
        AttnMaskType.padding_causal,
        packed_seq_params=packed_seq_params,
    )

    assert len(captured) == 1
    assert captured[0]["qkv_format"] == "thd"
    assert captured[0]["cu_seqlens_q"] is packed_seq_params.cu_seqlens_q
    assert captured[0]["cu_seqlens_kv"] is packed_seq_params.cu_seqlens_kv
    assert packed_seq_params.cp_group is runtime_group
    assert packed_seq_params.tp_cp_group is tp_cp_group
    assert packed_seq_params.local_cp_size == runtime_cp_size
