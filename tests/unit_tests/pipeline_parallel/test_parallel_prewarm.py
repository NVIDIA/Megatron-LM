# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_layer import TransformerLayer


def _make_thd_transformer_layer(*, cuda_graph_impl):
    layer = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        bf16=True,
        context_parallel_size=1,
        create_attention_mask_in_dataloader=False,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=[],
        fp16=False,
        hidden_size=4,
        max_seqlen_per_dp_cp_rank=4,
        sequence_packing_scheduler=object(),
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        thd_max_packed_sequences=2,
    )
    layer.self_attention = SimpleNamespace(attn_mask_type=AttnMaskType.causal)
    layer.is_moe_layer = False
    return layer


def test_pipeline_prewarm_aliases_thd_qk_metadata_only_for_prewarm(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))

    prewarm_inputs = _make_thd_transformer_layer(cuda_graph_impl='none').get_layer_static_inputs(
        4, 1, for_pipeline_prewarm=True
    )
    assert prewarm_inputs['cu_seqlens_q'] is prewarm_inputs['cu_seqlens_kv']
    assert prewarm_inputs['cu_seqlens_q_padded'] is prewarm_inputs['cu_seqlens_kv_padded']
    assert prewarm_inputs['cu_seqlens_q'] is not prewarm_inputs['cu_seqlens_q_padded']

    graph_inputs = _make_thd_transformer_layer(cuda_graph_impl='local').get_layer_static_inputs(
        4, 1, for_pipeline_prewarm=False
    )
    qk_metadata = [
        graph_inputs['cu_seqlens_q'],
        graph_inputs['cu_seqlens_kv'],
        graph_inputs['cu_seqlens_q_padded'],
        graph_inputs['cu_seqlens_kv_padded'],
    ]
    assert len({tensor.data_ptr() for tensor in qk_metadata}) == 4
