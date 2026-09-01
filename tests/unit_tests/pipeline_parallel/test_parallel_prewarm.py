# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.pipeline_parallel.parallel_prewarm import _build_layer_inputs
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_layer import TransformerLayer


class _SyntheticLayer:
    def __init__(self, *, packed, marker):
        self.packed = packed
        self.marker = marker

    def get_layer_static_inputs(self, _seq_length, _micro_batch_size, **_kwargs):
        inputs = {
            'hidden_states': torch.full(
                (2, 1, 4), self.marker, dtype=torch.float32, requires_grad=True
            ),
            'mtp_dsa_context': object(),
        }
        if self.packed:
            cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
            inputs.update(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens.clone(),
                cu_seqlens_kv_padded=cu_seqlens.clone(),
            )
        else:
            inputs['attention_mask'] = torch.full((1, 1, 2, 2), self.marker, dtype=torch.bool)
        return inputs


def _build_inputs(layer, model_chunk, request_carrier_cache):
    config = SimpleNamespace(
        context_parallel_size=1,
        cp_partition_mode='contiguous',
        max_seqlen_per_dp_cp_rank=2,
        multi_latent_attention=True,
    )
    return _build_layer_inputs(
        layer,
        model_chunk,
        config,
        seq_length=2,
        micro_batch_size=1,
        rotary_pos_emb_cache={},
        request_carrier_cache=request_carrier_cache,
        is_mtp=False,
    )


def test_pipeline_prewarm_reuses_thd_carrier_within_model_chunk():
    chunk = object()
    other_chunk = object()
    cache = {}

    first_args, first_kwargs = _build_inputs(_SyntheticLayer(packed=True, marker=1), chunk, cache)
    second_args, second_kwargs = _build_inputs(_SyntheticLayer(packed=True, marker=2), chunk, cache)
    other_args, other_kwargs = _build_inputs(
        _SyntheticLayer(packed=True, marker=3), other_chunk, cache
    )

    assert first_kwargs['packed_seq_params'] is second_kwargs['packed_seq_params']
    assert first_kwargs['packed_seq_params'] is not other_kwargs['packed_seq_params']
    assert first_args[0] is not second_args[0]
    assert second_args[0] is not other_args[0]
    assert first_kwargs['mtp_dsa_context'] is not second_kwargs['mtp_dsa_context']


def test_pipeline_prewarm_reuses_sbhd_carrier_within_model_chunk():
    chunk = object()
    other_chunk = object()
    cache = {}

    first_args, first_kwargs = _build_inputs(_SyntheticLayer(packed=False, marker=1), chunk, cache)
    second_args, second_kwargs = _build_inputs(
        _SyntheticLayer(packed=False, marker=0), chunk, cache
    )
    _, other_kwargs = _build_inputs(_SyntheticLayer(packed=False, marker=1), other_chunk, cache)

    assert first_kwargs['attention_mask'] is second_kwargs['attention_mask']
    assert first_kwargs['attention_mask'] is not other_kwargs['attention_mask']
    assert first_args[0] is not second_args[0]
    assert first_kwargs['mtp_dsa_context'] is not second_kwargs['mtp_dsa_context']


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
