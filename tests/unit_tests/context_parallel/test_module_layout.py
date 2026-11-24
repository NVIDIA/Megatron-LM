# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Producer-to-compute-boundary contracts; real layer methods with mock collectives."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.context_parallel.module as cp_module
import megatron.core.models.hybrid.hybrid_block as hybrid_block_module
from megatron.core.context_parallel import get_batches_on_this_cp_rank
from megatron.core.context_parallel.module import (
    finalize_packed_seq_params,
    prepare_module_cp_input,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.transformer.module import TwoStageAttentionLayer
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


def _group(size=2):
    return SimpleNamespace(size=lambda: size, rank=lambda: 0)


def _batch(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())
    return get_batches_on_this_cp_rank(
        {
            "tokens": torch.arange(8).reshape(1, 8),
            "cu_seqlens": torch.tensor([[0, 0, 3, 8]], dtype=torch.int32),
            "max_seqlen": torch.tensor([5], dtype=torch.int32),
        },
        boundary_layout="contiguous",
        dynamic_cp=False,
        cp_group=_group(),
        additional_layouts=("zigzag",),
    )


def test_producer_prepares_acyclic_views_once(monkeypatch):
    batch = _batch(monkeypatch)
    source = batch.get_packed_seq_params()
    assert source.cp_layout_views is None  # Legacy graph metadata is unchanged.
    batch.prepare_module_layouts()
    params = batch.get_packed_seq_params()
    views = params.cp_layout_views
    assert views.plan is batch.thd_plan
    assert views.contiguous.cp_layout_views is None
    assert views.zigzag.cp_layout_views is None
    assert source.cp_layout_views is None
    assert params.seq_idx is source.seq_idx
    batch.prepare_module_layouts()
    assert batch.get_packed_seq_params() is params
    assert views.zigzag.cu_seqlens_q.tolist() == [0, 0, 3, 8]
    assert views.zigzag.cu_seqlens_q_padded.tolist() == [0, 0, 4, 12]
    # The producer's boundary max (5) is too short for the padded attention view (8).
    assert RotaryEmbedding.get_rotary_seq_len(None, None, None, None, None, params) == 8


def test_bridge_adapter_uses_the_same_route_builder(monkeypatch):
    batch = _batch(monkeypatch)
    params = batch.get_packed_seq_params()
    adapted = finalize_packed_seq_params(params, _group())
    assert params.cp_layout_views is None
    assert adapted.cp_layout_views.plan.contiguous_local_token_count == 4
    assert adapted.cp_layout_views.plan.zigzag_local_token_count == 6
    assert adapted.cu_seqlens_q.tolist() == [0, 0, 3, 8]
    assert finalize_packed_seq_params(adapted, _group()) is adapted


class _Attention(torch.nn.Module, TwoStageAttentionLayer):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, **kwargs):
        self.seen.append((x.detach().clone(), kwargs["packed_seq_params"]))
        return x * 2, None

    def forward_pre_attn_and_core_attn(self, x, **kwargs):
        self.seen.append((x.detach().clone(), kwargs["packed_seq_params"]))
        return x * 2

    def forward_post_core_attn(self, x):
        return x, None


def _layer():
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        global_cp_layout="contiguous",
        attention_cp_layout="zigzag",
        sequence_parallel=False,
        cuda_graph_impl="none",
        inference_fuse_tp_communication=False,
    )
    layer.pg_collection = SimpleNamespace(cp=_group(), tp=_group(1), tp_cp=None)
    layer.self_attention = _Attention()
    layer._run_input_layernorm = lambda x, **kwargs: (x, x, ())
    layer.supports_two_stage_attention = lambda: True
    layer.attention_bda_and_cross_attention = lambda out, residual, **kwargs: (
        out[0] + residual,
        None,
    )
    return layer


@pytest.mark.parametrize("split", [False, True])
def test_real_layer_restores_before_bda_and_interleaved_calls(monkeypatch, split):
    calls = []

    def convert(x, **kwargs):
        calls.append((kwargs["source_partition_mode"], kwargs["target_partition_mode"]))
        return x.flip(0)

    monkeypatch.setattr(cp_module, "convert_cp_partition_mode", convert)
    layer = _layer()
    x = torch.arange(8.0).reshape(4, 1, 2).requires_grad_()
    if split:
        state1 = layer.forward_pre_attn_and_core_attn(x)
        state2 = layer.forward_pre_attn_and_core_attn(x + 10)
        y2, _ = layer.forward_post_core_attn(*state2)
        y, _ = layer.forward_post_core_attn(*state1)
        torch.testing.assert_close(y2, 3 * (x + 10))
    else:
        y, _ = layer._forward_attention(x)
    torch.testing.assert_close(layer.self_attention.seen[0][0], x.flip(0))
    torch.testing.assert_close(y, 3 * x)  # BDA saw the original residual ordering.
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 3))
    assert calls.count(("contiguous", "zigzag")) == (2 if split else 1)
    assert calls.count(("zigzag", "contiguous")) == (2 if split else 1)


def test_thd_boundary_selects_prebuilt_target_metadata(monkeypatch):
    batch = _batch(monkeypatch).prepare_module_layouts()
    params = batch.get_packed_seq_params()
    monkeypatch.setattr(cp_module, "convert_cp_partition_mode", lambda x, **kw: x)
    layer = _layer()
    _, target, reverse = prepare_module_cp_input(
        layer, torch.zeros(4, 1, 2), params, target_layout="zigzag"
    )
    assert target is params.cp_layout_views.zigzag
    assert reverse.thd_plan is batch.thd_plan
    assert params.cp_partition_mode == "contiguous"
    with pytest.raises(ValueError, match="producer-prebuilt"):
        prepare_module_cp_input(
            layer,
            torch.zeros(4, 1, 2),
            PackedSeqParams(qkv_format="thd", cp_partition_mode="contiguous"),
            target_layout="zigzag",
        )


def test_runtime_singleton_needs_no_route_or_communication():
    layer = _layer()
    params = PackedSeqParams(
        qkv_format="thd", cp_partition_mode="contiguous", cp_group=_group(1), local_cp_size=1
    )
    x = torch.ones(4, 1, 2)
    result, metadata, reverse = prepare_module_cp_input(layer, x, params, target_layout="zigzag")
    assert result is x and metadata is params and reverse is None


@pytest.mark.parametrize("rank", [0, 1])
def test_metadata_only_chunkwise_stack_uses_consumer_physical_lengths(monkeypatch, rank):
    """Exercise the stack's preparation and GDP's consumer check without GPU kernels."""
    batch = _batch(monkeypatch).prepare_module_layouts()
    params = finalize_packed_seq_params(batch.get_packed_seq_params("zigzag"), _group())
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    stack = HybridStack.__new__(HybridStack)
    torch.nn.Module.__init__(stack)
    stack._has_custom_layers = False
    stack._has_linear_layer_with_chunkwise_cp = True
    stack.cp_group = group
    build_metadata = hybrid_block_module.build_packed_sequence_cp_metadata

    class ConsumerChecked(Exception):
        pass

    def check_consumer(*args, **kwargs):
        metadata = build_metadata(*args, **kwargs)
        # The converted contiguous activation has four tokens per CP rank,
        # even though the incoming zigzag view has six physical slots.
        GatedDeltaProductMixer._chunkwise_packed_metadata(
            None, params.cp_layout_views.contiguous, 4, metadata
        )
        assert metadata.local_seq_idx.tolist() == ([[1, 1, 1, 2]] if rank == 0 else [[2, 2, 2, 2]])
        assert metadata.local_cu_seqlens.tolist() == ([0, 3, 4] if rank == 0 else [0, 4])
        # Stop before downstream model execution; both real metadata methods ran.
        raise ConsumerChecked

    monkeypatch.setattr(hybrid_block_module, "build_packed_sequence_cp_metadata", check_consumer)
    with pytest.raises(ConsumerChecked):
        stack.forward(torch.zeros(6, 1, 2), None, packed_seq_params=params)


@pytest.mark.parametrize(
    "mode,linear_layout,tag,cp_size,needs_batch",
    [
        ("contiguous", "zigzag", None, 2, True),
        ("contiguous", "zigzag", "contiguous", 2, True),
        ("zigzag", "contiguous", None, 2, False),
        (None, "zigzag", "contiguous", 2, True),
        (None, "zigzag", None, 2, False),
        ("contiguous", "zigzag", "contiguous", 1, False),
    ],
)
def test_metadata_only_mtp_checks_the_actual_source_layout(
    mode, linear_layout, tag, cp_size, needs_batch
):
    """Missing batch views must not silently skip required MTP redistribution."""
    block = SimpleNamespace(
        config=SimpleNamespace(
            global_cp_layout=mode or linear_layout, attention_cp_layout="zigzag"
        ),
        cp_group=_group(cp_size),
        tp_group=_group(1),
        tp_cp_group=None,
        sequence_parallel=False,
    )
    hidden_states = torch.arange(8.0).reshape(4, 1, 2)
    params = None if tag is None else PackedSeqParams(qkv_format="thd", cp_partition_mode=tag)
    kwargs = dict(
        input_ids=torch.arange(4).reshape(1, 4),
        position_ids=torch.arange(4).reshape(1, 4),
        hidden_states=hidden_states,
        decoder_input=None,
        mhc_multistream=None,
        labels=None,
        loss_mask=None,
        mtp_input_mask=None,
        packed_seq_params=params,
        cp_batch=None,
    )
    if needs_batch:
        with pytest.raises(ValueError, match="cp_batch is required"):
            MultiTokenPredictionBlock.prepare_cp_layout(block, **kwargs)
    else:
        result = MultiTokenPredictionBlock.prepare_cp_layout(block, **kwargs)
        assert result.hidden_states is hidden_states
        assert result.packed_seq_params is params


@pytest.mark.parametrize("graph", ["local", "transformer_engine", "full_iteration"])
def test_new_thd_graph_boundary_is_rejected(monkeypatch, graph):
    params = _batch(monkeypatch).prepare_module_layouts().get_packed_seq_params()
    layer = _layer()
    layer.config.cuda_graph_impl = graph
    with pytest.raises(ValueError, match="CUDA graphs"):
        prepare_module_cp_input(layer, torch.zeros(4, 1, 2), params, target_layout="zigzag")


@pytest.mark.parametrize("script_name", ["pretrain_gpt", "pretrain_hybrid"])
@pytest.mark.parametrize("mode", ["contiguous", "zigzag"])
def test_pretrain_entry_prebuilds_module_views(monkeypatch, script_name, mode):
    import importlib

    script = importlib.import_module(script_name)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        global_cp_layout=mode,
        linear_attention_cp_layout=("zigzag" if script_name == "pretrain_hybrid" else "contiguous"),
        attention_cp_layout="zigzag",
        sequence_packing_scheduler=None,
        sequence_parallel=False,
        context_parallel_size=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        dynamic_context_parallel=False,
    )
    args = SimpleNamespace(
        sft=True,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=False,
        seq_length=8,
        micro_batch_size=1,
    )
    monkeypatch.setattr(script, "get_args", lambda: args)
    monkeypatch.setattr(
        script, "get_run_config", lambda: SimpleNamespace(model=SimpleNamespace(transformer=config))
    )
    monkeypatch.setattr(script, "is_first_or_last_pipeline_stage", lambda stage: True)
    monkeypatch.setattr(script, "mtp_on_this_rank_func", lambda **kwargs: False)
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_group", lambda: _group(1))
    monkeypatch.setattr(script.mpu, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(script.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(script, "get_context_parallel_group", lambda: _group())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())
    batch = {key: None for key in script.BATCH_KEYS}
    batch.update(
        tokens=torch.arange(8).view(1, 8),
        cu_seqlens=torch.tensor([[0, 3, 8]], dtype=torch.int32),
        max_seqlen=torch.tensor([5], dtype=torch.int32),
    )
    monkeypatch.setattr(script, "get_batch_on_this_tp_rank", lambda *args, **kw: batch)
    if script_name == "pretrain_hybrid":
        # A heterogeneous model can require a layout absent from the root config.
        result = script.get_batch(
            None,
            pg_collection=SimpleNamespace(pp=_group(1)),
            required_cp_layouts=("contiguous", "zigzag"),
        )
    else:
        result = script.get_batch(None, pg_collection=SimpleNamespace(pp=_group(1)))
    assert result.boundary_layout == mode
    params = result.get_packed_seq_params()
    assert params.cp_partition_mode == mode
    assert params.cp_layout_views.plan is result.thd_plan
    assert result.get_batch()["tokens"].tolist() == (
        [[0, 1, 2, 3]] if mode == "contiguous" else [[0, 0, 3, 4, 0, 0]]
    )


@pytest.mark.parametrize(
    "options,expected_layout",
    [
        ({}, "zigzag"),
        ({"linear_cp_mode": "chunkwise"}, "contiguous"),
        ({"linear_cp_mode": "chunkwise", "global_cp_layout": "zigzag"}, "zigzag"),
        ({"linear_attention_cp_layout": "contiguous"}, "contiguous"),
        ({"linear_cp_layout": "contiguous"}, "contiguous"),
        ({"linear_cp_mode": "chunkwise", "linear_attention_cp_layout": "zigzag"}, "zigzag"),
    ],
)
def test_config_resolution_drives_real_batch_partition(monkeypatch, options, expected_layout):
    config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=2, **options)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())
    batch = get_batches_on_this_cp_rank(
        {"tokens": torch.arange(8).reshape(1, 8)},
        boundary_layout=cp_module.get_cp_boundary_layout(config),
        dynamic_cp=False,
        cp_group=_group(),
    )
    assert batch.get_batch()["tokens"].tolist() == (
        [[0, 1, 2, 3]] if expected_layout == "contiguous" else [[0, 1, 6, 7]]
    )
    assert config.linear_cp_layout == config.linear_attention_cp_layout
    assert config.cp_partition_mode == config.global_cp_layout


def test_conflicting_aliases_and_legacy_run_policy_fail_at_configuration():
    with pytest.raises(ValueError, match="Conflicting"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=2,
            linear_cp_layout="zigzag",
            linear_attention_cp_layout="contiguous",
        )
    with pytest.raises(ValueError, match="concrete layout"):
        TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=2, cp_partition_mode="auto"
        )
