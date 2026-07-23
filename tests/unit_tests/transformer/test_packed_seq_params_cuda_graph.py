# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.packed_seq_params import (
    CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS,
    PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS,
    PackedSeqParams,
    build_packed_seq_params_from_cuda_graph_kwargs,
    has_packed_seq_params_cuda_graph_kwargs,
    split_packed_seq_params_for_cuda_graph,
)
from megatron.core.transformer.cuda_graphs import (
    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


class _TransformerLayerCudaGraphStub:
    _set_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._set_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._get_te_cuda_graph_packed_seq_params_static_metadata
    )
    _validate_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._validate_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_tensor_kwarg_names = (
        TransformerLayer._get_te_cuda_graph_packed_seq_params_tensor_kwarg_names
    )
    _validate_te_cuda_graph_packed_seq_params_tensor_kwargs = (
        TransformerLayer._validate_te_cuda_graph_packed_seq_params_tensor_kwargs
    )
    _rebuild_te_cuda_graph_packed_seq_params = (
        TransformerLayer._rebuild_te_cuda_graph_packed_seq_params
    )
    _flatten_te_cuda_graph_packed_seq_params = (
        TransformerLayer._flatten_te_cuda_graph_packed_seq_params
    )


def _make_packed_seq_params():
    cu_seqlens = torch.IntTensor([0, 4, 9, 16])
    cu_seqlens_padded = torch.IntTensor([0, 8, 12, 16])
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        local_cp_size=1,
        pad_between_seqs=False,
        cp_partition_mode="contiguous",
    )


def test_split_packed_seq_params_for_cuda_graph_separates_tensors_from_metadata():
    packed_seq_params = _make_packed_seq_params()

    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)

    assert set(static_metadata) == set(PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS)
    assert static_metadata == {
        "qkv_format": "thd",
        "max_seqlen_q": 8,
        "max_seqlen_kv": 8,
        "local_cp_size": 1,
        "cp_group": None,
        "pad_between_seqs": False,
        "cp_partition_mode": "contiguous",
    }
    assert all(not isinstance(value, torch.Tensor) for value in static_metadata.values())

    expected_tensor_fields = {
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
    }
    assert set(tensor_kwargs) == {
        f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}{field}" for field in expected_tensor_fields
    }
    assert set(PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS) >= expected_tensor_fields
    for value in tensor_kwargs.values():
        assert isinstance(value, torch.Tensor)


def test_has_packed_seq_params_cuda_graph_kwargs_detects_flattened_fields():
    tensor_kwargs, _ = split_packed_seq_params_for_cuda_graph(_make_packed_seq_params())

    assert has_packed_seq_params_cuda_graph_kwargs(tensor_kwargs)
    assert not has_packed_seq_params_cuda_graph_kwargs({"hidden_states": torch.ones(2, 1, 4)})
    assert build_packed_seq_params_from_cuda_graph_kwargs({}, None) is None


def test_build_packed_seq_params_from_cuda_graph_kwargs_pops_flattened_fields():
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    kwargs = {"hidden_states": torch.ones(2, 1, 4), **tensor_kwargs}

    rebuilt = build_packed_seq_params_from_cuda_graph_kwargs(kwargs, static_metadata)

    assert set(kwargs) == {"hidden_states"}
    assert rebuilt.qkv_format == "thd"
    assert rebuilt.max_seqlen_q == 8
    assert rebuilt.max_seqlen_kv == 8
    assert rebuilt.local_cp_size == 1
    assert rebuilt.cp_group is None
    assert rebuilt.pad_between_seqs is False
    assert rebuilt.cp_partition_mode == "contiguous"
    assert rebuilt.total_tokens is None
    assert rebuilt.seq_idx is None
    assert torch.equal(rebuilt.cu_seqlens_q, packed_seq_params.cu_seqlens_q)
    assert torch.equal(rebuilt.cu_seqlens_kv, packed_seq_params.cu_seqlens_kv)
    assert torch.equal(rebuilt.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q_padded)
    assert torch.equal(rebuilt.cu_seqlens_kv_padded, packed_seq_params.cu_seqlens_kv_padded)


def test_build_packed_seq_params_from_cuda_graph_kwargs_can_keep_kwargs_intact():
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(
        _make_packed_seq_params()
    )
    kwargs = dict(tensor_kwargs)

    build_packed_seq_params_from_cuda_graph_kwargs(
        kwargs, static_metadata, remove_from_kwargs=False
    )

    assert kwargs == tensor_kwargs


def test_split_packed_seq_params_for_cuda_graph_rejects_static_tensor_metadata():
    packed_seq_params = _make_packed_seq_params()
    packed_seq_params.max_seqlen_q = torch.IntTensor([8])

    with pytest.raises(TypeError, match="max_seqlen_q"):
        split_packed_seq_params_for_cuda_graph(packed_seq_params)


def test_split_packed_seq_params_for_cuda_graph_ignores_mamba_only_fields():
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv=torch.IntTensor([0, 2, 5]),
        max_seqlen_q=3,
        max_seqlen_kv=3,
        total_tokens=5,
    )
    assert packed_seq_params.seq_idx is not None

    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)

    assert f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_idx" not in tensor_kwargs
    assert "total_tokens" not in static_metadata


def test_transformer_layer_rebuilds_flattened_cuda_graph_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    kwargs = {"hidden_states": torch.ones(2, 1, 4), **tensor_kwargs}

    layer._rebuild_te_cuda_graph_packed_seq_params(kwargs)

    assert set(kwargs) == {"hidden_states", "packed_seq_params"}
    rebuilt = kwargs["packed_seq_params"]
    assert rebuilt.qkv_format == "thd"
    assert rebuilt.max_seqlen_q == 8
    assert rebuilt.max_seqlen_kv == 8
    assert rebuilt.cp_partition_mode == "contiguous"
    assert torch.equal(rebuilt.cu_seqlens_q, packed_seq_params.cu_seqlens_q)
    assert torch.equal(rebuilt.cu_seqlens_kv, packed_seq_params.cu_seqlens_kv)


def test_transformer_layer_flattens_replay_time_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    attention_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    kwargs = {"attention_mask": attention_mask, "packed_seq_params": packed_seq_params}

    layer._flatten_te_cuda_graph_packed_seq_params(kwargs)

    assert kwargs["attention_mask"] is attention_mask
    assert "packed_seq_params" not in kwargs
    assert set(tensor_kwargs).issubset(kwargs)
    for key, value in tensor_kwargs.items():
        assert kwargs[key] is value


def test_transformer_layer_rejects_replay_without_captured_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    _, static_metadata = split_packed_seq_params_for_cuda_graph(_make_packed_seq_params())
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata)

    with pytest.raises(AssertionError, match="captured with packed_seq_params"):
        layer._flatten_te_cuda_graph_packed_seq_params({"hidden_states": torch.ones(2, 1, 4)})


def test_transformer_layer_rejects_changed_packed_seq_params_static_metadata():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    _, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata)
    packed_seq_params.max_seqlen_q = 4

    with pytest.raises(AssertionError, match="max_seqlen_q"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": packed_seq_params})


def test_transformer_layer_rejects_changed_pad_between_seqs_metadata():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(
        static_metadata, tensor_kwargs
    )
    packed_seq_params.pad_between_seqs = True

    with pytest.raises(AssertionError, match="pad_between_seqs"):
        layer._flatten_te_cuda_graph_packed_seq_params(
            {"packed_seq_params": packed_seq_params}
        )


def test_transformer_layer_rejects_changed_cp_partition_mode_metadata():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(
        static_metadata, tensor_kwargs
    )
    packed_seq_params.cp_partition_mode = "zigzag"

    with pytest.raises(AssertionError, match="cp_partition_mode"):
        layer._flatten_te_cuda_graph_packed_seq_params(
            {"packed_seq_params": packed_seq_params}
        )


def test_hybrid_wrapper_delegates_prefixed_packed_sequence_contract():
    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer

    inner = _TransformerLayerCudaGraphStub()
    wrapper = object.__new__(HyperConnectionHybridLayer)
    object.__setattr__(wrapper, "inner_layer", inner)
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )

    wrapper._set_te_cuda_graph_packed_seq_params_static_metadata(
        static_metadata, tensor_kwargs
    )
    replay_kwargs = {"packed_seq_params": packed_seq_params}
    wrapper._flatten_te_cuda_graph_packed_seq_params(replay_kwargs)

    assert "packed_seq_params" not in replay_kwargs
    assert set(replay_kwargs) == set(tensor_kwargs)
    wrapper._rebuild_te_cuda_graph_packed_seq_params(replay_kwargs)
    assert replay_kwargs["packed_seq_params"].pad_between_seqs is False
    assert torch.equal(
        replay_kwargs["packed_seq_params"].cu_seqlens_q,
        packed_seq_params.cu_seqlens_q,
    )


def test_thd_graph_discovery_excludes_wrapped_mamba_without_metadata_contract():
    from types import SimpleNamespace

    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.cuda_graphs import _layer_is_graphable
    from megatron.core.transformer.enums import CudaGraphModule

    wrapper = object.__new__(HyperConnectionHybridLayer)
    object.__setattr__(wrapper, "inner_layer", object.__new__(MambaLayer))
    config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.mamba],
        sequence_packing_scheduler="dp_balanced",
    )

    assert not _layer_is_graphable(wrapper, config)


def test_hybrid_wrapper_leaves_local_cudagraph_manager_on_inner_layer():
    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.transformer.module import GraphableMegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    class _LocalGraphInner(GraphableMegatronModule):
        def __init__(self, config):
            super().__init__(config)
            self.layer_number = 1
            self.offload_module_in_cuda_graph = True

        def create_mcore_cudagraph_manager(self, _config):
            self.cudagraph_manager = object()

    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        ffn_hidden_size=16,
        cuda_graph_impl="local",
        enable_hyper_connections=True,
        num_residual_streams=2,
    )
    inner = _LocalGraphInner(config)
    wrapper = HyperConnectionHybridLayer(config, inner)

    assert hasattr(inner, "cudagraph_manager")
    assert not hasattr(wrapper, "cudagraph_manager")
    assert wrapper.offload_module_in_cuda_graph


def test_hybrid_wrapper_forwards_offload_stream_and_event_to_te_graph(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
    )

    transformer_engine = ModuleType("transformer_engine")
    transformer_engine.__path__ = []
    transformer_engine_pytorch = ModuleType("transformer_engine.pytorch")
    transformer_engine.pytorch = transformer_engine_pytorch
    monkeypatch.setitem(sys.modules, "transformer_engine", transformer_engine)
    monkeypatch.setitem(
        sys.modules, "transformer_engine.pytorch", transformer_engine_pytorch
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.is_te_min_version",
        lambda *_args, **_kwargs: True,
    )

    cuda_graph_stream = object()
    cuda_graph_event = object()
    monkeypatch.setattr(
        FineGrainedActivationOffloadingInterface,
        "cuda_graph_stream",
        lambda: cuda_graph_stream,
    )
    monkeypatch.setattr(
        FineGrainedActivationOffloadingInterface,
        "cuda_graph_event",
        lambda: cuda_graph_event,
    )

    config = SimpleNamespace(
        create_attention_mask_in_dataloader=False,
        fine_grained_activation_offloading=True,
    )
    inner = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(inner)
    inner.config = config
    inner.offload_module_in_cuda_graph = True
    inner.current_microbatch = 7
    wrapper = object.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = inner
    wrapper.config = config
    wrapper.current_microbatch = 1
    hidden_states = torch.ones(2, 1, 4)

    graph_args, graph_kwargs = wrapper._get_te_cuda_graph_replay_args(
        hidden_states, attention_mask=None
    )

    assert len(graph_args) == 1
    assert graph_args[0] is hidden_states
    assert graph_kwargs["is_first_microbatch"] is False
    assert "attention_mask" not in graph_kwargs
    assert graph_kwargs["cuda_graph_stream"] is cuda_graph_stream
    assert graph_kwargs["cuda_graph_event"] is cuda_graph_event
    assert inner.current_microbatch == 7


def test_hybrid_wrapper_drops_runtime_packed_params_when_graph_has_no_contract(
    monkeypatch,
):
    from types import SimpleNamespace

    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.transformer.module import GraphableMegatronModule

    inner = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(inner)
    inner.config = SimpleNamespace(delay_offload_until_cuda_graph=False)
    wrapper = object.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = inner
    wrapper.config = SimpleNamespace(cuda_graph_modules=[])
    wrapper._te_cuda_graph_sample_kwarg_names = frozenset()
    hidden_states = torch.ones(2, 1, 4)
    replay_kwargs = {}

    def fake_graph_replay(_self, *args, **kwargs):
        graph_hidden_states = (
            args[0] if args else kwargs.pop("hidden_states")
        )
        assert graph_hidden_states is hidden_states
        replay_kwargs.update(kwargs)
        return (hidden_states,)

    monkeypatch.setattr(
        GraphableMegatronModule, "_te_cuda_graph_replay", fake_graph_replay
    )

    replayed, context = wrapper._te_cuda_graph_replay(
        hidden_states=hidden_states,
        packed_seq_params=_make_packed_seq_params(),
    )

    assert replayed is hidden_states
    assert context is None
    assert "packed_seq_params" not in replay_kwargs
    assert not has_packed_seq_params_cuda_graph_kwargs(replay_kwargs)


@pytest.mark.parametrize("layer_kind", ["attention", "moe", "hash_moe"])
def test_hybrid_wrapper_replay_filters_to_captured_signature(
    monkeypatch, layer_kind
):
    from types import SimpleNamespace

    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.transformer.module import GraphableMegatronModule

    inner = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(inner)
    inner.config = SimpleNamespace(delay_offload_until_cuda_graph=False)
    inner.is_moe_layer = layer_kind != "attention"
    if inner.is_moe_layer:
        inner.mlp = SimpleNamespace(
            router=SimpleNamespace(is_hash_layer=layer_kind == "hash_moe")
        )

    packed_seq_params = _make_packed_seq_params()
    sample_kwarg_names = {"padding_mask"}
    expected_flattened_kwargs = {}
    if layer_kind == "attention":
        expected_flattened_kwargs, static_metadata = (
            split_packed_seq_params_for_cuda_graph(packed_seq_params)
        )
        inner._set_te_cuda_graph_packed_seq_params_static_metadata(
            static_metadata, expected_flattened_kwargs
        )
        sample_kwarg_names.update(expected_flattened_kwargs)
        sample_kwarg_names.add("rotary_pos_emb")
    elif layer_kind == "hash_moe":
        sample_kwarg_names.add("input_ids")

    wrapper = object.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = inner
    wrapper.config = SimpleNamespace(cuda_graph_modules=[])
    wrapper._te_cuda_graph_sample_kwarg_names = frozenset(sample_kwarg_names)

    hidden_states = torch.ones(2, 1, 4)
    padding_mask = torch.zeros(1, 2, dtype=torch.bool)
    rotary_pos_emb = torch.ones(2, 1, 1, 4)
    input_ids = torch.ones(1, 2, dtype=torch.long)
    replay_kwargs = {}

    def fake_graph_replay(_self, *args, **kwargs):
        graph_hidden_states = args[0] if args else kwargs.pop("hidden_states")
        assert graph_hidden_states is hidden_states
        replay_kwargs.update(kwargs)
        return (hidden_states,)

    monkeypatch.setattr(
        GraphableMegatronModule, "_te_cuda_graph_replay", fake_graph_replay
    )

    replayed, context = wrapper._te_cuda_graph_replay(
        hidden_states=hidden_states,
        attention_mask=torch.ones(1),
        inference_context=object(),
        rotary_pos_emb=rotary_pos_emb,
        sequence_len_offset=torch.ones(1),
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        input_ids=input_ids,
    )

    assert replayed is hidden_states
    assert context is None
    assert set(replay_kwargs) == sample_kwarg_names
    assert replay_kwargs["padding_mask"] is padding_mask
    assert "packed_seq_params" not in replay_kwargs
    assert has_packed_seq_params_cuda_graph_kwargs(replay_kwargs) == (
        layer_kind == "attention"
    )
    if layer_kind == "attention":
        assert replay_kwargs["rotary_pos_emb"] is rotary_pos_emb
        assert set(expected_flattened_kwargs).issubset(replay_kwargs)
        assert "input_ids" not in replay_kwargs
    elif layer_kind == "hash_moe":
        assert replay_kwargs["input_ids"] is input_ids
    else:
        assert "input_ids" not in replay_kwargs


def test_hybrid_wrapper_nonpacked_mamba_capture_and_replay(monkeypatch):
    from types import SimpleNamespace

    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.module import GraphableMegatronModule

    class _MambaStub(MambaLayer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(cuda_graph_impl="none")
            self.layer_number = 1

        def forward(self, hidden_states, **_kwargs):
            return hidden_states

    class _HyperConnectionStub:
        def __call__(self, hidden_states, **_kwargs):
            aggregated = hidden_states[..., :1]
            return aggregated, torch.ones(1), torch.ones(1), hidden_states

        def fused_h_res_h_post_bda(
            self, _h_res, residual, _h_post, _output_with_bias, **_kwargs
        ):
            return residual

    wrapper = object.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = _MambaStub()
    wrapper.hyper_connection = _HyperConnectionStub()
    wrapper.config = SimpleNamespace(
        fp32_residual_connection=False, params_dtype=None
    )
    wrapper.training = True
    hidden_states = torch.ones(2, 1, 2)

    captured = wrapper._te_cuda_graph_capture(hidden_states)
    assert len(captured) == 1
    assert captured[0] is hidden_states
    assert wrapper._te_cuda_graph_sample_kwarg_names == frozenset()

    replay_kwargs = {}

    def fake_graph_replay(_self, *_args, **kwargs):
        replay_kwargs.update(kwargs)
        return captured

    monkeypatch.setattr(
        GraphableMegatronModule, "_te_cuda_graph_replay", fake_graph_replay
    )
    replayed, context = wrapper._te_cuda_graph_replay(
        hidden_states,
        attention_mask=torch.ones(1),
        inference_context=None,
        rotary_pos_emb=torch.ones(1),
        sequence_len_offset=torch.ones(1),
        packed_seq_params=None,
        padding_mask=torch.ones(1),
        input_ids=torch.ones(1),
    )

    assert replayed is hidden_states
    assert context is None
    assert replay_kwargs == {}


def test_hybrid_wrapper_partial_moe_capture_and_replay_order(monkeypatch):
    from types import SimpleNamespace

    from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    from megatron.core.transformer.module import GraphableMegatronModule

    aggregated = torch.tensor([1.0])
    h_res = torch.tensor([2.0])
    h_post = torch.tensor([3.0])
    residual = torch.tensor([4.0])
    router_hidden = torch.tensor([5.0])
    probs = torch.tensor([6.0])
    inner_residual = torch.tensor([7.0])
    expert_output = torch.tensor([8.0])
    final_output = torch.tensor([9.0])
    lifecycle_events = []

    class _HyperConnectionStub:
        def __call__(self, hidden_states, return_residual=False):
            assert return_residual
            return aggregated, h_res, h_post, residual

        def fused_h_res_h_post_bda(
            self,
            actual_h_res,
            actual_residual,
            actual_h_post,
            output_with_bias,
            **_kwargs,
        ):
            assert actual_h_res is h_res
            assert actual_residual is residual
            assert actual_h_post is h_post
            assert output_with_bias == (expert_output, None)
            lifecycle_events.append("bda")
            return final_output

    class _OffloadInterfaceStub:
        def enter_replay(self):
            lifecycle_events.append("enter")

        def flush_delayed_groups(self):
            lifecycle_events.append("flush")

        def exit_replay(self):
            lifecycle_events.append("exit")

    class _InnerStub(TransformerLayer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.hidden_dropout = 0.0
            self.config = SimpleNamespace(
                bias_dropout_fusion=False,
                delay_offload_until_cuda_graph=True,
            )
            self.off_interface = _OffloadInterfaceStub()

        def _rebuild_te_cuda_graph_packed_seq_params(self, kwargs):
            return None

        def _flatten_te_cuda_graph_packed_seq_params(self, kwargs):
            return None

        def _te_cuda_graph_capture(self, actual_aggregated, **_kwargs):
            assert actual_aggregated is aggregated
            return router_hidden, probs, inner_residual

        def resume_moe_experts_after_partial_cudagraph(self, outputs):
            assert outputs == [router_hidden, probs, inner_residual]
            lifecycle_events.append("experts")
            return inner_residual, (expert_output, None)

    wrapper = object.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = _InnerStub()
    wrapper.hyper_connection = _HyperConnectionStub()
    wrapper.config = SimpleNamespace(
        fp32_residual_connection=False,
        params_dtype=None,
        bias_dropout_fusion=False,
    )
    wrapper.training = True
    monkeypatch.setattr(
        HyperConnectionHybridLayer,
        "_inner_is_partial_moe_capture",
        lambda _self: True,
    )

    captured = wrapper._te_cuda_graph_capture(torch.tensor([0.0]))
    assert captured == (
        router_hidden,
        probs,
        inner_residual,
        h_post,
        h_res,
        residual,
    )

    monkeypatch.setattr(
        GraphableMegatronModule,
        "_te_cuda_graph_replay",
        lambda _self, *_args, **_kwargs: (
            lifecycle_events.append("graph") or captured
        ),
    )
    replayed, context = wrapper._te_cuda_graph_replay(torch.tensor([0.0]))
    assert replayed is final_output
    assert context is None
    assert lifecycle_events == ["enter", "graph", "flush", "experts", "bda", "exit"]


def test_transformer_layer_rejects_changed_packed_seq_params_tensor_fields():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    packed_seq_params.cu_seqlens_q_padded = None

    with pytest.raises(AssertionError, match="Tensor fields"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": packed_seq_params})


def test_transformer_layer_rejects_replay_with_overlapping_flattened_kwargs():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    existing_key = f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q"

    with pytest.raises(AssertionError, match="overlap"):
        layer._flatten_te_cuda_graph_packed_seq_params(
            {existing_key: torch.IntTensor([0]), "packed_seq_params": packed_seq_params}
        )


def test_te_cuda_graph_sample_kwargs_include_flattened_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    expected_tensor_kwargs, expected_static_metadata = split_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    attention_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    sample_kwargs = {"attention_mask": attention_mask}

    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs(layer, sample_kwargs, packed_seq_params)

    assert sample_kwargs["attention_mask"] is attention_mask
    assert set(expected_tensor_kwargs).issubset(sample_kwargs)
    for key, value in expected_tensor_kwargs.items():
        assert sample_kwargs[key] is value
    assert layer._get_te_cuda_graph_packed_seq_params_static_metadata() == expected_static_metadata
    assert layer._get_te_cuda_graph_packed_seq_params_tensor_kwarg_names() == tuple(
        sorted(expected_tensor_kwargs)
    )


def test_te_cuda_graph_sample_kwargs_noop_without_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    attention_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    sample_kwargs = {"attention_mask": attention_mask}

    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs(layer, sample_kwargs, None)

    assert sample_kwargs == {"attention_mask": attention_mask}
    assert layer._get_te_cuda_graph_packed_seq_params_static_metadata() is None


def test_te_cuda_graph_sample_kwargs_reject_overlapping_flattened_keys():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    sample_kwargs = {f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q": torch.IntTensor([0])}

    with pytest.raises(AssertionError, match="overlap"):
        _add_packed_seq_params_to_te_cuda_graph_sample_kwargs(
            layer, sample_kwargs, packed_seq_params
        )


def test_te_cuda_graph_partial_attn_only_flow():
    from megatron.core.transformer.enums import CudaGraphModule

    class _ConfigStub:
        def __init__(self, cuda_graph_modules):
            self.cuda_graph_modules = cuda_graph_modules
            self.delay_offload_until_cuda_graph = False

    class _TestLayer(_TransformerLayerCudaGraphStub):
        _te_cuda_graph_replay = TransformerLayer._te_cuda_graph_replay

        def __init__(self, cuda_graph_modules):
            self.config = _ConfigStub(cuda_graph_modules)
            self.attn_called = False
            self.replay_impl_called = False
            self.replay_impl_args = None
            self.replay_impl_kwargs = None
            self.replay_impl_context = None

        def _forward_attention(self, *args, **kwargs):
            self.attn_called = True
            return torch.ones(2, 1, 4) * 2.0, "attn_context"

        def _te_cuda_graph_replay_impl(self, args, kwargs, context):
            self.replay_impl_called = True
            self.replay_impl_args = args
            self.replay_impl_kwargs = kwargs
            self.replay_impl_context = context
            return torch.ones(2, 1, 4) * 3.0

    # Case 1: When CudaGraphModule.attn is captured
    layer_attn = _TestLayer([CudaGraphModule.attn])
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer_attn._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)

    kwargs = {"packed_seq_params": packed_seq_params, "hidden_states": torch.ones(2, 1, 4)}
    layer_attn._te_cuda_graph_replay(**kwargs)

    assert not layer_attn.attn_called
    assert layer_attn.replay_impl_called
    assert layer_attn.replay_impl_context is None
    assert "packed_seq_params" not in layer_attn.replay_impl_kwargs
    assert f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q" in layer_attn.replay_impl_kwargs

    # Case 2: When CudaGraphModule.attn is NOT captured (e.g. only mlp is captured)
    layer_mlp = _TestLayer([CudaGraphModule.mlp])

    kwargs = {"packed_seq_params": packed_seq_params, "hidden_states": torch.ones(2, 1, 4)}
    layer_mlp._te_cuda_graph_replay(**kwargs)

    assert layer_mlp.attn_called
    assert layer_mlp.replay_impl_called
    assert layer_mlp.replay_impl_context == "attn_context"
    assert len(layer_mlp.replay_impl_args) == 1
    assert torch.equal(layer_mlp.replay_impl_args[0], torch.ones(2, 1, 4) * 2.0)
    assert layer_mlp.replay_impl_kwargs == {}


def test_seq_idx_determinism_across_replays():
    cu_seqlens = torch.IntTensor([0, 3, 7, 10])
    cu_seqlens_padded = torch.IntTensor([0, 4, 8, 12])

    params1 = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=10,
    )

    params2 = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=10,
    )

    assert params1.seq_idx is not None
    assert params2.seq_idx is not None
    assert torch.equal(params1.seq_idx, params2.seq_idx)
    assert params1.seq_idx.shape == params2.seq_idx.shape
    assert params1.seq_idx.dtype == torch.int32
