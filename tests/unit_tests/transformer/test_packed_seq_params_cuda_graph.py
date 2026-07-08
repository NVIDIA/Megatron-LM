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

