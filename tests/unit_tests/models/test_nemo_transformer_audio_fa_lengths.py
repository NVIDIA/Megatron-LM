# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect

import pytest
import torch

from megatron.core.models.audio import nemo_transformer_encoder


def _install_flash(monkeypatch):
    """Replace both flash-attn entry points and record their calls."""
    calls = {"dense": [], "varlen": []}

    def fake_dense(query, key, value, **kwargs):
        calls["dense"].append({"q": query.detach().clone(), "kwargs": dict(kwargs)})
        return torch.zeros_like(query)

    def fake_varlen(query, key, value, cu_q, cu_k, max_q, max_k, **kwargs):
        calls["varlen"].append(
            {
                "q": query.detach().clone(),
                "k": key.detach().clone(),
                "v": value.detach().clone(),
                "cu_q": cu_q.detach().clone(),
                "cu_k": cu_k.detach().clone(),
                "max_q": max_q,
                "max_k": max_k,
                "kwargs": dict(kwargs),
            }
        )
        return query

    monkeypatch.setattr(nemo_transformer_encoder, "_flash_attn_func", fake_dense)
    monkeypatch.setattr(nemo_transformer_encoder, "_flash_attn_varlen_func", fake_varlen)
    return calls


def _identity_(linear):
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.in_features))
        if linear.bias is not None:
            linear.bias.zero_()


def _attention(dim=4, **kwargs):
    module = nemo_transformer_encoder.MultiHeadAttentionWithFA(
        dim_in=dim, dim_out=dim, num_heads=1, dropout=0.0, **kwargs
    )
    module.eval()
    return module


def test_varlen_symbol_is_flash_attn_v2_not_cute():
    source = inspect.getsource(nemo_transformer_encoder._flash_attn_varlen_func)
    assert "from flash_attn import flash_attn_varlen_func" in source
    assert "from flash_attn.cute import" not in source


def test_equal_lengths_use_only_dense_flash_and_optional_window(monkeypatch):
    calls = _install_flash(monkeypatch)
    bounded = _attention(causal_mask=True, left_context=4)
    unbounded = _attention()
    tokens = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 5])

    bounded(tokens, lengths=lengths)
    unbounded(tokens, lengths=lengths)
    unbounded(tokens)

    assert calls["varlen"] == []
    assert len(calls["dense"]) == 3
    assert calls["dense"][0]["kwargs"]["window_size"] == (4, 0)
    assert calls["dense"][0]["q"].shape[:2] == (2, 5)
    assert "window_size" not in calls["dense"][1]["kwargs"]
    assert "window_size" not in calls["dense"][2]["kwargs"]


def test_unequal_lengths_pack_with_broadcast_and_project_packed_rows(monkeypatch):
    calls = _install_flash(monkeypatch)
    module = _attention()
    _identity_(module.w_query)
    _identity_(module.w_key)
    _identity_(module.w_value)
    _identity_(module.out_proj)
    projected_shapes = []

    def _record_proj(_layer, inputs, _output):
        projected_shapes.append(tuple(inputs[0].shape))

    module.out_proj.register_forward_hook(_record_proj)
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).view(2, 5, 4)

    output = module(hidden, lengths=torch.tensor([5, 2], dtype=torch.int64))

    assert calls["dense"] == []
    assert len(calls["varlen"]) == 1
    recorded = calls["varlen"][0]
    assert recorded["q"].shape[0] == 7
    assert recorded["k"].shape[0] == 7
    assert recorded["v"].shape[0] == 7
    assert recorded["cu_q"].dtype == torch.int32
    assert recorded["cu_k"].dtype == torch.int32
    assert recorded["cu_q"].tolist() == [0, 5, 7]
    assert recorded["cu_k"].tolist() == [0, 5, 7]
    assert recorded["max_q"] == 5
    assert recorded["max_k"] == 5
    assert isinstance(recorded["max_q"], int)
    assert recorded["kwargs"]["window_size"] == (-1, -1)
    assert "dropout_p" in recorded["kwargs"]
    assert "causal" in recorded["kwargs"]
    assert projected_shapes == [(7, 4)]
    expected = hidden.new_zeros(2, 5, 4)
    expected[0] = hidden[0]
    expected[1, :2] = hidden[1, :2]
    torch.testing.assert_close(output, expected)


def test_square_batch_packs_rows_with_broadcast_compare(monkeypatch):
    calls = _install_flash(monkeypatch)
    module = _attention()
    for linear in (module.w_query, module.w_key, module.w_value, module.out_proj):
        _identity_(linear)
    hidden = torch.arange(3 * 3 * 4, dtype=torch.float32).view(3, 3, 4)
    lengths = torch.tensor([3, 1, 2])

    output = module(hidden, lengths=lengths)

    assert calls["dense"] == []
    recorded = calls["varlen"][0]
    assert recorded["q"].shape[0] == 6
    assert recorded["cu_q"].tolist() == [0, 3, 4, 6]
    assert recorded["cu_q"].dtype == torch.int32
    assert recorded["max_q"] == 3
    expected = hidden.new_zeros(3, 3, 4)
    expected[0] = hidden[0]
    expected[1, :1] = hidden[1, :1]
    expected[2, :2] = hidden[2, :2]
    torch.testing.assert_close(output, expected)


def test_varlen_passes_configured_window(monkeypatch):
    calls = _install_flash(monkeypatch)
    module = _attention(causal_mask=True, left_context=3)

    module(torch.randn(2, 5, 4), lengths=torch.tensor([5, 2]))

    assert calls["dense"] == []
    assert calls["varlen"][0]["kwargs"]["window_size"] == (3, 0)
    assert calls["varlen"][0]["kwargs"]["causal"] is True


@pytest.mark.parametrize(
    "lengths",
    [
        torch.tensor([5]),
        torch.tensor([[5, 2]]),
        torch.tensor([5, -1]),
        torch.tensor([6, 2]),
        [5, 2],
    ],
)
def test_bad_lengths_raise_before_flash(monkeypatch, lengths):
    calls = _install_flash(monkeypatch)
    module = _attention()

    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 4), lengths=lengths)

    assert calls["dense"] == []
    assert calls["varlen"] == []


def test_zero_token_batches_skip_flash_before_dense_path(monkeypatch):
    calls = _install_flash(monkeypatch)
    module = _attention()
    with torch.no_grad():
        module.out_proj.bias.fill_(4.0)

    empty_lengths = module(torch.randn(2, 5, 4), lengths=torch.tensor([0, 0]))
    empty_time = module(torch.randn(2, 0, 4))
    empty_time_and_lengths = module(torch.randn(2, 0, 4), lengths=torch.zeros(2, dtype=torch.int32))

    assert empty_lengths.shape == (2, 5, 4)
    assert torch.count_nonzero(empty_lengths) == 0
    assert empty_time.shape == (2, 0, 4)
    assert empty_time_and_lengths.shape == (2, 0, 4)
    assert calls["dense"] == []
    assert calls["varlen"] == []


def test_fa_block_forwards_lengths_and_sdpa_block_does_not():
    encoder = nemo_transformer_encoder
    fa_cfg = encoder.TransformerEncoderConfig(
        d_model=8, n_heads=2, n_layers=1, drop_rate=0.0, attn_impl="fa"
    )
    fa_block = encoder.TransformerBlock(fa_cfg)
    seen = {}

    def fake_fa(hidden, lengths=None):
        seen["lengths"] = lengths
        return torch.zeros_like(hidden)

    fa_block.mha.forward = fake_fa
    lengths = torch.tensor([3, 1])
    fa_block(torch.randn(2, 4, 8), lengths=lengths)
    assert torch.equal(seen["lengths"], lengths)

    sdpa_cfg = encoder.TransformerEncoderConfig(
        d_model=8, n_heads=2, n_layers=1, drop_rate=0.0, attn_impl="sdpa"
    )
    sdpa_block = encoder.TransformerBlock(sdpa_cfg)
    sdpa_seen = {}

    def fake_sdpa(hidden, attn_mask=None, use_cache=False):
        sdpa_seen["attn_mask_shape"] = None if attn_mask is None else tuple(attn_mask.shape)
        sdpa_seen["use_cache"] = use_cache
        return torch.zeros_like(hidden)

    sdpa_block.mha.forward = fake_sdpa
    mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)
    sdpa_block(torch.randn(2, 4, 8), attn_mask=mask, lengths=lengths)
    assert sdpa_seen["attn_mask_shape"] == (2, 1, 1, 4)
    assert sdpa_seen["use_cache"] is False
