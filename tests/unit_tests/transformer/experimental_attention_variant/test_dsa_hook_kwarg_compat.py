# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility coverage for optional DSA backend hook kwargs."""

import pytest

from megatron.core.transformer.experimental_attention_variant import dsa_kernels


@pytest.fixture(autouse=True)
def _clear_hook_signature_cache():
    dsa_kernels._HOOK_KWARG_SIGNATURE_CACHE.clear()
    yield
    dsa_kernels._HOOK_KWARG_SIGNATURE_CACHE.clear()


def _run_split_hook(*, varlen_is_plain_causal=True):
    marker = object()
    result = dsa_kernels.run_fused_qk_topk(
        object(),
        marker,
        marker,
        marker,
        8,
        marker,
        marker,
        128,
        varlen_is_plain_causal=varlen_is_plain_causal,
    )
    return result, marker


def _run_full_hook(*, varlen_is_plain_causal=True):
    marker = object()
    result = dsa_kernels.run_fused_dsa_attention(
        config=object(),
        query=marker,
        key=marker,
        value=None,
        up_v_weight=None,
        q_indexer=marker,
        k_indexer=marker,
        indexer_weights=marker,
        indexer_topk=8,
        softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        absorbed_mla=True,
        cp_size=1,
        attn_mask_type=None,
        packed_seq_params=None,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        varlen_is_plain_causal=varlen_is_plain_causal,
        use_relu=True,
    )
    return result, marker


def test_hook_kwarg_signature_is_inspected_once(monkeypatch):
    inspected = []
    original_signature = dsa_kernels.inspect.signature

    def hook(*, varlen_is_plain_causal=False):
        return varlen_is_plain_causal

    def track_signature(fn):
        inspected.append(fn)
        return original_signature(fn)

    monkeypatch.setattr(dsa_kernels.inspect, "signature", track_signature)

    expected = {"varlen_is_plain_causal": True}
    assert (
        dsa_kernels._hook_kwargs_accepting(hook, varlen_is_plain_causal=True, unsupported=True)
        == expected
    )
    assert (
        dsa_kernels._hook_kwargs_accepting(hook, varlen_is_plain_causal=True, unsupported=True)
        == expected
    )
    assert inspected == [hook]


def test_legacy_exact_split_hook_excludes_new_optional_kwarg(monkeypatch):
    calls = []
    expected = object()

    def legacy_hook(
        *,
        q,
        k,
        weights,
        index_topk,
        starts,
        ends,
        block_size,
        use_relu,
        use_local_indexer_varlen,
        single_packed_thd_sequence,
        local_packed_cp_rank,
        local_packed_cp_query_start,
        local_packed_cp_query_len,
        packed_seq_params,
        cp_size,
    ):
        calls.append((q, k, weights, index_topk, starts, ends, block_size, cp_size))
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: legacy_hook)

    result, marker = _run_split_hook(varlen_is_plain_causal=True)

    assert result is expected
    assert calls == [(marker, marker, marker, 8, marker, marker, 128, 1)]


def test_kwargs_split_hook_receives_new_optional_kwarg(monkeypatch):
    seen = {}
    expected = object()

    def kwargs_hook(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: kwargs_hook)

    result, _ = _run_split_hook(varlen_is_plain_causal=True)

    assert result is expected
    assert seen["varlen_is_plain_causal"] is True


def test_opaque_split_hook_receives_no_new_optional_kwarg(monkeypatch):
    seen = {}
    expected = object()

    def opaque_hook(**kwargs):
        seen.update(kwargs)
        return expected

    def fail_signature(_fn):
        raise ValueError("opaque callable")

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: opaque_hook)
    monkeypatch.setattr(dsa_kernels.inspect, "signature", fail_signature)

    result, _ = _run_split_hook(varlen_is_plain_causal=True)

    assert result is expected
    assert "varlen_is_plain_causal" not in seen


def test_split_hook_type_error_propagates_without_retry(monkeypatch):
    calls = 0

    def failing_hook(**_kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("backend implementation failed")

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: failing_hook)

    with pytest.raises(TypeError, match="backend implementation failed"):
        _run_split_hook(varlen_is_plain_causal=True)
    assert calls == 1


def test_legacy_exact_full_hook_excludes_new_optional_kwarg(monkeypatch):
    calls = []
    expected = object()

    def full_hook(
        *,
        config,
        query,
        key,
        value,
        up_v_weight,
        q_indexer,
        k_indexer,
        indexer_weights,
        indexer_topk,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        calculate_per_token_loss,
        absorbed_mla,
        cp_size,
        attn_mask_type,
        packed_seq_params,
        varlen_starts,
        varlen_ends,
        key_positions,
        query_valid_rows,
        use_relu,
        use_local_indexer_varlen,
        single_packed_thd_sequence,
        local_packed_cp_rank,
        local_packed_cp_query_start,
        local_packed_cp_query_len,
        pg_collection,
    ):
        del (
            config,
            value,
            up_v_weight,
            softmax_scale,
            loss_coeff,
            sparse_loss,
            calculate_per_token_loss,
            absorbed_mla,
            attn_mask_type,
            packed_seq_params,
            varlen_starts,
            varlen_ends,
            key_positions,
            query_valid_rows,
            use_relu,
            use_local_indexer_varlen,
            single_packed_thd_sequence,
            local_packed_cp_rank,
            local_packed_cp_query_start,
            local_packed_cp_query_len,
            pg_collection,
        )
        calls.append((query, key, q_indexer, k_indexer, indexer_weights, indexer_topk, cp_size))
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: full_hook)

    result, marker = _run_full_hook(varlen_is_plain_causal=True)

    assert result is expected
    assert calls == [(marker, marker, marker, marker, marker, 8, 1)]


def test_kwargs_full_hook_receives_new_optional_kwarg(monkeypatch):
    seen = {}
    expected = object()

    def full_hook(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: full_hook)

    result, _ = _run_full_hook(varlen_is_plain_causal=True)

    assert result is expected
    assert seen["varlen_is_plain_causal"] is True
