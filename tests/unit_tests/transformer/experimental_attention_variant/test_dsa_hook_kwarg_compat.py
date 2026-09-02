# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility coverage for optional DSA backend hook kwargs."""

import pytest

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels


@pytest.fixture(autouse=True)
def _clear_hook_signature_cache():
    dsa_kernels._HOOK_KWARG_SIGNATURE_CACHE.clear()
    yield
    dsa_kernels._HOOK_KWARG_SIGNATURE_CACHE.clear()


def _run_split_hook(
    *,
    varlen_is_plain_causal=True,
    use_local_indexer_varlen=True,
    single_packed_thd_sequence=True,
    cp_size=1,
):
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
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
        cp_size=cp_size,
        varlen_is_plain_causal=varlen_is_plain_causal,
    )
    return result, marker


def _run_full_hook(
    *,
    varlen_is_plain_causal=True,
    use_local_indexer_varlen=True,
    single_packed_thd_sequence=True,
    cp_size=1,
):
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
        cp_size=cp_size,
        attn_mask_type=None,
        packed_seq_params=None,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        varlen_is_plain_causal=varlen_is_plain_causal,
        use_relu=True,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
    )
    return result, marker


def _run_split_loss_hook(
    *,
    varlen_is_plain_causal=True,
    use_local_indexer_varlen=True,
    single_packed_thd_sequence=True,
    cp_size=1,
):
    marker = object()
    result = dsa_kernels.run_fused_qk_topk_with_loss(
        object(),
        marker,
        marker,
        marker,
        8,
        marker,
        marker,
        128,
        marker,
        marker,
        1.0,
        1.0,
        marker,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
        cp_size=cp_size,
        varlen_is_plain_causal=varlen_is_plain_causal,
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
        calls.append(
            (
                q,
                k,
                weights,
                index_topk,
                starts,
                ends,
                block_size,
                use_local_indexer_varlen,
                single_packed_thd_sequence,
                cp_size,
            )
        )
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: legacy_hook)

    result, marker = _run_split_hook(varlen_is_plain_causal=True, cp_size=1)
    result_cp2, marker_cp2 = _run_split_hook(varlen_is_plain_causal=True, cp_size=2)

    assert result is expected
    assert result_cp2 is expected
    assert calls == [
        (marker, marker, marker, 8, marker, marker, 128, False, False, 1),
        (marker_cp2, marker_cp2, marker_cp2, 8, marker_cp2, marker_cp2, 128, True, True, 2),
    ]


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
    assert seen["use_local_indexer_varlen"] is False
    assert seen["single_packed_thd_sequence"] is False
    assert seen["packed_thd_causal_identity_layout"] is True
    assert seen["packed_thd_single_sequence"] is True


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
    assert "packed_thd_causal_identity_layout" not in seen
    assert "packed_thd_single_sequence" not in seen


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


def test_sparse_attention_hook_filters_static_row_certificate_for_legacy_signature(monkeypatch):
    """The fixed-row certificate must not break an older exact backend hook."""
    expected = object()

    def legacy_hook(query, key, topk_indices, softmax_scale, v_channels, topk_length):
        del query, key, topk_indices, softmax_scale, v_channels, topk_length
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: legacy_hook)

    marker = object()
    result = dsa_kernels.run_fused_absorbed_sparse_attention(
        object(), marker, marker, marker, 1.0, 8, marker, all_topk_rows_nonempty=True
    )

    assert result is expected


def test_opaque_sparse_attention_hook_receives_no_static_row_certificate(monkeypatch):
    """An opaque out-of-tree hook receives only its established positional contract."""
    calls = []
    expected = object()

    def opaque_hook(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    def fail_signature(_fn):
        raise ValueError("opaque callable")

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: opaque_hook)
    monkeypatch.setattr(dsa_kernels.inspect, "signature", fail_signature)

    marker = object()
    result = dsa_kernels.run_fused_absorbed_sparse_attention(
        object(), marker, marker, marker, 1.0, 8, marker, all_topk_rows_nonempty=True
    )

    assert result is expected
    assert len(calls[0][0]) == 6
    assert calls[0][1] == {}


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
            local_packed_cp_rank,
            local_packed_cp_query_start,
            local_packed_cp_query_len,
            pg_collection,
        )
        calls.append(
            (
                query,
                key,
                q_indexer,
                k_indexer,
                indexer_weights,
                indexer_topk,
                use_local_indexer_varlen,
                single_packed_thd_sequence,
                cp_size,
            )
        )
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: full_hook)

    result, marker = _run_full_hook(varlen_is_plain_causal=True, cp_size=1)
    result_cp2, marker_cp2 = _run_full_hook(varlen_is_plain_causal=True, cp_size=2)

    assert result is expected
    assert result_cp2 is expected
    assert calls == [
        (marker, marker, marker, marker, marker, 8, False, False, 1),
        (marker_cp2, marker_cp2, marker_cp2, marker_cp2, marker_cp2, 8, True, True, 2),
    ]


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
    assert seen["use_local_indexer_varlen"] is False
    assert seen["single_packed_thd_sequence"] is False
    assert seen["packed_thd_causal_identity_layout"] is True
    assert seen["packed_thd_single_sequence"] is True


def test_opaque_full_hook_preserves_existing_varlen_kwarg(monkeypatch):
    seen = {}
    expected = object()

    def opaque_full_hook(**kwargs):
        seen.update(kwargs)
        return expected

    def fail_signature(_fn):
        raise ValueError("opaque callable")

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: opaque_full_hook)
    monkeypatch.setattr(dsa_kernels.inspect, "signature", fail_signature)

    result, _ = _run_full_hook(varlen_is_plain_causal=True)

    assert result is expected
    assert seen["varlen_is_plain_causal"] is True
    assert seen["use_local_indexer_varlen"] is False
    assert seen["single_packed_thd_sequence"] is False
    assert "packed_thd_causal_identity_layout" not in seen
    assert "packed_thd_single_sequence" not in seen


def test_kwargs_split_loss_hook_receives_compatible_layout_flags(monkeypatch):
    seen = {}
    expected = object()

    def loss_hook(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: loss_hook)

    result, _ = _run_split_loss_hook(cp_size=1)

    assert result is expected
    assert seen["varlen_is_plain_causal"] is True
    assert seen["use_local_indexer_varlen"] is False
    assert seen["single_packed_thd_sequence"] is False
    assert seen["packed_thd_causal_identity_layout"] is True
    assert seen["packed_thd_single_sequence"] is True


def test_cudnn_hook_prefers_new_cp_independent_layout_flags():
    assert dsa_cudnn_kernels._resolve_packed_layout_flags(
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        packed_thd_causal_identity_layout=True,
        packed_thd_single_sequence=True,
    ) == (True, True)
    assert dsa_cudnn_kernels._resolve_packed_layout_flags(
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        packed_thd_causal_identity_layout=None,
        packed_thd_single_sequence=None,
    ) == (True, True)
