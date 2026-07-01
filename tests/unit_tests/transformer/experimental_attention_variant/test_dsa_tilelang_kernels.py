# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_tilelang_kernels
from megatron.core.transformer.experimental_attention_variant.ops import (
    indexer,
    sparse_mla,
    tilelang_dsa,
    tilelang_indexer_bwd,
    tilelang_indexer_fwd,
    tilelang_sparse_mla_bwd,
    tilelang_utils,
)


def test_run_fused_qk_topk_forwards_to_tilelang_backend(monkeypatch):
    q = torch.empty(2, 1, 3, 4)
    k = torch.empty(5, 1, 4)
    weights = torch.empty(2, 1, 3)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    ends = torch.tensor([3, 5], dtype=torch.int32)
    expected_indices = torch.tensor([[[2, 1], [4, 3]]], dtype=torch.int32)
    call = {}

    def fake_run_fused_qk_topk(
        q_arg, k_arg, weights_arg, index_topk, starts_arg, ends_arg, block_size, use_relu
    ):
        call.update(
            q=q_arg,
            k=k_arg,
            weights=weights_arg,
            index_topk=index_topk,
            starts=starts_arg,
            ends=ends_arg,
            block_size=block_size,
            use_relu=use_relu,
        )
        return expected_indices

    monkeypatch.setattr(
        dsa_tilelang_kernels.tilelang_dsa, "run_fused_qk_topk", fake_run_fused_qk_topk
    )

    result = dsa_tilelang_kernels.run_fused_qk_topk(
        q,
        k,
        weights,
        index_topk=2,
        starts=starts,
        ends=ends,
        block_size=8,
        use_relu=False,
        use_local_indexer_varlen=True,
    )

    indices, topk_length = result
    assert indices is expected_indices
    assert topk_length is None
    assert call["q"] is q
    assert call["k"] is k
    assert call["weights"] is weights
    assert call["index_topk"] == 2
    assert call["starts"] is starts
    assert call["ends"] is ends
    assert call["block_size"] == 8
    assert call["use_relu"] is False


def test_run_fused_qk_topk_preserves_unavailable_backend(monkeypatch):
    def fake_run_fused_qk_topk(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        dsa_tilelang_kernels.tilelang_dsa, "run_fused_qk_topk", fake_run_fused_qk_topk
    )

    result = dsa_tilelang_kernels.run_fused_qk_topk(
        torch.empty(2, 1, 3, 4),
        torch.empty(5, 1, 4),
        torch.empty(2, 1, 3),
        index_topk=2,
        starts=torch.tensor([0, 1], dtype=torch.int32),
        ends=torch.tensor([3, 5], dtype=torch.int32),
        block_size=8,
    )

    assert result is None


def test_run_fused_qk_topk_with_loss_preserves_unavailable_backend(monkeypatch):
    def fake_run_fused_qk_topk_with_loss(**kwargs):
        return None

    monkeypatch.setattr(
        dsa_tilelang_kernels.tilelang_dsa,
        "run_fused_qk_topk_with_loss",
        fake_run_fused_qk_topk_with_loss,
    )

    result = dsa_tilelang_kernels.run_fused_qk_topk_with_loss(
        q=torch.empty(2, 1, 3, 4),
        k=torch.empty(5, 1, 4),
        weights=torch.empty(2, 1, 3),
        index_topk=2,
        starts=torch.tensor([0, 1], dtype=torch.int32),
        ends=torch.tensor([3, 5], dtype=torch.int32),
        block_size=8,
        query=torch.empty(2, 1, 3, 4),
        key=torch.empty(5, 1, 1, 4),
        softmax_scale=0.5,
        loss_coeff=0.1,
        pg_collection=SimpleNamespace(),
        config=SimpleNamespace(),
        use_local_indexer_varlen=True,
    )

    assert result is None


def test_run_fused_qk_topk_with_loss_adds_empty_topk_length(monkeypatch):
    q = torch.empty(2, 1, 3, 4)
    k = torch.empty(5, 1, 4)
    weights = torch.empty(2, 1, 3)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    ends = torch.tensor([3, 5], dtype=torch.int32)
    query = torch.empty(2, 1, 3, 4)
    key = torch.empty(5, 1, 1, 4)
    query_valid_rows = torch.tensor([[True, False]])
    pg_collection = SimpleNamespace()
    expected_indices = torch.tensor([[[2, 1], [4, 3]]], dtype=torch.int32)
    expected_loss = torch.tensor(1.25)
    call = {}

    def fake_run_fused_qk_topk_with_loss(**kwargs):
        call.update(kwargs)
        return expected_indices, expected_loss

    monkeypatch.setattr(
        dsa_tilelang_kernels.tilelang_dsa,
        "run_fused_qk_topk_with_loss",
        fake_run_fused_qk_topk_with_loss,
    )

    result = dsa_tilelang_kernels.run_fused_qk_topk_with_loss(
        q=q,
        k=k,
        weights=weights,
        index_topk=2,
        starts=starts,
        ends=ends,
        block_size=8,
        query=query,
        key=key,
        softmax_scale=0.5,
        loss_coeff=0.1,
        pg_collection=pg_collection,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=True,
        use_relu=False,
        config=SimpleNamespace(),
        use_local_indexer_varlen=True,
    )

    indices, topk_length, indexer_loss = result
    assert indices is expected_indices
    assert topk_length is None
    assert indexer_loss is expected_loss
    assert call["q"] is q
    assert call["k"] is k
    assert call["weights"] is weights
    assert call["index_topk"] == 2
    assert call["starts"] is starts
    assert call["ends"] is ends
    assert call["block_size"] == 8
    assert call["query"] is query
    assert call["key"] is key
    assert call["softmax_scale"] == 0.5
    assert call["loss_coeff"] == 0.1
    assert call["pg_collection"] is pg_collection
    assert call["query_valid_rows"] is query_valid_rows
    assert call["calculate_per_token_loss"] is True
    assert call["use_relu"] is False


def test_run_fused_absorbed_sparse_attention_forwards_to_tilelang_backend(monkeypatch):
    query = torch.empty(2, 1, 3, 4)
    key = torch.empty(5, 1, 1, 4)
    topk_indices = torch.tensor([[[0, 1], [1, -1]]], dtype=torch.int32)
    topk_length = torch.tensor([[2, 1]], dtype=torch.int32)
    expected_output = torch.empty(2, 1, 3, 4)
    call = {}

    def fake_run_fused_absorbed_sparse_attention(
        query_arg, key_arg, topk_indices_arg, softmax_scale, v_channels
    ):
        call.update(
            query=query_arg,
            key=key_arg,
            topk_indices=topk_indices_arg,
            softmax_scale=softmax_scale,
            v_channels=v_channels,
        )
        return expected_output

    monkeypatch.setattr(
        dsa_tilelang_kernels.tilelang_dsa,
        "run_fused_absorbed_sparse_attention",
        fake_run_fused_absorbed_sparse_attention,
    )

    result = dsa_tilelang_kernels.run_fused_absorbed_sparse_attention(
        query, key, topk_indices, softmax_scale=0.5, v_channels=4, topk_length=topk_length
    )

    assert result is expected_output
    assert call["query"] is query
    assert call["key"] is key
    assert call["topk_indices"] is topk_indices
    assert call["softmax_scale"] == 0.5
    assert call["v_channels"] == 4


def test_indexer_topk_helpers_mask_invalid_entries():
    logits = torch.tensor([[1.0, 3.0, float("-inf")], [0.0, 2.0, 1.0]])
    requested_indices = torch.tensor([[1, -1, 4], [0, 2, 1]], dtype=torch.int32)

    gathered = indexer.pytorch_extract_topk_scores(logits, requested_indices)

    assert torch.equal(gathered[0], torch.tensor([3.0, float("-inf"), float("-inf")]))
    assert torch.equal(gathered[1], torch.tensor([0.0, 1.0, 2.0]))

    topk_scores, topk_indices = indexer._select_topk_from_logits(logits, topk=4)
    assert topk_scores.shape == (2, 3)
    assert topk_indices.shape == (2, 3)
    assert topk_indices.dtype == torch.int32
    assert -1 in topk_indices[0].tolist()

    empty_scores, empty_indices = indexer._select_topk_from_logits(torch.empty(2, 0), topk=3)
    assert empty_scores.shape == (2, 0)
    assert empty_indices.shape == (2, 0)
    assert empty_indices.dtype == torch.int32


def test_sparse_mla_head_mask_helpers():
    indices = torch.tensor([[[0, -1], [-1, -1]], [[1, 2], [3, -1]]], dtype=torch.int32)

    valid_heads = sparse_mla._valid_head_mask(indices, num_heads=4)

    assert torch.equal(
        valid_heads, torch.tensor([[True, True, False, False], [True, True, True, True]])
    )

    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    zeroed = sparse_mla._zero_invalid_heads(tensor, valid_heads)

    assert torch.equal(zeroed[0, :2], tensor[0, :2])
    assert torch.equal(zeroed[0, 2:], torch.zeros_like(tensor[0, 2:]))
    assert torch.equal(zeroed[1], tensor[1])

    batched_indices = indices.unsqueeze(0)
    batched_valid_heads = sparse_mla._valid_head_mask(batched_indices, num_heads=4)
    assert torch.equal(batched_valid_heads, valid_heads.unsqueeze(0))

    batched_tensor = tensor.unsqueeze(0)
    batched_zeroed = sparse_mla._zero_invalid_heads(batched_tensor, batched_valid_heads)
    assert torch.equal(batched_zeroed, zeroed.unsqueeze(0))

    with pytest.raises(RuntimeError, match="heads must be divisible"):
        sparse_mla._valid_head_mask(indices, num_heads=3)


def test_tilelang_dsa_sanitize_helper():
    topk_indices = torch.tensor([[0, 2, 5], [-1, 3, 4]], dtype=torch.int32)
    topk_scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    starts = torch.tensor([1, 3], dtype=torch.int32)
    ends = torch.tensor([5, 4], dtype=torch.int32)

    sanitized_indices, sanitized_scores = tilelang_dsa._sanitize_fused_topk_outputs(
        topk_indices, starts, ends, topk_scores
    )

    assert torch.equal(sanitized_indices, torch.tensor([[-1, 2, -1], [-1, 3, -1]]))
    assert torch.equal(
        torch.isneginf(sanitized_scores), torch.tensor([[True, False, True], [True, False, True]])
    )


def test_tilelang_dsa_scratch_cache_reuses_buffers(monkeypatch):
    tilelang_dsa._DSA_SCRATCH_CACHE.clear()
    monkeypatch.setattr(tilelang_dsa, "_DSA_SCRATCH_CACHE_TOTAL_BYTES", 0)
    monkeypatch.setattr(tilelang_dsa, "_DSA_SCRATCH_CACHE_MAX_ENTRIES", 1)
    monkeypatch.setattr(tilelang_dsa, "_DSA_SCRATCH_CACHE_MAX_BYTES", 1024)

    first = tilelang_dsa._get_scratch_buffer("a", (2,), torch.float32, torch.device("cpu"))
    first.fill_(3.0)
    reused = tilelang_dsa._get_scratch_buffer("a", (2,), torch.float32, torch.device("cpu"))
    second = tilelang_dsa._get_scratch_buffer("b", (2,), torch.float32, torch.device("cpu"))

    assert reused is first
    assert torch.equal(reused, torch.full((2,), 3.0))
    assert list(tilelang_dsa._DSA_SCRATCH_CACHE) == [
        ("b", (2,), torch.float32, torch.device("cpu"))
    ]
    assert tilelang_dsa._scratch_cache_total_bytes() == second.numel() * second.element_size()


def test_tilelang_kernel_helper_caches_and_env_parsing(monkeypatch):
    monkeypatch.delenv("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", raising=False)
    assert tilelang_utils._env_int("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", 7) == 7

    monkeypatch.setenv("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", "bad")
    assert tilelang_utils._env_int("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", 7) == 7

    monkeypatch.setenv("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", "-3")
    assert tilelang_utils._env_int("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", 7) == 7

    monkeypatch.setenv("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", "2")
    assert tilelang_utils._env_int("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", 7) == 2

    # Shared numeric/layout helpers now live in tilelang_utils.
    assert tilelang_utils._next_power_of_two(0) == 1
    assert tilelang_utils._next_power_of_two(9) == 16
    assert tilelang_utils._ceil_div(9, 4) == 3
    assert tilelang_utils._round_up(9, 4) == 12
    assert tilelang_utils._round_up(9, 1) == 9
    assert tilelang_utils._as_2d_weights(torch.empty(2, 1)).shape == (2, 1)
    assert tilelang_utils._as_2d_weights(torch.empty(2, 1, 1)).shape == (2, 1)
    assert tilelang_utils._normalize_sm_scale(None) is None
    assert tilelang_utils._normalize_sm_scale(torch.tensor(0.5)) == 0.5

    # Kernel-specific helpers stay with their modules.
    assert tilelang_indexer_bwd._canonical_topk(33) == 64
    assert tilelang_indexer_bwd.is_supported_indexer_bwd_head_count(8)
    assert tilelang_indexer_bwd.is_supported_indexer_bwd_head_count(64)
    assert not tilelang_indexer_bwd.is_supported_indexer_bwd_head_count(7)
    assert not tilelang_indexer_bwd.is_supported_indexer_bwd_head_count(72)
    assert tilelang_sparse_mla_bwd._normalize_block_h(12) == 16
    assert tilelang_sparse_mla_bwd._normalize_block_h(40) == 32
    assert tilelang_sparse_mla_bwd._normalize_block_h(80) == 64
    assert tilelang_dsa._is_supported_sparse_mla_head_count(16)
    assert tilelang_dsa._is_supported_sparse_mla_head_count(32)
    assert tilelang_dsa._is_supported_sparse_mla_head_count(64)
    assert tilelang_dsa._is_supported_sparse_mla_head_count(128)
    assert tilelang_dsa._is_supported_sparse_mla_head_count(256, kv_group=2)
    assert not tilelang_dsa._is_supported_sparse_mla_head_count(96)
    assert not tilelang_dsa._is_supported_sparse_mla_head_count(96, kv_group=0)
    # head_kv that is not a power of two >= 16 pads to a larger head dim in the kernels and
    # would index past the real head count, so it must decline to the unfused path.
    assert not tilelang_dsa._is_supported_sparse_mla_head_count(8)
    assert not tilelang_dsa._is_supported_sparse_mla_head_count(48)
    assert not tilelang_dsa._is_supported_sparse_mla_head_count(192)
    assert not tilelang_dsa._is_supported_sparse_mla_head_count(96, kv_group=2)


def test_sparse_mla_canonicalizes_size_one_batch_stride_without_copy():
    tensor_sbhd = torch.empty(256, 1, 1, 4)
    tensor_bshd = tensor_sbhd.permute(1, 0, 2, 3)

    assert tensor_bshd.is_contiguous()
    assert tensor_bshd.stride(0) != tensor_bshd.numel()

    normalized = sparse_mla._canonicalize_batch_stride(tensor_bshd)

    assert normalized.stride(0) == normalized.numel()
    assert normalized.data_ptr() == tensor_bshd.data_ptr()


def test_indexer_bwd_returns_grad_k_in_index_k_dtype(monkeypatch):
    captured = {}

    def fake_kernel(index_q, index_k, weights, topk_indices, grad_scores, grad_q, grad_w, grad_k):
        del index_q, index_k, weights, topk_indices, grad_scores
        captured["grad_k_kernel_dtype"] = grad_k.dtype
        grad_q.fill_(1)
        grad_w.fill_(2)
        grad_k.fill_(3)

    monkeypatch.setattr(tilelang_indexer_bwd, "require_tilelang", lambda: None)
    monkeypatch.setattr(
        tilelang_indexer_bwd, "_get_indexer_bwd_kernel", lambda *_args, **_kwargs: fake_kernel
    )

    index_q = torch.empty((2, 8, 4), dtype=torch.bfloat16)
    index_k = torch.empty((3, 4), dtype=torch.bfloat16)
    weights = torch.empty((2, 8), dtype=torch.float32)
    topk_indices = torch.zeros((2, 1), dtype=torch.int32)
    grad_scores = torch.empty((2, 1), dtype=torch.float32)

    _, _, grad_k = tilelang_indexer_bwd.indexer_bwd_interface(
        index_q, weights, index_k, topk_indices, grad_scores
    )

    assert captured["grad_k_kernel_dtype"] == torch.float32
    assert grad_k.dtype == index_k.dtype
    torch.testing.assert_close(grad_k.float(), torch.full_like(grad_k, 3, dtype=torch.float32))


def test_sparse_mla_delta_pads_partial_sequence_tile(monkeypatch):
    seq_len = 65
    padded_seq_len = 96
    heads = 2
    dim = 4
    o = torch.arange(seq_len * heads * dim, dtype=torch.float32).view(seq_len, heads, dim)
    do = torch.full_like(o, 2.0)

    monkeypatch.setattr(tilelang_sparse_mla_bwd, "require_tilelang", lambda: None)

    def fake_get_preprocess_kernel(H, D):
        assert H == heads
        assert D == dim

        def fake_preprocess_kernel(o_arg, do_arg):
            assert o_arg.shape == (1, padded_seq_len, heads, dim)
            assert do_arg.shape == (1, padded_seq_len, heads, dim)
            assert torch.equal(o_arg[:, :seq_len], o.unsqueeze(0))
            assert torch.equal(do_arg[:, :seq_len], do.unsqueeze(0))
            assert torch.equal(o_arg[:, seq_len:], torch.zeros_like(o_arg[:, seq_len:]))
            assert torch.equal(do_arg[:, seq_len:], torch.zeros_like(do_arg[:, seq_len:]))
            return torch.arange(padded_seq_len * heads, dtype=torch.float32).view(
                1, padded_seq_len, heads
            )

        return fake_preprocess_kernel

    monkeypatch.setattr(
        tilelang_sparse_mla_bwd, "_get_preprocess_kernel", fake_get_preprocess_kernel
    )

    delta = tilelang_sparse_mla_bwd.sparse_mla_delta(o.contiguous(), do.contiguous())

    assert delta.shape == (seq_len, heads)
    assert delta.is_contiguous()
    expected = torch.arange(padded_seq_len * heads, dtype=torch.float32).view(
        padded_seq_len, heads
    )[:seq_len]
    torch.testing.assert_close(delta, expected)


def test_lighting_indexer_indices_preserves_single_head_weight_axis(monkeypatch):
    seen = {}

    def fake_indexer_fwd_interface(
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits, use_relu
    ):
        del index_q, index_k, cu_seqlen_ks, cu_seqlen_ke
        seen["weights_shape"] = weights.shape
        seen["clean_logits"] = clean_logits
        seen["use_relu"] = use_relu
        return torch.arange(6, dtype=torch.float32).view(2, 3)

    monkeypatch.setattr(indexer, "indexer_fwd_interface", fake_indexer_fwd_interface)

    topk_indices = indexer.lighting_indexer_indices(
        index_q=torch.empty(2, 1, 4),
        index_k=torch.empty(3, 4),
        weights=torch.ones(2, 1),
        cu_seqlen_ks=torch.zeros(2, dtype=torch.int32),
        cu_seqlen_ke=torch.full((2,), 3, dtype=torch.int32),
        topk=2,
        use_relu=False,
    )

    assert seen["weights_shape"] == (2, 1)
    assert seen["clean_logits"] is True
    assert seen["use_relu"] is False
    torch.testing.assert_close(topk_indices, torch.tensor([[2, 1], [2, 1]], dtype=torch.int32))


def test_tilelang_kernel_getters_reuse_cached_builders(monkeypatch):
    def make_fake_kernel():
        return lambda *_args, **_kwargs: None

    try:
        monkeypatch.setattr(tilelang_utils, "_TILELANG_KERNEL_CACHE_MAX", 1)
        tilelang_indexer_fwd._tilelang_indexer_fwd_kernel_cache.clear()
        tilelang_indexer_fwd._tilelang_indexer_clean_logits_kernel_cache.clear()
        fwd_builds = []
        clean_builds = []

        def fake_indexer_builder(**kwargs):
            fwd_builds.append(kwargs)
            return make_fake_kernel()

        def fake_clean_builder(**kwargs):
            clean_builds.append(kwargs)
            return make_fake_kernel()

        monkeypatch.setattr(tilelang_indexer_fwd, "tl_indexer_fwd_impl", fake_indexer_builder)
        monkeypatch.setattr(tilelang_indexer_fwd, "clean_logits_", fake_clean_builder)

        first = tilelang_indexer_fwd._get_indexer_fwd_kernel(2, 4)
        second = tilelang_indexer_fwd._get_indexer_fwd_kernel(2, 4)
        third = tilelang_indexer_fwd._get_indexer_fwd_kernel(4, 4)
        clean_first = tilelang_indexer_fwd._get_clean_logits_kernel()
        clean_second = tilelang_indexer_fwd._get_clean_logits_kernel()

        assert first is second
        assert third is not first
        assert len(fwd_builds) == 2
        assert clean_first is clean_second
        assert len(clean_builds) == 1

        tilelang_indexer_bwd._tilelang_indexer_bwd_kernel_cache.clear()
        bwd_builds = []

        def fake_bwd_builder(*args, **kwargs):
            bwd_builds.append((args, kwargs))
            return make_fake_kernel()

        monkeypatch.setattr(tilelang_indexer_bwd, "tl_indexer_bwd_impl", fake_bwd_builder)
        bwd_first = tilelang_indexer_bwd._get_indexer_bwd_kernel(8, 4, 32)
        bwd_second = tilelang_indexer_bwd._get_indexer_bwd_kernel(8, 4, 32)
        bwd_third = tilelang_indexer_bwd._get_indexer_bwd_kernel(16, 4, 32)

        assert bwd_first is bwd_second
        assert bwd_third is not bwd_first
        assert len(bwd_builds) == 2

        tilelang_sparse_mla_bwd._tilelang_sparse_mla_preprocess_kernel_cache.clear()
        tilelang_sparse_mla_bwd._tilelang_sparse_mla_bwd_kernel_cache.clear()
        tilelang_sparse_mla_bwd._tilelang_sparse_mla_postprocess_kernel_cache.clear()
        monkeypatch.setattr(
            tilelang_sparse_mla_bwd, "preprocess", lambda *args, **kwargs: make_fake_kernel()
        )
        monkeypatch.setattr(
            tilelang_sparse_mla_bwd, "bwd", lambda *args, **kwargs: make_fake_kernel()
        )
        monkeypatch.setattr(
            tilelang_sparse_mla_bwd, "postprocess", lambda *args, **kwargs: make_fake_kernel()
        )

        preprocess_first = tilelang_sparse_mla_bwd._get_preprocess_kernel(2, 4)
        preprocess_second = tilelang_sparse_mla_bwd._get_preprocess_kernel(2, 4)
        sparse_bwd_first = tilelang_sparse_mla_bwd._get_bwd_kernel(2, 512, 64, 32, 1, 0.5, 80)
        sparse_bwd_second = tilelang_sparse_mla_bwd._get_bwd_kernel(2, 512, 64, 32, 1, 0.5, 80)
        postprocess_first = tilelang_sparse_mla_bwd._get_postprocess_kernel(512, 64, 1)
        postprocess_second = tilelang_sparse_mla_bwd._get_postprocess_kernel(512, 64, 1)

        assert preprocess_first is preprocess_second
        assert sparse_bwd_first is sparse_bwd_second
        assert postprocess_first is postprocess_second
    finally:
        tilelang_indexer_fwd._tilelang_indexer_fwd_kernel_cache.clear()
        tilelang_indexer_fwd._tilelang_indexer_clean_logits_kernel_cache.clear()
        tilelang_indexer_bwd._tilelang_indexer_bwd_kernel_cache.clear()
        tilelang_sparse_mla_bwd._tilelang_sparse_mla_preprocess_kernel_cache.clear()
        tilelang_sparse_mla_bwd._tilelang_sparse_mla_bwd_kernel_cache.clear()
        tilelang_sparse_mla_bwd._tilelang_sparse_mla_postprocess_kernel_cache.clear()


def test_tilelang_utils_noop_jit_and_require_tilelang(monkeypatch):
    def fn():
        return "ok"

    monkeypatch.setattr(tilelang_utils, "HAVE_TILELANG", False)
    assert tilelang_utils._noop_jit(fn) is fn
    assert tilelang_utils._noop_jit()(fn) is fn
    assert tilelang_utils.tilelang_jit(fn) is fn
    with pytest.raises(ImportError, match="TileLang is required"):
        tilelang_utils.require_tilelang()


def test_compute_topk_target_chunk_sum_shared_and_per_head_paths(monkeypatch):
    tilelang_dsa._DSA_SCRATCH_CACHE.clear()
    monkeypatch.setattr(tilelang_dsa, "_DSA_SCRATCH_CACHE_TOTAL_BYTES", 0)

    query_h = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]])
    key_shared = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    idx_seq = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    valid_seq = torch.tensor([[True, True], [True, False]])

    shared = tilelang_dsa._compute_topk_target_chunk_sum(
        query_h=query_h,
        key_shared=key_shared,
        key_per_head=None,
        s0=0,
        s1=2,
        idx_seq=idx_seq,
        valid_seq=valid_seq,
        softmax_scale=1.0,
        head_chunk_size=1,
        topk_chunk_size=1,
        sk=3,
        hn=2,
    )

    assert shared.shape == (2, 2)
    assert torch.allclose(shared.sum(dim=-1), torch.tensor([2.0, 2.0]), atol=1e-6)
    assert shared[1, 1] == 0

    key_per_head = torch.stack((key_shared, key_shared + 1.0))
    per_head = tilelang_dsa._compute_topk_target_chunk_sum(
        query_h=query_h,
        key_shared=None,
        key_per_head=key_per_head,
        s0=0,
        s1=2,
        idx_seq=idx_seq,
        valid_seq=valid_seq,
        softmax_scale=1.0,
        head_chunk_size=2,
        topk_chunk_size=2,
        sk=3,
        hn=2,
    )

    assert per_head.shape == (2, 2)
    assert torch.allclose(per_head.sum(dim=-1), torch.tensor([2.0, 2.0]), atol=1e-6)
    assert per_head[1, 1] == 0


def test_tilelang_dsa_fused_hook_guard_paths(monkeypatch):
    q = torch.empty(2, 1, 2, 4)
    k = torch.empty(3, 1, 4)
    weights = torch.empty(2, 1, 2)
    starts = torch.zeros(2, dtype=torch.int32)
    ends = torch.ones(2, dtype=torch.int32)

    monkeypatch.setattr(tilelang_dsa, "lighting_indexer_indices", None)
    assert tilelang_dsa.fused_qk_topk_lighting(q, k, weights, 2, starts, ends, 1) is None

    monkeypatch.setattr(tilelang_dsa, "lighting_indexer_indices", lambda *args, **kwargs: None)
    assert tilelang_dsa.fused_qk_topk_lighting(q.squeeze(1), k, weights, 2, starts, ends, 1) is None
    assert tilelang_dsa.fused_qk_topk_lighting(q, k[:, :0], weights, 2, starts, ends, 1) is None

    monkeypatch.setattr(tilelang_dsa, "lighting_indexer", None)
    query = torch.empty(2, 1, 2, 4)
    key = torch.empty(3, 1, 1, 4)
    assert (
        tilelang_dsa.fused_qk_topk_lighting_with_streaming_sparse_kl(
            q,
            k,
            weights,
            2,
            starts,
            ends,
            1,
            query,
            key,
            1.0,
            0.1,
            SimpleNamespace(tp=SimpleNamespace(size=lambda: 1)),
        )
        is None
    )

    def fail_lighting_indexer(*_args, **_kwargs):
        raise AssertionError("unsupported indexer head count should fall back before TileLang")

    monkeypatch.setattr(tilelang_dsa, "lighting_indexer", fail_lighting_indexer)
    assert (
        tilelang_dsa.fused_qk_topk_lighting_with_streaming_sparse_kl(
            torch.empty(2, 1, 7, 4),
            k,
            torch.empty(2, 1, 7),
            2,
            starts,
            ends,
            1,
            torch.empty(2, 1, 2, 4),
            key,
            1.0,
            0.1,
            SimpleNamespace(tp=SimpleNamespace(size=lambda: 1)),
        )
        is None
    )

    monkeypatch.setattr(tilelang_dsa, "SparseMLA", None)
    topk_indices = torch.zeros(1, 2, 64, dtype=torch.int32)
    assert tilelang_dsa.fused_sparse_mla_absorbed(query, key, topk_indices, 1.0, 512) is None

    class FakeSparseMLA:
        @staticmethod
        def apply(q_t, kv_t, idx_t, softmax_scale):
            del q_t, kv_t, idx_t, softmax_scale
            return torch.empty(2, 2, 128), torch.empty(2, 2)

    monkeypatch.setattr(tilelang_dsa, "SparseMLA", FakeSparseMLA)
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(query.squeeze(1), key, topk_indices, 1.0, 512)
        is None
    )
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(query, key.squeeze(2), topk_indices, 1.0, 512)
        is None
    )
    assert tilelang_dsa.fused_sparse_mla_absorbed(query, key, topk_indices[:0], 1.0, 512) is None
    assert tilelang_dsa.fused_sparse_mla_absorbed(query, key, topk_indices[:, :1], 1.0, 512) is None
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(query, key[..., :3], topk_indices, 1.0, 512) is None
    )
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(query, key, topk_indices[..., :63], 1.0, 512) is None
    )

    query_supported = torch.empty(2, 1, 3, 576)
    key_supported = torch.empty(2, 1, 1, 576)
    topk_supported = torch.zeros(1, 2, 64, dtype=torch.int32)
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(
            query_supported, key_supported, topk_supported, 1.0, 256
        )
        is None
    )
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(
            query_supported, key_supported, topk_supported[..., :63], 1.0, 512
        )
        is None
    )

    class FailSparseMLA:
        @staticmethod
        def apply(*_args, **_kwargs):
            raise AssertionError(
                "unsupported SparseMLA head count should fall back before TileLang"
            )

    monkeypatch.setattr(tilelang_dsa, "SparseMLA", FailSparseMLA)
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(
            torch.empty(2, 1, 96, 576), key_supported, topk_supported, 1.0, 512
        )
        is None
    )

    monkeypatch.setattr(tilelang_dsa, "SparseMLA", FakeSparseMLA)
    assert (
        tilelang_dsa.fused_sparse_mla_absorbed(
            query_supported, key_supported, topk_supported, 1.0, 512
        )
        is None
    )


def test_fused_qk_topk_lighting_sanitizes_mocked_tilelang_indices(monkeypatch):
    q = torch.empty(3, 1, 2, 4, dtype=torch.bfloat16)
    k = torch.empty(5, 1, 4, dtype=torch.bfloat16)
    weights = torch.empty(3, 1, 2)
    starts = torch.tensor([0, 2, 4], dtype=torch.int32)
    ends = torch.tensor([2, 4, 5], dtype=torch.int32)
    calls = []

    def fake_lighting_indexer_indices(
        index_q, index_k, index_w, starts_arg, ends_arg, index_topk, use_relu=True
    ):
        del index_k, index_w, index_topk
        calls.append((tuple(index_q.shape), starts_arg.clone(), ends_arg.clone(), use_relu))
        return torch.stack((starts_arg, ends_arg), dim=-1).to(torch.int32)

    monkeypatch.setattr(tilelang_dsa, "lighting_indexer_indices", fake_lighting_indexer_indices)

    topk = tilelang_dsa.fused_qk_topk_lighting(
        q, k, weights, index_topk=2, starts=starts, ends=ends, block_size=2, use_relu=False
    )

    assert torch.equal(topk, torch.tensor([[[0, -1], [2, -1], [4, -1]]], dtype=torch.int32))
    assert [call[0] for call in calls] == [(2, 2, 4), (1, 2, 4)]
    assert all(call[3] is False for call in calls)


def test_fused_sparse_mla_absorbed_batches_mocked_tilelang_outputs(monkeypatch):
    class FakeSparseMLA:
        @staticmethod
        def apply(q_t, kv_t, idx_t, softmax_scale):
            assert q_t.shape == (2, 2, 16, 576)
            assert kv_t.shape == (2, 2, 1, 576)
            assert idx_t.shape == (2, 2, 1, 64)
            assert softmax_scale == 0.25
            batch_sums = q_t.float().sum(dim=(1, 2, 3)).to(dtype=q_t.dtype)
            out = batch_sums.view(q_t.size(0), 1, 1, 1).expand(
                q_t.size(0), q_t.size(1), q_t.size(2), 512
            )
            lse = torch.zeros(q_t.size(0), q_t.size(1), q_t.size(2))
            return out, lse

    monkeypatch.setattr(tilelang_dsa, "SparseMLA", FakeSparseMLA)
    query = torch.zeros(2, 2, 16, 576, dtype=torch.bfloat16)
    query[:, 1].fill_(1.0)
    key = torch.zeros(2, 2, 1, 576, dtype=torch.bfloat16)
    topk_indices = torch.zeros(2, 2, 64, dtype=torch.int32)

    output = tilelang_dsa.fused_sparse_mla_absorbed(
        query, key, topk_indices, softmax_scale=0.25, v_channels=512
    )

    assert output.shape == (2, 2, 16, 512)
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))
    assert torch.equal(output[:, 1], torch.full_like(output[:, 1], 18432.0))


def test_streaming_sparse_kl_path_with_mocked_tilelang_indexer(monkeypatch):
    q = torch.empty(2, 1, 2, 4, dtype=torch.bfloat16)
    k = torch.empty(4, 1, 4, dtype=torch.bfloat16)
    weights = torch.empty(2, 1, 2)
    starts = torch.tensor([0, 0], dtype=torch.int32)
    ends = torch.tensor([4, 4], dtype=torch.int32)
    query = torch.empty(2, 1, 2, 4, dtype=torch.bfloat16)
    key = torch.empty(4, 1, 1, 4, dtype=torch.bfloat16)
    query_valid_rows = torch.tensor([[True, False]])

    def fake_lighting_indexer(
        index_q,
        index_k,
        index_w,
        starts_arg,
        ends_arg,
        index_topk,
        topk_indices=None,
        use_relu=True,
    ):
        del index_k, index_w, starts_arg, ends_arg, topk_indices, use_relu
        topk_scores = torch.zeros(index_q.size(0), index_topk)
        topk = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)[: index_q.size(0)]
        return topk_scores, topk

    def fake_compute_topk_target_chunk_sum(**kwargs):
        idx_seq = kwargs["idx_seq"]
        return torch.ones(idx_seq.shape, dtype=torch.float32, device=idx_seq.device)

    monkeypatch.setattr(tilelang_dsa, "lighting_indexer", fake_lighting_indexer)
    monkeypatch.setattr(tilelang_dsa, "is_supported_indexer_bwd_head_count", lambda *_args: True)
    monkeypatch.setattr(
        tilelang_dsa, "_compute_topk_target_chunk_sum", fake_compute_topk_target_chunk_sum
    )

    topk, loss = tilelang_dsa.fused_qk_topk_lighting_with_streaming_sparse_kl(
        q=q,
        k=k,
        weights=weights,
        index_topk=2,
        starts=starts,
        ends=ends,
        block_size=2,
        query=query,
        key=key,
        softmax_scale=0.5,
        loss_coeff=2.0,
        pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: 1)),
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=False,
        seq_chunk_size=1,
        head_chunk_size=1,
        topk_chunk_size=1,
        use_relu=False,
    )

    assert torch.equal(topk, torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32))
    assert loss.item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_tilelang_ops_decline_non_bfloat16_inputs(monkeypatch, dtype):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("TileLang kernel should not run for non-BF16 inputs")

    class FailSparseMLA:
        @staticmethod
        def apply(*args):
            return fail_if_called(*args)

    monkeypatch.setattr(tilelang_dsa, "lighting_indexer_indices", fail_if_called)
    monkeypatch.setattr(tilelang_dsa, "lighting_indexer", fail_if_called)
    monkeypatch.setattr(tilelang_dsa, "SparseMLA", FailSparseMLA)

    q_indexer = torch.zeros((1, 1, 1, 1), dtype=dtype)
    k_indexer = torch.zeros((1, 1, 1), dtype=dtype)
    weights = torch.zeros((1, 1, 1), dtype=dtype)
    starts = torch.tensor([0], dtype=torch.int32)
    ends = torch.tensor([1], dtype=torch.int32)
    query = torch.zeros((1, 1, 1, 1), dtype=dtype)
    key = torch.zeros((1, 1, 1, 1), dtype=dtype)
    topk_indices = torch.zeros((1, 1, 1), dtype=torch.int32)

    assert (
        tilelang_dsa.fused_qk_topk_lighting(q_indexer, k_indexer, weights, 1, starts, ends, 128)
        is None
    )
    assert (
        tilelang_dsa.fused_qk_topk_lighting_with_streaming_sparse_kl(
            q=q_indexer,
            k=k_indexer,
            weights=weights,
            index_topk=1,
            starts=starts,
            ends=ends,
            block_size=128,
            query=query,
            key=key,
            softmax_scale=1.0,
            loss_coeff=0.01,
            pg_collection=object(),
        )
        is None
    )
    assert tilelang_dsa.fused_sparse_mla_absorbed(query, key, topk_indices, 1.0, 1) is None


def test_fused_sparse_mla_absorbed_accepts_thd_sentinels():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang SparseMLA tests")
    if tilelang_dsa.SparseMLA is None:
        pytest.skip("TileLang SparseMLA kernel is unavailable")

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # Match the sequence bucket so the kernel receives the original B=1 tensor views.
    # This exercises the canonical batch-stride handling rather than hiding it with padding.
    seqlen = 256
    num_heads = 64
    dim = 576
    v_channels = 512
    topk = 64

    query = torch.randn(
        (seqlen, 1, num_heads, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    key = torch.randn((seqlen, 1, 1, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    topk_indices = torch.full((1, seqlen, topk), -1, dtype=torch.int32, device="cuda")
    for row in range(1, seqlen):
        valid = min(row, topk)
        topk_indices[0, row, :valid] = torch.arange(valid, dtype=torch.int32, device="cuda")

    output = tilelang_dsa.fused_sparse_mla_absorbed(
        query, key, topk_indices, softmax_scale=1.0 / math.sqrt(dim), v_channels=v_channels
    )

    assert output is not None
    assert output.shape == (seqlen, 1, num_heads, v_channels)
    assert torch.isfinite(output).all()
    assert output[0].abs().max() == 0

    output.float().square().mean().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert query.grad[0].abs().max() == 0
