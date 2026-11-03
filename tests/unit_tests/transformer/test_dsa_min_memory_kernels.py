# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the min-memory DSA kernel backends.

These exercise dsa_min_memory and its Triton counterpart directly, against the PyTorch
reference implementations in the same modules. They deliberately do not construct a DSA
attention layer: the kernels take plain tensors, so testing them at this level keeps a
numerical failure attributable to one kernel rather than to the layer that called it.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _MAX_KEY_CHUNK,
    _MAX_QUERY_CHUNK,
    _SCORE_TILE_BUDGET_BYTES,
    _accumulate_simplified_learned_k_wgrad,
    _plan_dense_warmup,
    _plan_execution,
    _simplified_topk_index_tile,
    _sparse_attention_tile,
    dsa_dense_indexer_loss,
    dsa_min_memory_gqa,
    dsa_min_memory_gqa_forward_only,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory_triton import (
    HAVE_TRITON,
    triton_indexer_loss_grad,
    triton_linear_wgrad,
    triton_scatter_selected_grad_to_sequence,
    triton_simplified_gathered_linear_wgrad,
    triton_simplified_index_scores_block,
    triton_simplified_input_norm_stats,
    triton_simplified_selected_index_scores,
    triton_simplified_selected_index_scores_backward,
    triton_simplified_selected_index_scores_backward_qk,
)


class _DummyTPGroup:
    def size(self):
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


def _simplified_test_indexer(hidden_size, head_dim, topk):
    indexer = SimpleNamespace(
        index_n_heads=1,
        index_head_dim=head_dim,
        index_topk=topk,
        softmax_scale=head_dim**-0.5,
        index_rotary_dim=0,
        rotary_pos_emb=None,
        pg_collection=_DummyPGCollection(),
        config=SimpleNamespace(dsa_indexer_mode="simplified", rotary_interleaved=False),
    )
    indexer.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
    indexer.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False)
    return indexer


def test_simplified_learned_k_only_persists_full_k_when_cached():
    torch.manual_seed(119)
    seqlen, batch_size, hidden_size = 7, 2, 11
    attention_dim, index_dim = 3, 5
    query = torch.randn(seqlen, batch_size, 4, attention_dim, requires_grad=True)
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk=3)
    full_k_shape = (seqlen, batch_size, 1, index_dim)

    def saved_shapes(cache_indexer_k):
        shapes = []

        def pack(tensor):
            shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            dsa_min_memory_gqa(
                query,
                key,
                value,
                hidden_states,
                indexer,
                attention_dim**-0.5,
                0.2,
                False,
                query_chunk_size=4,
                key_chunk_size=4,
                cache_indexer_k=cache_indexer_k,
                use_triton=False,
            )
        return shapes

    assert full_k_shape not in saved_shapes(False)
    assert full_k_shape in saved_shapes(True)


def test_simplified_learned_k_bounds_selected_k_scratch(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa_min_memory as min_memory

    torch.manual_seed(120)
    seqlen, batch_size, hidden_size = 70, 1, 11
    attention_dim, index_dim, topk = 3, 5, 70
    query = torch.randn(seqlen, batch_size, 4, attention_dim, requires_grad=True)
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk)
    gathered_support_sizes = []

    original_gather_key = min_memory._gather_simplified_selected_key
    original_gather_indexer_k = min_memory._gather_selected_indexer_k

    def gather_key(key_tensor, indices):
        gathered_support_sizes.append(indices.size(-1))
        return original_gather_key(key_tensor, indices)

    def gather_indexer_k(key_tensor, indices):
        gathered_support_sizes.append(indices.size(-1))
        return original_gather_indexer_k(key_tensor, indices)

    monkeypatch.setattr(min_memory, "_gather_simplified_selected_key", gather_key)
    monkeypatch.setattr(min_memory, "_gather_selected_indexer_k", gather_indexer_k)

    output, loss = dsa_min_memory_gqa(
        query,
        key,
        value,
        hidden_states,
        indexer,
        attention_dim**-0.5,
        0.2,
        False,
        query_chunk_size=seqlen,
        key_chunk_size=17,
        cache_indexer_k=False,
        use_triton=False,
    )
    (output.float().sum() + loss).backward()

    assert gathered_support_sizes
    assert max(gathered_support_sizes) <= 64


@pytest.mark.parametrize("freeze_indexer", [False, True])
def test_simplified_train_main_only_zero_loss_produces_no_indexer_update(freeze_indexer):
    torch.manual_seed(654)
    seqlen, batch_size, hidden_size = 6, 1, 8
    num_query_heads, head_dim, topk = 4, 2, 3
    query = torch.randn(seqlen, batch_size, num_query_heads, head_dim, requires_grad=True)
    key = torch.randn(seqlen, batch_size, 1, head_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, head_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = _simplified_test_indexer(hidden_size, head_dim, topk)
    if freeze_indexer:
        for param in (indexer.linear_q.weight, indexer.linear_k.weight):
            param.requires_grad_(False)

    output, indexer_loss = dsa_min_memory_gqa(
        query,
        key,
        value,
        hidden_states,
        indexer,
        head_dim**-0.5,
        0.0,
        False,
        query_chunk_size=4,
        key_chunk_size=3,
        use_triton=False,
    )
    indexer_weights = (indexer.linear_q.weight, indexer.linear_k.weight)
    grad_inputs = (query, key, value)
    if not freeze_indexer:
        grad_inputs += indexer_weights
    grads = torch.autograd.grad(output.float().sum() + indexer_loss, grad_inputs)

    torch.testing.assert_close(indexer_loss, torch.zeros_like(indexer_loss))
    assert any(torch.count_nonzero(grad) for grad in grads[:3])
    if freeze_indexer:
        assert all(not weight.requires_grad for weight in indexer_weights)
    else:
        for grad in grads[3:]:
            torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_min_memory_routes_over_the_full_key_length():
    """Both backends route over every key, for different reasons."""
    # Torch: a streamed top-k is not tie-equivalent to a single full top-k, so reading every key
    # is what keeps min-memory-torch faithful to reference routing.
    assert _plan_execution(1, 8192, 8192, use_triton=False).routing_key_chunk == 8192
    # ...even when a caller asks for a smaller key chunk.
    assert _plan_execution(1, 8192, 8192, False, key_chunk_override=1024).routing_key_chunk == 8192

    # Triton: its router streams the key dimension internally, so an outer key chunk saves no
    # memory and only repeats the merge.
    assert _plan_execution(1, 8192, 8192, use_triton=True).routing_key_chunk == 8192
    # An explicit override still forces the outer multi-chunk merge, which the tests below need.
    assert _plan_execution(1, 8192, 8192, True, key_chunk_override=2048).routing_key_chunk == 2048


def test_execution_plan_bounds_the_score_tile():
    """The budget bounds the routing tile on both backends, each giving on a different axis."""
    # Short sequence: the query cap applies and the whole key length fits one block.
    short = _plan_execution(1, 8192, 8192, use_triton=True)
    assert short.query_chunk == 8192
    assert short.routing_key_chunk == 8192

    # Triton holds the query chunk and narrows the key block. At batch 1 the budget covers the
    # full 131072-key prefix in one block; batch 2 halves the block rather than doubling the tile.
    assert _plan_execution(1, 131072, 131072, use_triton=True).routing_key_chunk == 131072
    assert _plan_execution(2, 131072, 131072, use_triton=True).routing_key_chunk == 65536

    # Torch may not chunk keys without breaking tie-equivalence, so it shrinks the query chunk
    # to stay inside the same budget, and batch enters the same product.
    long_plan = _plan_execution(1, 262144, 262144, use_triton=False)
    assert long_plan.routing_key_chunk == 262144
    assert long_plan.query_chunk == 4096
    assert _plan_execution(4, 262144, 262144, use_triton=False).query_chunk == 1024

    # Whichever axis gives, the tile the budget exists to bound never exceeds it.
    for batch, seq, triton in [
        (1, 131072, True),
        (2, 131072, True),
        (1, 262144, False),
        (4, 262144, False),
    ]:
        plan = _plan_execution(batch, seq, seq, use_triton=triton)
        tile = batch * plan.query_chunk * plan.routing_key_chunk * 4
        assert tile <= _SCORE_TILE_BUDGET_BYTES, (batch, seq, triton, tile)


def test_tile_sizes_are_optional_on_every_public_entrypoint():
    """Production callers let _plan_execution choose, so no entrypoint may require a tile size.

    Regression test: dsa_min_memory_gqa was given None defaults when the tile sizes stopped being
    configurable, but dsa_min_memory_gqa_forward_only and dsa_dense_indexer_loss kept theirs
    required. The layer passes neither, so the eval and dense-warmup paths raised TypeError at
    runtime -- in a forward that megatron.core.utils catches and logs, so it never surfaced as a
    hard failure.
    """
    for fn in (dsa_min_memory_gqa, dsa_min_memory_gqa_forward_only, dsa_dense_indexer_loss):
        parameters = inspect.signature(fn).parameters
        for name in ("query_chunk_size", "key_chunk_size"):
            assert name in parameters, f"{fn.__name__} lost {name}"
            assert (
                parameters[name].default is None
            ), f"{fn.__name__} requires {name}; production callers do not pass it"


def test_triton_query_chunk_is_not_shrunk_by_long_key_length():
    """Triton keeps the query chunk at the cap and narrows the key block instead.

    Shrinking the query chunk costs launches without bounding anything the key block cannot
    bound more cheaply. Regression test: the planner once charged the query chunk for the full
    routing width, which at sequence 131072 produced a 1024-wide key chunk and 128 routing
    launches per query tile -- 3.1x slower end to end than routing wider blocks.
    """
    plan = _plan_execution(1, 131072, 131072, use_triton=True)
    assert plan.routing_key_chunk == 131072
    assert plan.query_chunk == _MAX_QUERY_CHUNK

    # Torch cannot narrow the key block, so the same budget binds on its query chunk once the
    # key length is long enough that one full-width tile no longer fits.
    assert _plan_execution(1, 524288, 524288, use_triton=False).query_chunk < _MAX_QUERY_CHUNK


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_simplified_selected_scores_matches_reference():
    torch.manual_seed(123)
    device = torch.device("cuda")
    sequence_length = 83
    query_len = 37
    batch_size = 2
    topk = 67
    head_dim = 128
    q_start = 11
    score_scale = head_dim**-0.5

    q_index = torch.randn(query_len, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(sequence_length, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    topk_indices = torch.randint(0, sequence_length, (batch_size, query_len, topk), device=device)

    actual = triton_simplified_selected_index_scores(
        q_index, key, topk_indices, score_scale, q_start
    )
    assert actual is not None
    assert actual.dtype == torch.float32

    key_by_batch = key[:, :, 0, :].permute(1, 0, 2)
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    selected_key = key_by_batch[batch_indices, topk_indices]
    q_by_batch = q_index[:, :, 0, :].permute(1, 0, 2).float()
    expected = (q_by_batch.unsqueeze(2) * selected_key.float()).sum(dim=-1) * score_scale
    query_positions = q_start + torch.arange(query_len, device=device)
    invalid = topk_indices > query_positions.view(1, query_len, 1)
    expected = expected.masked_fill(invalid, float("-inf"))

    assert torch.equal(torch.isneginf(actual), invalid)
    torch.testing.assert_close(actual[~invalid], expected[~invalid], rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_simplified_selected_scores_backward_matches_reference():
    torch.manual_seed(321)
    device = torch.device("cuda")
    sequence_length = 83
    query_len = 37
    batch_size = 2
    topk = 67
    head_dim = 128
    q_start = 11
    score_scale = head_dim**-0.5

    key = torch.randn(sequence_length, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    topk_indices = torch.randint(0, sequence_length, (batch_size, query_len, topk), device=device)
    grad_scores = torch.randn(batch_size, query_len, topk, device=device, dtype=torch.float32)

    actual = triton_simplified_selected_index_scores_backward(
        key, topk_indices, grad_scores, score_scale, q_start
    )
    assert actual is not None
    assert actual.dtype == torch.float32

    query_positions = q_start + torch.arange(query_len, device=device)
    invalid = topk_indices > query_positions.view(1, query_len, 1)
    masked_grad_scores = grad_scores.masked_fill(invalid, 0.0)
    key_by_batch = key[:, :, 0, :].permute(1, 0, 2)
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    selected_key = key_by_batch[batch_indices, topk_indices]
    expected = (masked_grad_scores.unsqueeze(-1) * selected_key.float()).sum(dim=2)
    expected = (expected * score_scale).permute(1, 0, 2).unsqueeze(2).contiguous()

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_simplified_selected_scores_backward_qk_matches_reference(dtype):
    """The first autotuned call must clear atomic dQ between candidate configs."""
    torch.manual_seed(654)
    device = torch.device("cuda")
    query_len, batch_size, topk, head_dim = 37, 2, 67, 96
    q_start = 11
    score_scale = head_dim**-0.5
    q_index = torch.randn(query_len, batch_size, 1, head_dim, device=device, dtype=dtype)
    selected_k = torch.randn(batch_size, query_len, topk, head_dim, device=device, dtype=dtype)
    topk_indices = torch.randint(
        0, q_start + query_len + 5, (batch_size, query_len, topk), device=device
    )
    grad_scores = torch.randn(batch_size, query_len, topk, device=device)

    actual = triton_simplified_selected_index_scores_backward_qk(
        q_index, selected_k, topk_indices, grad_scores, score_scale, q_start
    )
    assert actual is not None
    actual_q, actual_k = actual

    query_positions = q_start + torch.arange(query_len, device=device)
    invalid = topk_indices > query_positions.view(1, query_len, 1)
    masked_grad = grad_scores.masked_fill(invalid, 0.0)
    q = q_index[:, :, 0, :].permute(1, 0, 2).float()
    expected_q = torch.einsum("bqk,bqkd->bqd", masked_grad, selected_k.float()) * score_scale
    expected_q = expected_q.permute(1, 0, 2).unsqueeze(2)
    expected_k = masked_grad.unsqueeze(-1) * q.unsqueeze(2) * score_scale

    torch.testing.assert_close(actual_q, expected_q, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_k, expected_k, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_scatter_selected_grad_repeated_indices_matches_fp32_reference(dtype):
    """The autotuned atomic scatter must clear its FP32 output before every candidate run."""
    torch.manual_seed(711)
    device = torch.device("cuda")
    sequence_length, batch_size, query_len, topk, head_dim = 41, 2, 29, 37, 64
    grad_selected = torch.randn(batch_size, query_len, topk, head_dim, device=device, dtype=dtype)
    # Deliberately create heavy collisions, including the same key repeated within a row.
    topk_indices = torch.randint(
        0, 7, (batch_size, query_len, topk), device=device, dtype=torch.int64
    )

    actual = triton_scatter_selected_grad_to_sequence(grad_selected, topk_indices, sequence_length)
    assert actual is not None
    assert actual.dtype == torch.float32

    expected = torch.zeros(
        sequence_length, batch_size, head_dim, device=device, dtype=torch.float32
    )
    for batch_idx in range(batch_size):
        expected[:, batch_idx].index_add_(
            0,
            topk_indices[batch_idx].reshape(-1),
            grad_selected[batch_idx].reshape(-1, head_dim).float(),
        )
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_simplified_rmsnorm_gathered_wgrad_matches_reference(dtype):
    torch.manual_seed(741)
    device = torch.device("cuda")
    sequence_length, batch_size, hidden_size = 71, 2, 96
    query_len, topk, out_features = 29, 17, 64
    hidden = torch.randn(sequence_length, batch_size, hidden_size, device=device, dtype=dtype)
    grad_output = torch.randn(batch_size, query_len, topk, out_features, device=device, dtype=dtype)
    topk_indices = torch.randint(0, sequence_length, (batch_size, query_len, topk), device=device)
    norm_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    norm_bias = None
    eps = 1.0e-5
    zero_centered_gamma = True

    stats = triton_simplified_input_norm_stats(hidden, eps, "RMSNorm")
    assert stats is not None
    actual = torch.zeros(out_features, hidden_size, device=device, dtype=torch.float32)
    assert triton_simplified_gathered_linear_wgrad(
        grad_output,
        hidden,
        topk_indices,
        norm_weight,
        norm_bias,
        stats,
        "RMSNorm",
        zero_centered_gamma,
        actual,
    )

    effective_weight = (norm_weight + 1.0).float()
    hidden_float = hidden.float()
    normalized = hidden_float * torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized * effective_weight
    normalized = normalized.to(hidden.dtype)
    normalized_by_batch = normalized.permute(1, 0, 2)
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    selected_input = normalized_by_batch[batch_indices, topk_indices]
    expected = (
        grad_output.reshape(-1, out_features)
        .float()
        .t()
        .matmul(selected_input.reshape(-1, hidden_size).float())
    )
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_simplified_layernorm_wgrad_uses_exact_recompute_fallback(dtype):
    torch.manual_seed(742)
    device = torch.device("cuda")
    sequence_length, batch_size, hidden_size, out_features = 71, 2, 96, 64
    hidden = torch.randn(sequence_length, batch_size, hidden_size, device=device, dtype=dtype)
    grad_output = torch.randn(sequence_length, batch_size, out_features, device=device, dtype=dtype)
    norm_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    norm_bias = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1.0e-5
    input_norm = SimpleNamespace(
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
        normalization="LayerNorm",
        zero_centered_gamma=True,
    )

    # LayerNorm deliberately avoids the gathered fast path because its
    # explicit Triton reduction can round differently from F.layer_norm.
    stats = triton_simplified_input_norm_stats(hidden, eps, "LayerNorm")
    assert stats is None

    actual = torch.zeros(out_features, hidden_size, device=device, dtype=torch.float32)
    _accumulate_simplified_learned_k_wgrad(
        grad_output.float(),
        hidden,
        actual,
        input_norm,
        norm_stats=stats,
        row_chunk_size=sequence_length,
    )

    effective_weight = norm_weight + 1.0
    normalized = F.layer_norm(hidden, (hidden_size,), effective_weight, norm_bias, eps)
    expected = (
        grad_output.reshape(-1, out_features)
        .float()
        .t()
        .matmul(normalized.reshape(-1, hidden_size).float())
    )
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-2)


def test_normalized_wgrad_fallback_reuses_supplied_rms_stats():
    torch.manual_seed(743)
    sequence_length, batch_size, hidden_size, out_features = 5, 2, 8, 4
    hidden = torch.randn(sequence_length, batch_size, hidden_size)
    grad_output = torch.randn(sequence_length, batch_size, out_features)
    norm_weight = torch.randn(hidden_size)
    # Deliberately perturb the mathematical RMS statistic so this test distinguishes using the
    # supplied forward statistic from silently recomputing it in the fallback.
    norm_stats = 1.25 * torch.rsqrt(hidden.square().mean(dim=-1) + 1.0e-5)
    input_norm = SimpleNamespace(
        weight=norm_weight,
        bias=None,
        eps=1.0e-5,
        normalization="RMSNorm",
        zero_centered_gamma=False,
    )

    actual = torch.zeros(out_features, hidden_size, dtype=torch.float32)
    _accumulate_simplified_learned_k_wgrad(
        grad_output,
        hidden,
        actual,
        input_norm,
        norm_stats=norm_stats,
        row_chunk_size=2,
        reuse_norm_stats_in_fallback=True,
    )

    normalized = hidden * norm_stats.unsqueeze(-1) * norm_weight
    expected = grad_output.reshape(-1, out_features).t().matmul(normalized.reshape(-1, hidden_size))
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_simplified_score_block_matches_reference():
    torch.manual_seed(456)
    device = torch.device("cuda")
    query_len = 37
    key_len = 73
    batch_size = 2
    head_dim = 128
    q_start = 19
    k_start = 7
    score_scale = head_dim**-0.5

    q_index = torch.randn(query_len, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    key_block = torch.randn(key_len, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)

    actual = triton_simplified_index_scores_block(q_index, key_block, score_scale, q_start, k_start)
    assert actual is not None
    assert actual.dtype == torch.float32

    q_by_batch = q_index[:, :, 0, :].permute(1, 0, 2).float()
    key_by_batch = key_block[:, :, 0, :].permute(1, 0, 2).float()
    expected = (q_by_batch.unsqueeze(2) * key_by_batch.unsqueeze(1)).sum(dim=-1) * score_scale
    query_positions = q_start + torch.arange(query_len, device=device)
    key_positions = k_start + torch.arange(key_len, device=device)
    invalid = key_positions.view(1, 1, key_len) > query_positions.view(1, query_len, 1)
    invalid = invalid.expand(batch_size, -1, -1)
    expected = expected.masked_fill(invalid, float("-inf"))

    assert torch.equal(torch.isneginf(actual), invalid)
    torch.testing.assert_close(actual[~invalid], expected[~invalid], rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_indexer_loss_grad_matches_reference():
    torch.manual_seed(123)
    device = torch.device("cuda")
    selected_scores = torch.randn(2, 9, 17, device=device)
    teacher = torch.softmax(torch.randn(2, 9, 17, device=device), dim=-1)
    scale = torch.tensor(0.125, device=device)

    tri_grad = triton_indexer_loss_grad(selected_scores, teacher, scale)
    student = torch.nn.functional.softmax(selected_scores, dim=-1, dtype=torch.float32)
    teacher_over_student = teacher * student / (student + 1e-10)
    ref_grad = student * teacher_over_student.sum(dim=-1, keepdim=True) - teacher_over_student
    ref_grad = ref_grad * scale

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_linear_wgrad_matches_reference(dtype):
    torch.manual_seed(123)
    device = torch.device("cuda")
    rows = 37
    out_features = 19
    in_features = 41
    grad_output = torch.randn(rows, out_features, device=device, dtype=dtype)
    input_tensor = torch.randn(rows, in_features, device=device, dtype=dtype)
    grad_weight = torch.zeros(out_features, in_features, device=device, dtype=torch.float32)

    assert triton_linear_wgrad(grad_output, input_tensor, grad_weight)

    if dtype == torch.float32:
        # The kernel explicitly requests TF32 dot inputs with FP32 accumulation. Emulate the
        # hardware's round-to-nearest-even 10-bit mantissa instead of comparing against an
        # IEEE-FP32 matmul that the kernel does not claim to implement.
        def _round_to_tf32(tensor):
            bits = tensor.contiguous().view(torch.int32)
            rounding_bias = 0xFFF + ((bits >> 13) & 1)
            return ((bits + rounding_bias) & ~0x1FFF).view(torch.float32)

        ref = _round_to_tf32(grad_output).t().matmul(_round_to_tf32(input_tensor))
    else:
        ref = grad_output.float().t().matmul(input_tensor.float())
    if dtype == torch.float32:
        # Different autotuned BLOCK_N choices regroup the FP32 partial sums after TF32 operand
        # rounding. Bound the resulting reduction error relative to the WGRAD magnitude rather
        # than requiring the same reduction tree as cuBLAS.
        error = (grad_weight - ref).abs()
        max_allowed = 1.0e-2 + 5.0e-3 * ref.abs()
        assert torch.all(error <= max_allowed), (
            f"TF32 WGRAD error exceeded its mixed-precision bound: "
            f"max_abs={error.max().item():.6e}, "
            f"max_allowed={max_allowed.max().item():.6e}"
        )
    else:
        torch.testing.assert_close(grad_weight, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_simplified_routing_merges_key_blocks_into_the_true_topk():
    """Splitting the key prefix into blocks must select the same keys as one pass over it.

    Replaces the coverage the fused router's sub-block merge used to carry. The budget now
    decides how many blocks routing makes, so the outer merge runs in production whenever a
    prefix exceeds one tile, and its index bookkeeping -- the ``+ k_start`` that turns a
    block-local position into a global one -- is what this pins.
    """
    torch.manual_seed(789)
    device = torch.device("cuda")
    batch_size, hidden, head_dim = 1, 64, 128
    seq, topk = 2048, 128
    # A query tile late in the sequence, so the causal prefix spans the whole key range.
    q_start, q_end = seq - 256, seq

    hidden_states = torch.randn(seq, batch_size, hidden, device=device, dtype=torch.bfloat16)
    linear_q_weight = torch.randn(head_dim, hidden, device=device, dtype=torch.bfloat16)
    linear_k_weight = torch.randn(head_dim, hidden, device=device, dtype=torch.bfloat16)
    full_k_index = torch.randn(seq, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.empty(seq, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)

    def route(key_chunk_size):
        return _simplified_topk_index_tile(
            hidden_states,
            key,
            q_start,
            q_end,
            linear_q_weight,
            topk,
            head_dim,
            0,
            None,
            False,
            False,
            head_dim**-0.5,
            key_chunk_size,
            None,
            linear_k_weight,
            full_k_index,
        )

    single_scores, single_indices, _ = route(seq)
    # 512-wide blocks each prune 512 candidates to topk=128, so the merge has to do real work.
    blocked_scores, blocked_indices, _ = route(512)

    assert torch.equal(single_indices, blocked_indices)
    torch.testing.assert_close(single_scores, blocked_scores, rtol=0, atol=0)
    # Indices address global positions and stay inside the causal prefix.
    assert int(blocked_indices.max()) < q_end
    assert int(blocked_indices.min()) >= 0


@pytest.mark.parametrize("num_query_heads", [8, 32, 64, 128, 256])
def test_dense_warmup_plan_bounds_the_teacher_tile(num_query_heads):
    """The teacher scores every main-attention head, so the head count has to enter the budget.

    _plan_execution models routing's [batch, query_chunk, key_chunk]; the teacher allocates
    [batch, num_query_heads, query_chunk, key_chunk]. Sizing warmup from the routing plan
    understated it by the head count, which is why the two plans are separate.
    """
    for batch_size, seq in [(1, 131072), (2, 131072), (1, 262144), (4, 8192)]:
        plan = _plan_dense_warmup(batch_size, seq, seq, num_query_heads)
        tile = batch_size * num_query_heads * plan.query_chunk * plan.key_chunk * 4
        assert tile <= _SCORE_TILE_BUDGET_BYTES, (batch_size, seq, num_query_heads, tile)
        assert plan.query_chunk >= 1 and plan.key_chunk >= 1
        assert plan.key_chunk <= _MAX_KEY_CHUNK
        assert plan.query_chunk <= _MAX_QUERY_CHUNK


def test_dense_warmup_plan_shrinks_the_query_chunk_before_the_key_chunk():
    """Keys are streamed on the outside here, so the query axis is the one that gives."""
    wide = _plan_dense_warmup(1, 131072, 131072, 32)
    wider = _plan_dense_warmup(1, 131072, 131072, 512)
    assert wide.key_chunk == wider.key_chunk == _MAX_KEY_CHUNK
    assert wider.query_chunk < wide.query_chunk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize(
    "num_query_heads, num_query_groups", [(8, 1), (32, 1), (32, 8), (64, 8), (128, 8)]
)
def test_dense_teacher_allocation_stays_within_the_budget(num_query_heads, num_query_groups):
    """Measure what the teacher actually allocates, not what the plan computes.

    The plan arithmetic is checked above; this pins it to the tensor it is meant to bound, so a
    change to _dense_teacher_logits_block that reintroduces an extra full-size copy fails here
    rather than at a user's OOM.
    """
    from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
        _dense_teacher_logits_block,
    )

    torch.manual_seed(31)
    device = torch.device("cuda")
    batch_size, head_dim, seq = 1, 64, 1024
    plan = _plan_dense_warmup(batch_size, seq, seq, num_query_heads)
    q_len = min(plan.query_chunk, seq)
    k_len = min(plan.key_chunk, seq)

    query_tile = torch.randn(
        q_len, batch_size, num_query_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key_block = torch.randn(
        k_len, batch_size, num_query_groups, head_dim, device=device, dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    scores = _dense_teacher_logits_block(query_tile, key_block, head_dim**-0.5, 0, 0)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline

    tile = batch_size * num_query_heads * q_len * k_len * 4
    assert scores.shape == (batch_size, num_query_heads, q_len, k_len)
    assert scores.dtype == torch.float32
    assert scores.numel() * scores.element_size() == tile
    assert tile <= _SCORE_TILE_BUDGET_BYTES

    # The result plus one group's transient. With more than one group that transient is a
    # fraction of the tile, so a collect-then-concatenate implementation -- which holds every
    # group and the joined tensor at once -- shows up as roughly twice the tile. At one group
    # the transient is the whole tile either way, so the two are indistinguishable there.
    if num_query_groups > 1:
        assert peak < 1.5 * tile, (num_query_heads, num_query_groups, peak, tile)
    assert peak <= 2 * _SCORE_TILE_BUDGET_BYTES


@pytest.mark.parametrize(
    "needs_grad", [(True, False, True), (True, True, False), (False, True, True)]
)
def test_sparse_attention_rejects_partial_qkv_gradients(needs_grad):
    """Mixed requires_grad has no working backward, so say so instead of failing in autograd.

    The fused backward runs only when all three of Q/K/V need gradients. Below that it drops to
    replaying the forward under autograd, which cannot work while Triton dispatch is on:
    _sparse_attention_tile returns a raw kernel result with no grad_fn.
    """
    torch.manual_seed(17)
    seqlen, batch_size, hidden_size = 8, 1, 12
    attention_dim, index_dim = 4, 6
    needs_q, needs_k, needs_v = needs_grad
    query = torch.randn(seqlen, batch_size, 4, attention_dim, requires_grad=needs_q)
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=needs_k)
    value = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=needs_v)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk=3)

    output, _ = dsa_min_memory_gqa(
        query,
        key,
        value,
        hidden_states,
        indexer,
        attention_dim**-0.5,
        0.2,
        False,
        query_chunk_size=4,
        key_chunk_size=4,
        use_triton=False,
    )
    with pytest.raises(RuntimeError, match="all of query, key and value or none"):
        output.sum().backward()
