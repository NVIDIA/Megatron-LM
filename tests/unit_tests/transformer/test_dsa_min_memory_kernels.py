# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the min-memory DSA kernel backends.

These exercise dsa_min_memory and its Triton counterpart directly, against the PyTorch
reference implementations in the same modules. They deliberately do not construct a DSA
attention layer: the kernels take plain tensors, so testing them at this level keeps a
numerical failure attributable to one kernel rather than to the layer that called it.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _accumulate_simplified_learned_k_wgrad,
    _plan_execution,
    _sparse_attention_backward_torch_fp32,
    _sparse_attention_tile,
    dsa_min_memory_gqa,
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
    triton_topk_index_block,
)


class _DummyTPGroup:
    def size(self):
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


def _simplified_test_indexer(hidden_size, head_dim, topk, learned_k=False):
    indexer = SimpleNamespace(
        index_n_heads=1,
        index_head_dim=head_dim,
        index_topk=topk,
        softmax_scale=head_dim**-0.5,
        index_rotary_dim=0,
        rotary_pos_emb=None,
        pg_collection=_DummyPGCollection(),
        config=SimpleNamespace(
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=learned_k,
            rotary_interleaved=False,
        ),
    )
    indexer.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
    indexer.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False) if learned_k else None
    return indexer


def test_sparse_attention_backward_torch_accumulates_repeated_keys_in_fp32():
    torch.manual_seed(1704)
    dtype = torch.bfloat16
    sequence_length, query_length, batch_size = 8, 4, 1
    num_query_heads, num_query_groups = 4, 2
    head_dim, value_dim, q_start = 5, 3, 3
    query = torch.randn(query_length, batch_size, num_query_heads, head_dim, dtype=dtype)
    key = torch.randn(sequence_length, batch_size, num_query_groups, head_dim, dtype=dtype)
    value = torch.randn(sequence_length, batch_size, num_query_groups, value_dim, dtype=dtype)
    # Keys 0 and 1 are deliberately hot across every query to exercise collision-heavy scatter.
    selected_indices = torch.tensor([[[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]]])
    grad_output = torch.randn(
        query_length, batch_size, num_query_heads, value_dim, dtype=torch.float32
    )
    scale = head_dim**-0.5

    actual_grad_query = torch.zeros_like(query, dtype=torch.float32)
    actual_grad_key = torch.zeros_like(key, dtype=torch.float32)
    actual_grad_value = torch.zeros_like(value, dtype=torch.float32)
    _sparse_attention_backward_torch_fp32(
        query,
        key,
        value,
        selected_indices,
        grad_output,
        scale,
        q_start,
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
    )

    query_ref = query.float().requires_grad_(True)
    key_ref = key.float().requires_grad_(True)
    value_ref = value.float().requires_grad_(True)
    repeat_factor = num_query_heads // num_query_groups
    group_outputs = []
    for group_idx in range(num_query_groups):
        head_start = group_idx * repeat_factor
        head_end = head_start + repeat_factor
        query_group = query_ref[:, :, head_start:head_end].permute(1, 2, 0, 3)
        key_group = key_ref[:, :, group_idx].permute(1, 0, 2)
        value_group = value_ref[:, :, group_idx].permute(1, 0, 2)
        key_gather_index = selected_indices[..., None].expand(-1, -1, -1, head_dim)
        value_gather_index = selected_indices[..., None].expand(-1, -1, -1, value_dim)
        selected_key = torch.gather(
            key_group[:, None].expand(-1, query_length, -1, -1), 2, key_gather_index
        )
        selected_value = torch.gather(
            value_group[:, None].expand(-1, query_length, -1, -1), 2, value_gather_index
        )
        scores = torch.einsum("brqd,bqkd->brqk", query_group, selected_key) * scale
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        # Preserve model-dtype probability/output rounding while keeping the oracle leaves and
        # repeated-index accumulation in FP32.
        probs_for_value = probs + (probs.to(dtype).float() - probs).detach()
        group_output = torch.einsum("brqk,bqkd->brqd", probs_for_value, selected_value)
        group_outputs.append(group_output)
    output_ref = torch.cat(group_outputs, dim=1).permute(2, 0, 1, 3)
    output_ref = output_ref.to(dtype).float()
    (output_ref * grad_output).sum().backward()

    torch.testing.assert_close(actual_grad_query, query_ref.grad, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(actual_grad_key, key_ref.grad, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(actual_grad_value, value_ref.grad, rtol=2.0e-5, atol=2.0e-6)


def test_simplified_learned_k_only_persists_full_k_when_cached():
    torch.manual_seed(119)
    seqlen, batch_size, hidden_size = 7, 2, 11
    attention_dim, index_dim = 3, 5
    query = torch.randn(seqlen, batch_size, 4, attention_dim, requires_grad=True)
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk=3, learned_k=True)
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
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk, learned_k=True)
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


@pytest.mark.parametrize("learned_k", [False, True])
@pytest.mark.parametrize("freeze_indexer", [False, True])
def test_simplified_train_main_only_zero_loss_produces_no_indexer_update(learned_k, freeze_indexer):
    torch.manual_seed(654)
    seqlen, batch_size, hidden_size = 6, 1, 8
    num_query_heads, head_dim, topk = 4, 2, 3
    query = torch.randn(seqlen, batch_size, num_query_heads, head_dim, requires_grad=True)
    key = torch.randn(seqlen, batch_size, 1, head_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, head_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = _simplified_test_indexer(hidden_size, head_dim, topk, learned_k=learned_k)
    if freeze_indexer:
        for param in (indexer.linear_q.weight, indexer.linear_k.weight if learned_k else None):
            if param is not None:
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
    indexer_weights = (indexer.linear_q.weight,)
    if learned_k:
        indexer_weights += (indexer.linear_k.weight,)
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


def test_torch_min_memory_forces_full_key_routing_chunk():
    """Routing under torch reads every key, so its top-k stays tie-equivalent to the reference."""
    torch_plan = _plan_execution(1, 8192, 8192, use_triton=False)
    assert torch_plan.routing_key_chunk == 8192
    # ...even when a caller asks for a smaller key chunk.
    assert _plan_execution(1, 8192, 8192, False, key_chunk_override=1024).routing_key_chunk == 8192

    triton_plan = _plan_execution(1, 8192, 8192, use_triton=True)
    assert triton_plan.routing_key_chunk == 1024
    assert _plan_execution(1, 8192, 8192, True, key_chunk_override=2048).routing_key_chunk == 2048


def test_execution_plan_bounds_the_score_tile():
    """The query chunk shrinks with the tile, so long context does not blow up temporaries."""
    # Short sequence: the cap applies, not the budget.
    assert _plan_execution(1, 8192, 8192, use_triton=True).query_chunk == 8192
    # Long sequence under torch, where routing reads all 262144 keys: 256 MiB / (262144 * 4)
    # leaves room for 256 query rows, far below the 8192 cap.
    long_plan = _plan_execution(1, 262144, 262144, use_triton=False)
    assert long_plan.routing_key_chunk == 262144
    assert long_plan.query_chunk == 256
    # Batch enters the same product, so a larger batch shrinks the chunk proportionally.
    assert _plan_execution(4, 262144, 262144, use_triton=False).query_chunk == 64


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_topk_index_block_matches_reference():
    torch.manual_seed(123)
    device = torch.device("cuda")
    batch_size = 2
    query_len = 35
    key_len = 257
    index_heads = 3
    index_head_dim = 32
    topk = 7
    q_start = 256

    # Standard DSA routes BF16 activations and accumulates their dot products in FP32. The
    # Triton kernel deliberately uses Tensor Core input precision, so an FP32-input test with
    # 1e-5 tolerance would incorrectly require an IEEE-FP32 routing contract.
    q_index = torch.randn(
        query_len, batch_size, index_heads, index_head_dim, device=device, dtype=torch.bfloat16
    )
    k_index = torch.randn(key_len, batch_size, index_head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(query_len, batch_size, index_heads, device=device, dtype=torch.bfloat16)
    scores = torch.einsum("qbhd,tbd->bqht", q_index.float(), k_index.float())
    scores = torch.relu(scores)
    scores = (scores * weights.permute(1, 0, 2).unsqueeze(-1).float()).sum(dim=2)
    query_positions = (q_start + torch.arange(query_len, device=device)).view(query_len, 1)
    key_positions = torch.arange(key_len, device=device).view(1, key_len)
    scores = scores.masked_fill((key_positions > query_positions).unsqueeze(0), float("-inf"))
    ref_scores, ref_indices = scores.topk(topk, dim=-1)
    ref_topk_plus_one = scores.topk(topk + 1, dim=-1).values

    tri_scores, tri_indices = triton_topk_index_block(
        q_index, weights, k_index, topk, q_start=q_start, k_start=0
    )

    # Every returned score must correspond to its returned key and be numerically close to the
    # FP32 oracle evaluated on the same BF16 operands.
    ref_scores_at_tri_indices = scores.gather(-1, tri_indices)
    torch.testing.assert_close(tri_scores, ref_scores_at_tri_indices, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        tri_scores, tri_scores.sort(dim=-1, descending=True).values, rtol=0, atol=0
    )

    # A small score perturbation may legitimately exchange nearly tied candidates at the top-k
    # boundary. Require exact support for rows whose reference margin is larger than the measured
    # score error, and otherwise require every selected candidate to remain within that error of
    # the true top-k threshold.
    row_error = (tri_scores - ref_scores_at_tri_indices).abs().amax(dim=-1)
    allowance = row_error + 1.0e-5
    ref_threshold = ref_scores[..., -1]
    assert torch.all(ref_scores_at_tri_indices.amin(dim=-1) >= ref_threshold - allowance)
    ref_margin = ref_topk_plus_one[..., -2] - ref_topk_plus_one[..., -1]
    stable_rows = ref_margin > (2.0 * allowance)
    if stable_rows.any():
        tri_support = tri_indices.sort(dim=-1).values
        ref_support = ref_indices.sort(dim=-1).values
        assert torch.equal(tri_support[stable_rows], ref_support[stable_rows])


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_topk_index_block_large_topk_is_numerically_optimal():
    """Exercise the 256-key sub-block merge used by production top-k=512 routing."""
    torch.manual_seed(789)
    device = torch.device("cuda")
    batch_size = 1
    query_len = 3
    key_len = 1024
    index_heads = 64
    index_head_dim = 128
    topk = 512
    q_start = 764

    q_index = torch.randn(
        query_len, batch_size, index_heads, index_head_dim, device=device, dtype=torch.bfloat16
    )
    k_index = torch.randn(key_len, batch_size, index_head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(query_len, batch_size, index_heads, device=device, dtype=torch.bfloat16)
    weights.mul_((index_heads * index_head_dim) ** -0.5)

    reference_scores = torch.einsum("qbhd,tbd->bqht", q_index.float(), k_index.float())
    reference_scores = torch.relu(reference_scores)
    reference_scores = (reference_scores * weights.permute(1, 0, 2).unsqueeze(-1).float()).sum(
        dim=2
    )
    query_positions = q_start + torch.arange(query_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    reference_scores.masked_fill_(
        key_positions.view(1, 1, key_len) > query_positions.view(1, query_len, 1), float("-inf")
    )

    actual = triton_topk_index_block(q_index, weights, k_index, topk, q_start=q_start, k_start=0)
    assert actual is not None
    actual_scores, actual_indices = actual
    sorted_indices = actual_indices.sort(dim=-1).values
    assert not (sorted_indices[..., 1:] == sorted_indices[..., :-1]).any()

    reference_at_actual = reference_scores.gather(-1, actual_indices)
    score_error = (actual_scores - reference_at_actual).abs()
    max_allowed = 5.0e-3 + 5.0e-3 * reference_at_actual.abs()
    assert torch.all(score_error <= max_allowed)

    reference_top_values, reference_top_indices = reference_scores.topk(topk, dim=-1)
    row_error = score_error.amax(dim=-1)
    allowance = row_error + 1.0e-5
    selected_min = reference_at_actual.amin(dim=-1)
    threshold = reference_top_values[..., -1]
    assert torch.all(selected_min >= threshold - allowance)

    top_plus_one = reference_scores.topk(topk + 1, dim=-1).values
    boundary_margin = top_plus_one[..., -2] - top_plus_one[..., -1]
    stable_rows = boundary_margin > (2.0 * allowance)
    if stable_rows.any():
        actual_support = actual_indices.sort(dim=-1).values
        reference_support = reference_top_indices.sort(dim=-1).values
        assert torch.equal(actual_support[stable_rows], reference_support[stable_rows])


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
