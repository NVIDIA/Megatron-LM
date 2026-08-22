import torch
import torch.utils.checkpoint as torch_checkpoint

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.transformer.experimental_attention_variant.dsa import (
    fused_qk_topk_chunked,
    fused_qk_topk_naive,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGroupedSelfAttention,
    _build_shifted_causal_mask,
    compute_gqa_dsa_indexer_loss,
    unfused_grouped_dsa_fn,
)


class _DummyTPGroup:
    def size(self):
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


def test_mamba_stack_spec_uses_dsa_grouped_self_attention():
    attention_module = mamba_stack_spec.submodules.attention_layer.submodules.self_attention.module
    assert attention_module is DSGroupedSelfAttention


def test_compute_gqa_dsa_indexer_loss_dense_and_sparse():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32)
    topk_indices = index_scores.topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    dense_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores.clone(),
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=False,
        pg_collection=pg_collection,
    )
    sparse_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores.clone(),
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
    )

    assert dense_loss.ndim == 0
    assert sparse_loss.ndim == 0
    assert torch.isfinite(dense_loss)
    assert torch.isfinite(sparse_loss)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_matches_reference():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(
        batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
    )
    reference_loss.backward()
    reference_grad = index_scores.grad.clone()

    index_scores.grad = None

    sparse_topk_only_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
    )
    sparse_topk_only_loss.backward()

    torch.testing.assert_close(sparse_topk_only_loss, reference_loss)
    torch.testing.assert_close(index_scores.grad, reference_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_chunked_matches_unchunked():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(
        batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    unchunked_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
    )
    unchunked_loss.backward()
    unchunked_grad = index_scores.grad.clone()

    index_scores.grad = None

    chunked_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        query_chunk_size=3,
    )
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, unchunked_loss)
    torch.testing.assert_close(index_scores.grad, unchunked_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_selected_scores_matches_reference():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(
        batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    selected_index_scores = (
        index_scores.detach().gather(-1, topk_indices).clone().requires_grad_(True)
    )
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
    )
    reference_loss.backward()
    reference_grad = index_scores.grad.gather(-1, topk_indices)

    selected_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=selected_index_scores,
    )
    selected_loss.backward()

    torch.testing.assert_close(selected_loss, reference_loss)
    torch.testing.assert_close(selected_index_scores.grad, reference_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_selected_scores_chunked_matches_reference():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(
        batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    selected_index_scores = (
        index_scores.detach().gather(-1, topk_indices).clone().requires_grad_(True)
    )
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        query_chunk_size=3,
    )
    reference_loss.backward()
    reference_grad = index_scores.grad.gather(-1, topk_indices)

    selected_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        query_chunk_size=3,
        selected_index_scores=selected_index_scores,
    )
    selected_loss.backward()

    torch.testing.assert_close(selected_loss, reference_loss)
    torch.testing.assert_close(selected_index_scores.grad, reference_grad)


def test_unfused_grouped_dsa_fn_output_shape():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    topk_indices = torch.randint(0, seqlen, (batch_size, seqlen, topk))

    output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
    )

    assert output.shape == (seqlen, batch_size, num_heads * head_dim)
    assert output.dtype == query.dtype


def test_unfused_grouped_dsa_fn_matches_dense_reference():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = torch.randint(0, seqlen, (batch_size, seqlen, topk))
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    sparse_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        use_gather=True,
    )
    sparse_output.sum().backward()
    sparse_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    dense_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
    )
    dense_output.sum().backward()

    torch.testing.assert_close(sparse_output, dense_output)
    torch.testing.assert_close(query.grad, sparse_grads[0])
    torch.testing.assert_close(key.grad, sparse_grads[1])
    torch.testing.assert_close(value.grad, sparse_grads[2])


def test_unfused_grouped_dsa_fn_gather_bool_mask_matches_dense_float_mask():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = torch.randint(0, seqlen, (batch_size, seqlen, topk))
    bool_mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.bool)
    bool_mask[:, :, -1] = True
    float_mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32).masked_fill(
        bool_mask, float("-inf")
    )

    gather_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=bool_mask,
        use_gather=True,
    )
    gather_output.sum().backward()
    gather_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    dense_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=float_mask,
    )
    dense_output.sum().backward()

    torch.testing.assert_close(gather_output, dense_output)
    torch.testing.assert_close(query.grad, gather_grads[0])
    torch.testing.assert_close(key.grad, gather_grads[1])
    torch.testing.assert_close(value.grad, gather_grads[2])


def test_unfused_grouped_dsa_fn_chunked_matches_unchunked():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = torch.randint(0, seqlen, (batch_size, seqlen, topk))
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    unchunked_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        use_gather=True,
    )
    unchunked_output.sum().backward()
    unchunked_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    chunked_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        query_chunk_size=2,
        use_gather=True,
    )
    chunked_output.sum().backward()

    torch.testing.assert_close(chunked_output, unchunked_output)
    torch.testing.assert_close(query.grad, unchunked_grads[0])
    torch.testing.assert_close(key.grad, unchunked_grads[1])
    torch.testing.assert_close(value.grad, unchunked_grads[2])


def test_unfused_grouped_dsa_fn_recompute_matches_normal():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = torch.randint(0, seqlen, (batch_size, seqlen, topk))
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    normal_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
    )
    normal_output.sum().backward()
    normal_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    def _compute_recompute_output(
        query_tensor: torch.Tensor, key_tensor: torch.Tensor, value_tensor: torch.Tensor
    ) -> torch.Tensor:
        return unfused_grouped_dsa_fn(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            topk_indices=topk_indices,
            softmax_scale=head_dim**-0.5,
            mask=mask,
        )

    recompute_output = torch_checkpoint.checkpoint(
        _compute_recompute_output,
        query,
        key,
        value,
        use_reentrant=False,
    )
    recompute_output.sum().backward()

    torch.testing.assert_close(recompute_output, normal_output)
    torch.testing.assert_close(query.grad, normal_grads[0])
    torch.testing.assert_close(key.grad, normal_grads[1])
    torch.testing.assert_close(value.grad, normal_grads[2])


def test_unfused_grouped_dsa_fn_gather_recompute_matches_normal():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = torch.randint(0, seqlen, (batch_size, seqlen, topk))
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.bool)
    mask[:, :, -1] = True

    normal_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        query_chunk_size=2,
        use_gather=True,
    )
    normal_output.sum().backward()
    normal_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    def _compute_recompute_output(
        query_tensor: torch.Tensor, key_tensor: torch.Tensor, value_tensor: torch.Tensor
    ) -> torch.Tensor:
        return unfused_grouped_dsa_fn(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            topk_indices=topk_indices,
            softmax_scale=head_dim**-0.5,
            mask=mask,
            query_chunk_size=2,
            use_gather=True,
        )

    recompute_output = torch_checkpoint.checkpoint(
        _compute_recompute_output,
        query,
        key,
        value,
        use_reentrant=False,
    )
    recompute_output.sum().backward()

    torch.testing.assert_close(recompute_output, normal_output)
    torch.testing.assert_close(query.grad, normal_grads[0])
    torch.testing.assert_close(key.grad, normal_grads[1])
    torch.testing.assert_close(value.grad, normal_grads[2])


def test_fused_qk_topk_naive_caps_topk_by_key_length():
    torch.manual_seed(123)

    q = torch.randn(2, 1, 4, 8, dtype=torch.float32)
    k = torch.randn(5, 1, 8, dtype=torch.float32)
    weights = torch.randn(2, 1, 4, dtype=torch.float32)

    _, topk_indices = fused_qk_topk_naive(q=q, k=k, weights=weights, index_topk=4)

    assert topk_indices.shape == (1, 2, 4)
    assert torch.all((topk_indices >= 0) & (topk_indices < 5))


def test_fused_qk_topk_chunked_matches_dense_reference():
    torch.manual_seed(123)

    seqlen_q = 7
    seqlen_k = 9
    batch_size = 2
    num_index_heads = 4
    head_dim = 8
    topk = 3

    q = torch.randn(seqlen_q, batch_size, num_index_heads, head_dim, dtype=torch.float32)
    k = torch.randn(seqlen_k, batch_size, head_dim, dtype=torch.float32)
    weights = torch.randn(seqlen_q, batch_size, num_index_heads, dtype=torch.float32)
    mask = torch.zeros(batch_size, seqlen_q, seqlen_k, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    dense_scores, dense_indices = fused_qk_topk_naive(
        q=q,
        k=k,
        weights=weights,
        index_topk=topk,
        mask=mask,
    )
    chunked_scores, chunked_indices = fused_qk_topk_chunked(
        q=q,
        k=k,
        weights=weights,
        index_topk=topk,
        mask=mask,
        key_chunk_size=4,
    )

    expected_chunked_scores = dense_scores.gather(-1, chunked_indices)
    torch.testing.assert_close(chunked_scores, expected_chunked_scores)
    torch.testing.assert_close(
        torch.sort(chunked_indices, dim=-1).values,
        torch.sort(dense_indices, dim=-1).values,
    )


def test_fused_qk_topk_chunked_recompute_matches_normal():
    torch.manual_seed(123)

    seqlen_q = 7
    seqlen_k = 9
    batch_size = 2
    num_index_heads = 4
    head_dim = 8
    topk = 3

    q = torch.randn(
        seqlen_q, batch_size, num_index_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    k = torch.randn(seqlen_k, batch_size, head_dim, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(
        seqlen_q, batch_size, num_index_heads, dtype=torch.float32, requires_grad=True
    )
    mask = torch.zeros(batch_size, seqlen_q, seqlen_k, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    normal_scores, normal_indices = fused_qk_topk_chunked(
        q=q,
        k=k,
        weights=weights,
        index_topk=topk,
        mask=mask,
        key_chunk_size=4,
    )
    normal_scores.sum().backward()
    normal_grads = (q.grad.clone(), k.grad.clone(), weights.grad.clone())

    q.grad = None
    k.grad = None
    weights.grad = None

    def _compute_chunked_topk(q_tensor, k_tensor, weights_tensor):
        return fused_qk_topk_chunked(
            q=q_tensor,
            k=k_tensor,
            weights=weights_tensor,
            index_topk=topk,
            mask=mask,
            key_chunk_size=4,
        )

    recompute_scores, recompute_indices = torch_checkpoint.checkpoint(
        _compute_chunked_topk,
        q,
        k,
        weights,
        use_reentrant=False,
    )
    recompute_scores.sum().backward()

    torch.testing.assert_close(recompute_scores, normal_scores)
    torch.testing.assert_close(recompute_indices, normal_indices)
    torch.testing.assert_close(q.grad, normal_grads[0])
    torch.testing.assert_close(k.grad, normal_grads[1])
    torch.testing.assert_close(weights.grad, normal_grads[2])


def test_compute_gqa_dsa_indexer_loss_recompute_matches_normal():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(
        batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    def _compute_loss(index_scores_tensor):
        return compute_gqa_dsa_indexer_loss(
            index_scores=index_scores_tensor,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.7,
            sparse_loss=True,
            pg_collection=pg_collection,
        )

    normal_loss = _compute_loss(index_scores)
    normal_loss.backward()
    normal_grad = index_scores.grad.clone()

    index_scores.grad = None

    recompute_loss = torch_checkpoint.checkpoint(
        _compute_loss,
        index_scores,
        use_reentrant=False,
    )
    recompute_loss.backward()

    torch.testing.assert_close(recompute_loss, normal_loss)
    torch.testing.assert_close(index_scores.grad, normal_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_recompute_matches_normal():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(
        batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    def _compute_loss(index_scores_tensor):
        return compute_gqa_dsa_indexer_loss(
            index_scores=index_scores_tensor,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.7,
            sparse_loss=True,
            pg_collection=pg_collection,
            sparse_loss_use_topk_only=True,
        )

    normal_loss = _compute_loss(index_scores)
    normal_loss.backward()
    normal_grad = index_scores.grad.clone()

    index_scores.grad = None

    recompute_loss = torch_checkpoint.checkpoint(
        _compute_loss,
        index_scores,
        use_reentrant=False,
    )
    recompute_loss.backward()

    torch.testing.assert_close(recompute_loss, normal_loss)
    torch.testing.assert_close(index_scores.grad, normal_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_chunked_recompute_matches_normal():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(
        batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    def _compute_loss(index_scores_tensor):
        return compute_gqa_dsa_indexer_loss(
            index_scores=index_scores_tensor,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.7,
            sparse_loss=True,
            pg_collection=pg_collection,
            sparse_loss_use_topk_only=True,
            query_chunk_size=3,
        )

    normal_loss = _compute_loss(index_scores)
    normal_loss.backward()
    normal_grad = index_scores.grad.clone()

    index_scores.grad = None

    recompute_loss = torch_checkpoint.checkpoint(
        _compute_loss,
        index_scores,
        use_reentrant=False,
    )
    recompute_loss.backward()

    torch.testing.assert_close(recompute_loss, normal_loss)
    torch.testing.assert_close(index_scores.grad, normal_grad)


def test_build_shifted_causal_mask_respects_query_offset():
    mask = _build_shifted_causal_mask(query_length=2, key_length=5, query_start_position=3, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(mask, expected)
