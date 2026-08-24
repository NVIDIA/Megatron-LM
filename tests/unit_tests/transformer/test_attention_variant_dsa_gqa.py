from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    fused_qk_topk_chunked,
    fused_qk_topk_naive,
    hadamard_transform,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGQACoreAttention,
    DSGroupedSelfAttention,
    SimplifiedDSGQAIndexer,
    SimplifiedDSGQAIndexerSubmodules,
    _DSAZeroParamDependency,
    _indexer_input_norm_spec,
    _normalized_indexer_input,
    _simplified_index_scores,
    _simplified_indexer_input,
    _simplified_indexer_norm_spec,
    compute_gqa_dsa_indexer_loss,
    unfused_grouped_dsa_fn,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import (
    dsa_stack_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    DSAMinMemoryGQAFn,
    _accumulate_simplified_learned_k_wgrad,
    _captured_mass_backward_torch,
    _dense_main_attention_stats,
    _forward_min_memory_impl,
    _native_indexer_loss_wgrad_chunk,
    _project_k_index_block,
    _project_q_index_tile,
    _routing_key_chunk_size,
    _selected_index_scores_backward_torch,
    _selected_index_scores_tile,
    _sparse_attention_backward_torch_fp32,
    _sparse_attention_tile,
    _topk_index_tile,
    dsa_dense_indexer_loss,
    dsa_min_memory_gqa,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory_triton import (
    HAVE_TRITON,
    triton_indexer_loss_grad,
    triton_k_ln_backward_prepare,
    triton_k_ln_param_reduce,
    triton_linear_wgrad,
    triton_scatter_selected_grad_to_sequence,
    triton_selected_index_scores,
    triton_selected_index_scores_from_hidden,
    triton_selected_k_linear,
    triton_simplified_gathered_linear_wgrad,
    triton_simplified_index_scores_block,
    triton_simplified_input_norm_stats,
    triton_simplified_selected_index_scores,
    triton_simplified_selected_index_scores_backward,
    triton_simplified_selected_index_scores_backward_qk,
    triton_topk_index_block,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _DummyTPGroup:
    def size(self):
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


class _DummyRotary:
    def __init__(self, rotary_dim: int, rotary_interleaved: bool = False):
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = None


def _materialized_dense_attention_for_aux_test(query, key, value, q_start):
    repeat_factor = query.size(2) // key.size(2)
    key_heads = key.repeat_interleave(repeat_factor, dim=2)
    value_heads = value.repeat_interleave(repeat_factor, dim=2)
    scores = torch.einsum(
        "qbhd,kbhd->bhqk", query.float(), key_heads.float()
    ) * (query.size(-1) ** -0.5)
    query_positions = torch.arange(q_start, q_start + query.size(0)).view(1, 1, -1, 1)
    key_positions = torch.arange(key.size(0)).view(1, 1, 1, -1)
    scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.einsum(
        "bhqk,kbhd->qbhd",
        probs.to(value.dtype).float(),
        value_heads.float(),
    ).to(value.dtype).float()
    return probs, output


def test_dense_main_attention_stats_matches_materialized_gqa_oracle():
    torch.manual_seed(1701)
    q_start = 1
    query = torch.randn(3, 2, 4, 5)
    key = torch.randn(5, 2, 2, 5)
    value = torch.randn(5, 2, 2, 3)
    selected_indices = torch.tensor(
        [
            [[0, 1], [0, 2], [1, 3]],
            [[0, 1], [1, 2], [0, 3]],
        ]
    )
    scale = query.size(-1) ** -0.5

    dense_output, selected_mass, _, _ = _dense_main_attention_stats(
        query,
        key,
        value,
        selected_indices,
        scale,
        q_start,
        key_chunk_size=2,
        compute_dense_output=True,
    )
    probs, expected_output = _materialized_dense_attention_for_aux_test(
        query, key, value, q_start
    )
    gather_index = selected_indices[:, None].expand(-1, query.size(2), -1, -1)
    expected_mass = torch.gather(probs, -1, gather_index).sum(dim=-1)

    torch.testing.assert_close(dense_output, expected_output, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(selected_mass, expected_mass, rtol=1.0e-5, atol=1.0e-6)


def test_dense_main_attention_stats_bounds_causal_scan_and_reuses_logsumexp(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa_min_memory as min_memory

    torch.manual_seed(1705)
    q_start = 5
    query = torch.randn(3, 1, 4, 5)
    key = torch.randn(20, 1, 2, 5)
    value = torch.randn(20, 1, 2, 3)
    selected_indices = torch.tensor([[[0, 5], [1, 6], [2, 7]]])
    scale = query.size(-1) ** -0.5
    calls = []
    original = min_memory._dense_teacher_logits_block

    def _recording_logits(query_tile, key_block, softmax_scale, tile_q_start, k_start):
        calls.append((k_start, k_start + key_block.size(0)))
        return original(query_tile, key_block, softmax_scale, tile_q_start, k_start)

    monkeypatch.setattr(min_memory, "_dense_teacher_logits_block", _recording_logits)
    dense_output, selected_mass, running_max, running_sum = _dense_main_attention_stats(
        query,
        key,
        value,
        selected_indices,
        scale,
        q_start,
        key_chunk_size=2,
        compute_dense_output=True,
    )
    causal_key_end = q_start + query.size(0)
    assert max(k_end for _, k_end in calls) == causal_key_end
    assert all(k_end <= causal_key_end for _, k_end in calls)

    dense_logsumexp = running_max + torch.log(running_sum)
    torch.testing.assert_close(running_sum, torch.ones_like(running_sum), rtol=0.0, atol=0.0)
    calls.clear()
    reused_output, reused_mass, reused_max, reused_sum = _dense_main_attention_stats(
        query,
        key,
        value,
        selected_indices,
        scale,
        q_start,
        key_chunk_size=2,
        compute_dense_output=True,
        precomputed_logsumexp=dense_logsumexp,
    )
    # Reusing log-sum-exp leaves only the single dense-output pass.
    assert len(calls) == (causal_key_end + 1) // 2
    torch.testing.assert_close(reused_output, dense_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reused_mass, selected_mass, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reused_max, dense_logsumexp)
    torch.testing.assert_close(reused_sum, torch.ones_like(dense_logsumexp))

    calls.clear()
    _captured_mass_backward_torch(
        query,
        key,
        selected_indices,
        selected_mass,
        dense_logsumexp,
        torch.ones_like(dense_logsumexp),
        torch.ones_like(selected_mass),
        scale,
        q_start,
        key_chunk_size=2,
        grad_query=torch.zeros_like(query),
        grad_key=torch.zeros_like(key),
    )
    assert len(calls) == (causal_key_end + 1) // 2
    assert all(k_end <= causal_key_end for _, k_end in calls)




def test_captured_mass_backward_matches_materialized_gqa_oracle():
    torch.manual_seed(1702)
    q_start = 1
    query = torch.randn(3, 2, 4, 5)
    key = torch.randn(5, 2, 2, 5)
    value = torch.randn(5, 2, 2, 3)
    selected_indices = torch.tensor(
        [
            [[0, 1], [0, 2], [1, 3]],
            [[0, 1], [1, 2], [0, 3]],
        ]
    )
    scale = query.size(-1) ** -0.5
    grad_mass = torch.randn(2, 4, 3)
    _, selected_mass, running_max, running_sum = _dense_main_attention_stats(
        query,
        key,
        value,
        selected_indices,
        scale,
        q_start,
        key_chunk_size=2,
        compute_dense_output=False,
    )
    actual_grad_query = torch.zeros_like(query, dtype=torch.float32)
    actual_grad_key = torch.zeros_like(key, dtype=torch.float32)
    _captured_mass_backward_torch(
        query,
        key,
        selected_indices,
        selected_mass,
        running_max,
        running_sum,
        grad_mass,
        scale,
        q_start,
        key_chunk_size=2,
        grad_query=actual_grad_query,
        grad_key=actual_grad_key,
    )

    query_ref = query.detach().requires_grad_(True)
    key_ref = key.detach().requires_grad_(True)
    probs, _ = _materialized_dense_attention_for_aux_test(
        query_ref, key_ref, value, q_start
    )
    gather_index = selected_indices[:, None].expand(-1, query.size(2), -1, -1)
    mass_ref = torch.gather(probs, -1, gather_index).sum(dim=-1)
    (mass_ref * grad_mass).sum().backward()

    torch.testing.assert_close(
        actual_grad_query, query_ref.grad, rtol=2.0e-5, atol=2.0e-6
    )
    torch.testing.assert_close(
        actual_grad_key, key_ref.grad, rtol=2.0e-5, atol=2.0e-6
    )


def test_sparse_attention_backward_torch_accumulates_repeated_keys_in_fp32():
    torch.manual_seed(1704)
    dtype = torch.bfloat16
    sequence_length, query_length, batch_size = 8, 4, 1
    num_query_heads, num_query_groups = 4, 2
    head_dim, value_dim, q_start = 5, 3, 3
    query = torch.randn(
        query_length, batch_size, num_query_heads, head_dim, dtype=dtype
    )
    key = torch.randn(
        sequence_length, batch_size, num_query_groups, head_dim, dtype=dtype
    )
    value = torch.randn(
        sequence_length, batch_size, num_query_groups, value_dim, dtype=dtype
    )
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
            key_group[:, None].expand(-1, query_length, -1, -1),
            2,
            key_gather_index,
        )
        selected_value = torch.gather(
            value_group[:, None].expand(-1, query_length, -1, -1),
            2,
            value_gather_index,
        )
        scores = torch.einsum("brqd,bqkd->brqk", query_group, selected_key) * scale
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        # Preserve model-dtype probability/output rounding while keeping the oracle leaves and
        # repeated-index accumulation in FP32.
        probs_for_value = probs + (probs.to(dtype).float() - probs).detach()
        group_output = torch.einsum(
            "brqk,bqkd->brqd", probs_for_value, selected_value
        )
        group_outputs.append(group_output)
    output_ref = torch.cat(group_outputs, dim=1).permute(2, 0, 1, 3)
    output_ref = output_ref.to(dtype).float()
    (output_ref * grad_output).sum().backward()

    torch.testing.assert_close(actual_grad_query, query_ref.grad, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(actual_grad_key, key_ref.grad, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(actual_grad_value, value_ref.grad, rtol=2.0e-5, atol=2.0e-6)






@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
def test_simplified_indexer_uses_fused_main_qkv_normalized_input(normalization):
    torch.manual_seed(123)
    hidden = torch.randn(5, 2, 8, requires_grad=True)
    weight = torch.randn(8)
    bias = torch.randn(8) if normalization == "LayerNorm" else None
    linear_qkv = SimpleNamespace(
        layer_norm_weight=weight,
        layer_norm_bias=bias,
        eps=1.0e-5,
        skip_norm_and_all_gather=False,
    )
    config = SimpleNamespace(
        normalization=normalization,
        layernorm_epsilon=1.0e-5,
        layernorm_zero_centered_gamma=False,
    )

    norm_spec = _simplified_indexer_norm_spec(linear_qkv, config)
    actual = _simplified_indexer_input(hidden, norm_spec)
    if normalization == "RMSNorm":
        hidden_float = hidden.detach().float()
        expected = (
            hidden_float
            * torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + 1.0e-5)
            * weight.float()
        ).to(hidden.dtype)
    else:
        expected = F.layer_norm(hidden.detach(), (8,), weight, bias, 1.0e-5)

    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_simplified_indexer_recomputes_norm_when_qkv_uses_fused_input_buffer():
    hidden = torch.randn(5, 2, 8, requires_grad=True)
    linear_qkv = SimpleNamespace(
        layer_norm_weight=torch.randn(8),
        skip_norm_and_all_gather=True,
    )
    config = SimpleNamespace(
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        layernorm_zero_centered_gamma=False,
    )

    norm_spec = _simplified_indexer_norm_spec(linear_qkv, config)
    actual = _simplified_indexer_input(hidden, norm_spec)
    hidden_float = hidden.detach().float()
    expected = (
        hidden_float
        * torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * linear_qkv.layer_norm_weight.float()
    ).to(hidden.dtype)

    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
def test_simplified_indexer_input_honors_zero_centered_gamma_without_norm_grads(
    normalization,
):
    torch.manual_seed(321)
    hidden = torch.randn(4, 2, 6, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(6))
    bias = torch.nn.Parameter(torch.randn(6)) if normalization == "LayerNorm" else None
    linear_qkv = SimpleNamespace(
        layer_norm_weight=weight,
        layer_norm_bias=bias,
        eps=2.0e-5,
        skip_norm_and_all_gather=False,
    )
    config = SimpleNamespace(
        normalization=normalization,
        layernorm_epsilon=2.0e-5,
        layernorm_zero_centered_gamma=True,
    )

    norm_spec = _simplified_indexer_norm_spec(linear_qkv, config)
    actual = _simplified_indexer_input(hidden, norm_spec)
    effective_weight = weight.detach() + 1.0
    if normalization == "RMSNorm":
        hidden_float = hidden.detach().float()
        expected = (
            hidden_float
            * torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + 2.0e-5)
            * effective_weight.float()
        ).to(hidden.dtype)
    else:
        expected = F.layer_norm(
            hidden.detach(), (6,), effective_weight, bias.detach(), 2.0e-5
        )

    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert weight.grad is None
    if bias is not None:
        assert bias.grad is None


def test_skip_dsa_zero_dependency_preserves_output_and_produces_zero_param_grads():
    output = torch.randn(4, 3, requires_grad=True)
    indexer_weight = torch.nn.Parameter(torch.randn(5, 7))

    attached = _DSAZeroParamDependency.apply(output, indexer_weight)
    attached.square().sum().backward()

    torch.testing.assert_close(attached, output)
    torch.testing.assert_close(output.grad, 2.0 * output.detach())
    torch.testing.assert_close(indexer_weight.grad, torch.zeros_like(indexer_weight))


def test_dsa_stack_spec_uses_dsa_grouped_self_attention():
    attention_module = dsa_stack_spec.submodules.attention_layer.submodules.self_attention.module
    assert attention_module is DSGroupedSelfAttention


def test_dsa_stack_spec_does_not_mutate_upstream_hybrid_spec():
    """The spec is derived by deep copy; upstream's shared spec must be untouched."""
    upstream_module = (
        hybrid_stack_spec.submodules.attention_layer.submodules.self_attention.module
    )
    assert upstream_module is not DSGroupedSelfAttention


def _causal_mask(seqlen: int, device: torch.device):
    return torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )


def _causal_index_scores(index_scores: torch.Tensor):
    masked_scores = index_scores + _causal_mask(
        index_scores.size(1), index_scores.device
    ).view(1, index_scores.size(1), index_scores.size(2))
    return masked_scores.detach().requires_grad_(index_scores.requires_grad)


def _random_topk_indices(batch_size: int, seqlen: int, topk: int):
    return torch.randn(batch_size, seqlen, seqlen).topk(topk, dim=-1).indices


def _selected_index_scores_reference(q_index, weights, selected_k_index, topk_indices, q_start):
    q = q_index.permute(1, 0, 2, 3).float()
    w = weights.permute(1, 0, 2).float()
    scores = torch.einsum("bqhd,bqkd->bqhk", q, selected_k_index.float())
    scores = torch.relu(scores)
    scores = (scores * w.unsqueeze(-1)).sum(dim=2)
    query_positions = torch.arange(
        q_start,
        q_start + topk_indices.size(1),
        device=topk_indices.device,
        dtype=topk_indices.dtype,
    )
    invalid = topk_indices > query_positions.view(1, topk_indices.size(1), 1)
    return scores.masked_fill(invalid, float("-inf"))


def test_torch_selected_index_score_backward_matches_autograd():
    torch.manual_seed(123)

    batch_size = 2
    query_len = 4
    index_heads = 3
    index_head_dim = 5
    topk = 4
    q_start = 2

    q_index = torch.randn(query_len, batch_size, index_heads, index_head_dim, requires_grad=True)
    weights = torch.randn(query_len, batch_size, index_heads, requires_grad=True)
    selected_k_index = torch.randn(batch_size, query_len, topk, index_head_dim, requires_grad=True)
    topk_indices = torch.tensor(
        [
            [[0, 1, 2, 5], [0, 3, 4, 7], [2, 3, 4, 5], [0, 1, 5, 6]],
            [[0, 2, 3, 6], [1, 2, 4, 8], [0, 1, 4, 6], [2, 4, 5, 9]],
        ],
        dtype=torch.long,
    )
    grad_scores = torch.randn(batch_size, query_len, topk)

    selected_scores = _selected_index_scores_reference(
        q_index, weights, selected_k_index, topk_indices, q_start
    )
    ref_grads = torch.autograd.grad(
        selected_scores,
        (q_index, weights, selected_k_index),
        grad_outputs=grad_scores,
    )

    torch_grads = _selected_index_scores_backward_torch(
        q_index.detach(),
        weights.detach(),
        selected_k_index.detach(),
        topk_indices,
        grad_scores,
        q_start,
    )

    for actual, expected in zip(torch_grads, ref_grads):
        torch.testing.assert_close(actual, expected)


def test_dense_indexer_loss_matches_reference_dense_loss_and_grads():
    torch.manual_seed(123)
    seqlen = 5
    batch_size = 2
    num_query_heads = 4
    num_query_groups = 2
    head_dim = 3
    hidden_size = 7
    index_heads = 2
    index_head_dim = 4
    loss_coeff = 0.3
    softmax_scale = head_dim**-0.5

    query = torch.randn(seqlen, batch_size, num_query_heads, head_dim)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)

    indexer = SimpleNamespace(
        index_n_heads=index_heads,
        index_head_dim=index_head_dim,
        index_topk=2,
        index_rotary_dim=0,
        rotary_pos_emb=None,
        pg_collection=_DummyPGCollection(),
        config=SimpleNamespace(
            layernorm_epsilon=1e-5,
            dsa_indexer_use_hadamard=False,
            rotary_interleaved=False,
        ),
    )
    indexer.linear_q = torch.nn.Linear(hidden_size, index_heads * index_head_dim, bias=False)
    indexer.linear_k = torch.nn.Linear(hidden_size, index_head_dim, bias=False)
    indexer.k_norm = torch.nn.LayerNorm(index_head_dim, eps=1e-5)
    indexer.linear_weights_proj = torch.nn.Linear(hidden_size, index_heads, bias=False)

    q_index = indexer.linear_q(hidden_states).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    k_index = indexer.k_norm(indexer.linear_k(hidden_states)).reshape(
        seqlen, batch_size, index_head_dim
    )
    weights = (
        indexer.linear_weights_proj(hidden_states)
        * (index_heads**-0.5)
        * (index_head_dim**-0.5)
    )
    index_scores, topk_indices = fused_qk_topk_naive(
        q_index,
        k_index,
        weights,
        indexer.index_topk,
        _causal_mask(seqlen, query.device),
    )
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        False,
        indexer.pg_collection,
    )
    dense_loss = dsa_dense_indexer_loss(
        query.detach(),
        key.detach(),
        hidden_states.detach(),
        indexer,
        softmax_scale,
        loss_coeff,
        False,
        query_chunk_size=2,
        key_chunk_size=3,
        use_triton=False,
    )

    torch.testing.assert_close(dense_loss, reference_loss)

    params = (
        indexer.linear_q.weight,
        indexer.linear_k.weight,
        indexer.k_norm.weight,
        indexer.k_norm.bias,
        indexer.linear_weights_proj.weight,
    )
    reference_grads = torch.autograd.grad(reference_loss, params)
    dense_grads = torch.autograd.grad(dense_loss, params)
    for actual, expected in zip(dense_grads, reference_grads):
        torch.testing.assert_close(actual, expected)


def _rotary_freqs(rotary, seqlen: int, rotary_dim: int):
    positions = torch.arange(seqlen, dtype=rotary.inv_freq.dtype, device=rotary.inv_freq.device)
    freqs = torch.outer(positions, rotary.inv_freq[: rotary_dim // 2])
    if not rotary.rotary_interleaved:
        freqs = torch.cat((freqs, freqs), dim=-1)
    else:
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)
    return freqs[:, None, None, :]


def _apply_reference_indexer_rope(
    x: torch.Tensor, rotary, config_rotary_interleaved: bool, rotary_dim: int
):
    x_nope, x_pe = torch.split(x, [x.size(-1) - rotary_dim, rotary_dim], dim=-1)
    x_pe = _apply_rotary_pos_emb_bshd(
        x_pe,
        _rotary_freqs(rotary, x.size(0), rotary_dim),
        rotary_interleaved=config_rotary_interleaved,
        multi_latent_attention=False,
        mscale=1.0,
    )
    return torch.cat([x_nope, x_pe], dim=-1)


def test_transformer_config_accepts_min_memory_backend():
    for backend in ("triton-min-memory", "torch-min-memory"):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend=backend,
            dsa_kernel_cache_routing=True,
            dsa_kernel_cache_indexer_k=True,
            dsa_kernel_cache_selected_scores=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
            dsa_indexer_sparse_loss_use_topk_only=True,
            dsa_kernel_query_block_size=256,
            dsa_kernel_key_block_size=1024,
            dsa_indexer_use_hadamard=True,
            dsa_min_memory_profile=True,
            dsa_min_memory_profile_rank=-1,
        )

        assert config.dsa_kernel_backend == backend
        assert config.dsa_kernel_query_block_size == 256
        assert config.dsa_kernel_key_block_size == 1024
        assert config.dsa_kernel_cache_routing
        assert config.dsa_kernel_cache_indexer_k
        assert config.dsa_kernel_cache_selected_scores
        assert config.dsa_min_memory_profile
        assert config.dsa_min_memory_profile_rank == -1


def test_transformer_config_accepts_standard_main_input_norm():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_standard_indexer_use_main_input_norm=True,
    )

    assert config.dsa_standard_indexer_use_main_input_norm


def test_transformer_config_accepts_disabled_simplified_main_input_norm():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_simplified_indexer_disable_main_input_norm=True,
    )

    assert config.dsa_simplified_indexer_disable_main_input_norm


def test_transformer_config_rejects_disabled_simplified_norm_for_standard_dsa():
    with pytest.raises(AssertionError, match="dsa_indexer_mode='simplified'"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_simplified_indexer_disable_main_input_norm=True,
        )


def test_transformer_config_rejects_standard_main_input_norm_for_simplified_dsa():
    with pytest.raises(AssertionError, match="dsa_indexer_mode='standard'"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_indexer_topk=4,
            dsa_standard_indexer_use_main_input_norm=True,
        )


@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
def test_standard_indexer_projections_use_detached_main_input_norm(normalization):
    torch.manual_seed(1704)
    sequence_length, batch_size, hidden_size = 5, 2, 8
    index_heads, index_dim = 2, 4
    hidden_states = torch.randn(sequence_length, batch_size, hidden_size)
    norm_weight = torch.nn.Parameter(torch.randn(hidden_size))
    norm_bias = (
        torch.nn.Parameter(torch.randn(hidden_size))
        if normalization == "LayerNorm"
        else None
    )
    linear_qkv = SimpleNamespace(
        layer_norm_weight=norm_weight,
        layer_norm_bias=norm_bias,
        eps=1.0e-5,
    )
    norm_config = SimpleNamespace(
        normalization=normalization,
        layernorm_epsilon=1.0e-5,
        layernorm_zero_centered_gamma=False,
    )
    norm_spec = _indexer_input_norm_spec(linear_qkv, norm_config)
    normalized_hidden = _normalized_indexer_input(hidden_states, norm_spec)
    linear_q_weight = torch.randn(
        index_heads * index_dim, hidden_size, requires_grad=True
    )
    linear_k_weight = torch.randn(index_dim, hidden_size, requires_grad=True)
    linear_weights_weight = torch.randn(index_heads, hidden_size, requires_grad=True)
    k_norm_weight = torch.randn(index_dim)
    k_norm_bias = torch.randn(index_dim)

    q_index, routing_weights = _project_q_index_tile(
        hidden_states,
        0,
        sequence_length,
        linear_q_weight,
        linear_weights_weight,
        index_heads,
        index_dim,
        0,
        None,
        False,
        False,
        False,
        norm_spec,
    )
    k_index = _project_k_index_block(
        hidden_states,
        0,
        sequence_length,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        True,
        1.0e-5,
        index_dim,
        0,
        None,
        False,
        False,
        False,
        norm_spec,
    )

    expected_q = F.linear(normalized_hidden, linear_q_weight).reshape(
        sequence_length, batch_size, index_heads, index_dim
    )
    expected_k = F.layer_norm(
        F.linear(normalized_hidden, linear_k_weight),
        (index_dim,),
        k_norm_weight,
        k_norm_bias,
        1.0e-5,
    )
    expected_weights = F.linear(normalized_hidden, linear_weights_weight)
    expected_weights = expected_weights * (index_heads**-0.5) * (index_dim**-0.5)

    torch.testing.assert_close(q_index, expected_q)
    torch.testing.assert_close(k_index, expected_k)
    torch.testing.assert_close(routing_weights, expected_weights)
    (q_index.sum() + k_index.sum() + routing_weights.sum()).backward()
    assert norm_weight.grad is None
    assert norm_bias is None or norm_bias.grad is None


@pytest.mark.parametrize("enabled", [False, True])
def test_attention_wrapper_supplies_fused_norm_to_standard_indexer_only_when_enabled(enabled):
    hidden_states = torch.randn(3, 1, 8)
    linear_qkv = SimpleNamespace(
        layer_norm_weight=torch.randn(8),
        layer_norm_bias=None,
        eps=1.0e-5,
    )
    attention = SimpleNamespace(
        config=SimpleNamespace(
            experimental_attention_variant="dsa",
            dsa_indexer_mode="standard",
            dsa_standard_indexer_use_main_input_norm=enabled,
            dsa_fwd_skip_dsa=False,
            normalization="RMSNorm",
            layernorm_epsilon=1.0e-5,
            layernorm_zero_centered_gamma=False,
        ),
        linear_qkv=linear_qkv,
        _use_indexer_rope=lambda *args: False,
    )

    kwargs = DSGroupedSelfAttention._get_core_attention_extra_kwargs(
        attention,
        hidden_states,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        None,
        None,
        None,
        None,
        None,
        None,
        AttnMaskType.causal,
        None,
    )

    assert (kwargs["indexer_input_norm"] is not None) is enabled


@pytest.mark.parametrize("disable_main_input_norm", [False, True])
def test_attention_wrapper_honors_simplified_main_input_norm_disable(
    disable_main_input_norm,
):
    hidden_states = torch.randn(3, 1, 8)
    attention = SimpleNamespace(
        config=SimpleNamespace(
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_simplified_indexer_disable_main_input_norm=disable_main_input_norm,
            dsa_fwd_skip_dsa=False,
            normalization="RMSNorm",
            layernorm_epsilon=1.0e-5,
            layernorm_zero_centered_gamma=False,
        ),
        linear_qkv=SimpleNamespace(
            layer_norm_weight=torch.randn(8),
            layer_norm_bias=None,
            eps=1.0e-5,
        ),
        _use_indexer_rope=lambda *args: False,
    )

    kwargs = DSGroupedSelfAttention._get_core_attention_extra_kwargs(
        attention,
        hidden_states,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        None,
        None,
        None,
        None,
        None,
        None,
        AttnMaskType.causal,
        None,
    )

    assert (kwargs["indexer_input_norm"] is None) is disable_main_input_norm


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
    indexer.linear_k = (
        torch.nn.Linear(hidden_size, head_dim, bias=False) if learned_k else None
    )
    return indexer


def test_transformer_config_accepts_simplified_dsa_and_derives_shape():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_kernel_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )

    assert config.dsa_indexer_n_heads == 1
    assert config.dsa_indexer_head_dim == 8
    assert not config.dsa_indexer_use_hadamard


def test_transformer_config_accepts_simplified_learned_k_with_independent_dimension():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=True,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=4,
        dsa_kernel_backend="torch-min-memory",
        dsa_kernel_cache_indexer_k=True,
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )

    assert config.dsa_indexer_n_heads == 1
    assert config.dsa_indexer_head_dim == 6
    assert config.dsa_simplified_use_learned_k
    assert config.dsa_kernel_cache_indexer_k


def test_simplified_main_q_reset_requires_main_attention_dimension_with_learned_k():
    with pytest.raises(AssertionError, match="Main-Q initialization"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=True,
            dsa_indexer_head_dim=6,
            dsa_indexer_topk=4,
            dsa_indexer_reset_method="main-q-mean-rescaled",
            dsa_reset_indexer_on_load=True,
            dsa_kernel_backend="torch-min-memory",
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
        )


def test_transformer_config_rejects_simplified_mode_without_dsa_variant():
    with pytest.raises(AssertionError, match="requires experimental_attention_variant='dsa'"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            dsa_indexer_mode="simplified",
        )


def test_simplified_learned_k_is_model_defining_checkpoint_metadata(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=False,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(**common, dsa_simplified_use_learned_k=True)
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(**common, dsa_simplified_use_learned_k=True)
    )
    with pytest.raises(AssertionError, match="dsa_simplified_use_learned_k"):
        checkpointing.check_checkpoint_args(
            SimpleNamespace(**common, dsa_simplified_use_learned_k=False)
        )

    runtime_args.dsa_simplified_use_learned_k = False
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    load_args = SimpleNamespace(
        load="old-dsa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_simplified_use_learned_k=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    old_checkpoint_args = SimpleNamespace(
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": old_checkpoint_args, "iteration": 17},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(load_args)
    assert loaded_args.dsa_simplified_use_learned_k is False


def test_simplified_disabled_main_input_norm_is_model_defining_checkpoint_metadata(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=False,
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=False,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(
        **common, dsa_simplified_indexer_disable_main_input_norm=True
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(
            **common, dsa_simplified_indexer_disable_main_input_norm=True
        )
    )
    with pytest.raises(
        AssertionError, match="dsa_simplified_indexer_disable_main_input_norm"
    ):
        checkpointing.check_checkpoint_args(
            SimpleNamespace(
                **common, dsa_simplified_indexer_disable_main_input_norm=False
            )
        )

    runtime_args.dsa_simplified_indexer_disable_main_input_norm = False
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    load_args = SimpleNamespace(
        load="old-dsa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_simplified_indexer_disable_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    old_checkpoint_args = SimpleNamespace(
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": old_checkpoint_args, "iteration": 17},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(load_args)
    assert not loaded_args.dsa_simplified_indexer_disable_main_input_norm

    conversion_args = SimpleNamespace(
        load="gqa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_simplified_indexer_disable_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    gqa_checkpoint_args = SimpleNamespace(experimental_attention_variant=None)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": gqa_checkpoint_args, "iteration": 19},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(conversion_args)
    assert loaded_args.dsa_simplified_indexer_disable_main_input_norm


def test_standard_main_input_norm_is_model_defining_checkpoint_metadata(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="standard",
        dsa_simplified_use_learned_k=False,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=True,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(
        **common, dsa_standard_indexer_use_main_input_norm=True
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(**common, dsa_standard_indexer_use_main_input_norm=True)
    )
    with pytest.raises(
        AssertionError, match="dsa_standard_indexer_use_main_input_norm"
    ):
        checkpointing.check_checkpoint_args(
            SimpleNamespace(**common, dsa_standard_indexer_use_main_input_norm=False)
        )

    runtime_args.dsa_standard_indexer_use_main_input_norm = False
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    load_args = SimpleNamespace(
        load="old-dsa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_standard_indexer_use_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    old_checkpoint_args = SimpleNamespace(
        experimental_attention_variant="dsa",
        dsa_indexer_mode="standard",
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": old_checkpoint_args, "iteration": 17},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(load_args)
    assert loaded_args.dsa_standard_indexer_use_main_input_norm is False

    conversion_args = SimpleNamespace(
        load="gqa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_standard_indexer_use_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    gqa_checkpoint_args = SimpleNamespace(experimental_attention_variant=None)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": gqa_checkpoint_args, "iteration": 19},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(conversion_args)
    assert loaded_args.dsa_standard_indexer_use_main_input_norm is True


def test_dsa_trainability_mode_requires_no_load_optim_for_transitions(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="standard",
        dsa_simplified_use_learned_k=False,
        dsa_simplified_indexer_disable_main_input_norm=False,
        dsa_standard_indexer_use_main_input_norm=False,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=True,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(
        **common,
        dsa_train_main_only=True,
        dsa_train_indexer_only=False,
        no_load_optim=False,
        finetune=False,
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(
            **common,
            dsa_train_main_only=True,
            dsa_train_indexer_only=False,
        )
    )
    with pytest.raises(AssertionError, match="Use --no-load-optim"):
        checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    runtime_args.no_load_optim = True
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))


def test_transformer_config_rejects_reset_while_dsa_is_still_skipped():
    with pytest.raises(AssertionError, match="disabled when resetting"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_indexer_topk=4,
            dsa_fwd_skip_dsa=True,
            dsa_reset_indexer_on_load=True,
        )


def test_simplified_indexer_accepts_internal_tp_group_rewrite(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa_gqa as dsa_gqa

    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        rotary_percent=0.0,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_kernel_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    # Attention performs this rewrite internally when the global KV-group count is below TP.
    config.num_query_groups = 2

    class _TP2Group:
        def size(self):
            return 2

    pg_collection = SimpleNamespace(tp=_TP2Group(), cp=None)

    def _build_linear(_spec, input_size, output_size, **_kwargs):
        return torch.nn.Linear(input_size, output_size, bias=False)

    monkeypatch.setattr(dsa_gqa, "build_module", _build_linear)
    indexer = SimplifiedDSGQAIndexer(
        config,
        SimplifiedDSGQAIndexerSubmodules(linear_q=torch.nn.Linear),
        pg_collection=pg_collection,
    )

    assert indexer.linear_q.weight.shape == (8, 32)
    assert indexer.linear_k is None
    assert set(indexer.state_dict()) == {"linear_q.weight"}
    assert getattr(indexer.linear_q.weight, "average_gradients_across_tp_domain")


def test_simplified_learned_k_builds_replicated_independent_projection(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa_gqa as dsa_gqa

    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        rotary_percent=0.0,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=True,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=4,
        dsa_kernel_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    config.num_query_groups = 2

    class _TP2Group:
        def size(self):
            return 2

    def _build_linear(_spec, input_size, output_size, **_kwargs):
        return torch.nn.Linear(input_size, output_size, bias=False)

    monkeypatch.setattr(dsa_gqa, "build_module", _build_linear)
    indexer = SimplifiedDSGQAIndexer(
        config,
        SimplifiedDSGQAIndexerSubmodules(
            linear_q=torch.nn.Linear,
            linear_k=torch.nn.Linear,
        ),
        pg_collection=SimpleNamespace(tp=_TP2Group(), cp=None),
    )

    assert indexer.linear_q.weight.shape == (6, 32)
    assert indexer.linear_k.weight.shape == (6, 32)
    assert set(indexer.state_dict()) == {"linear_q.weight", "linear_k.weight"}
    assert getattr(indexer.linear_q.weight, "average_gradients_across_tp_domain")
    assert getattr(indexer.linear_k.weight, "average_gradients_across_tp_domain")


def test_simplified_indexer_rope_matches_model_rotary_config(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa_gqa as dsa_gqa

    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        rotary_percent=1.0,
        rotary_interleaved=True,
        rotary_seq_len_interpolation_factor=2.0,
        use_rope_scaling=True,
        rope_scaling_factor=4.0,
        use_cpu_initialization=True,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_kernel_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )

    def _build_linear(_spec, input_size, output_size, **_kwargs):
        return torch.nn.Linear(input_size, output_size, bias=False)

    monkeypatch.setattr(dsa_gqa, "build_module", _build_linear)
    indexer = SimplifiedDSGQAIndexer(
        config,
        SimplifiedDSGQAIndexerSubmodules(linear_q=torch.nn.Linear),
        pg_collection=SimpleNamespace(tp=_DummyTPGroup(), cp=None),
    )

    assert indexer.rotary_pos_emb.rotary_interleaved
    assert indexer.rotary_pos_emb.seq_len_interpolation_factor == 2.0
    assert indexer.rotary_pos_emb.inv_freq.device.type == "cpu"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_query_groups": 2}, "num_query_groups == 1"),
        ({"dsa_indexer_n_heads": 2}, "one indexer Q head"),
        ({"dsa_indexer_head_dim": 4}, "main attention head dimension"),
        ({"dsa_indexer_use_hadamard": True}, "does not support Hadamard"),
        ({"dsa_kernel_cache_indexer_k": True}, "no separate indexer K cache"),
    ],
)
def test_transformer_config_rejects_incompatible_simplified_dsa_options(kwargs, message):
    config_kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_kernel_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    config_kwargs.update(kwargs)
    with pytest.raises(AssertionError, match=message):
        TransformerConfig(**config_kwargs)


def test_simplified_dense_loss_matches_reference_and_only_grads_indexer_q():
    torch.manual_seed(123)
    seqlen, batch_size, hidden_size = 7, 2, 12
    num_query_heads, head_dim = 4, 3
    score_scale = 0.37
    loss_coeff = 0.4
    query = torch.randn(seqlen, batch_size, num_query_heads, head_dim)
    key = torch.randn(seqlen, batch_size, 1, head_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size, requires_grad=True)
    indexer = _simplified_test_indexer(hidden_size, head_dim, topk=3)
    linear_qkv = SimpleNamespace(
        layer_norm_weight=torch.randn(hidden_size),
        layer_norm_bias=None,
        eps=1.0e-5,
        skip_norm_and_all_gather=False,
    )
    norm_config = SimpleNamespace(
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        layernorm_zero_centered_gamma=False,
    )
    input_norm = _simplified_indexer_norm_spec(linear_qkv, norm_config)
    normalized_hidden = _simplified_indexer_input(hidden_states, input_norm)

    q_index = indexer.linear_q(normalized_hidden).reshape(
        seqlen, batch_size, 1, head_dim
    )
    index_scores = _simplified_index_scores(q_index, key.detach(), indexer.softmax_scale)
    index_scores = index_scores + _causal_mask(seqlen, query.device)
    topk_indices = index_scores.topk(indexer.index_topk, dim=-1).indices
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key.detach(),
        score_scale,
        loss_coeff,
        False,
        indexer.pg_collection,
    )
    reference_grad = torch.autograd.grad(reference_loss, indexer.linear_q.weight)[0]

    dense_loss = dsa_dense_indexer_loss(
        query.detach(),
        key.detach(),
        hidden_states.detach(),
        indexer,
        score_scale,
        loss_coeff,
        False,
        query_chunk_size=3,
        key_chunk_size=4,
        use_triton=False,
        simplified_input_norm=input_norm,
    )
    dense_grad, key_grad, hidden_grad = torch.autograd.grad(
        dense_loss,
        (indexer.linear_q.weight, key, hidden_states),
        allow_unused=True,
    )

    torch.testing.assert_close(dense_loss, reference_loss)
    torch.testing.assert_close(dense_grad, reference_grad, atol=2e-6, rtol=2e-5)
    assert key_grad is None
    assert hidden_grad is None


def test_simplified_sparse_min_memory_matches_reference_forward_loss_and_grads():
    torch.manual_seed(456)
    seqlen, batch_size, hidden_size = 8, 2, 12
    num_query_heads, head_dim, topk = 4, 3, 4
    score_scale = 0.37
    loss_coeff = 0.3
    query = torch.randn(
        seqlen, batch_size, num_query_heads, head_dim, requires_grad=True
    )
    key = torch.randn(seqlen, batch_size, 1, head_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, head_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = _simplified_test_indexer(hidden_size, head_dim, topk)
    linear_qkv = SimpleNamespace(
        layer_norm_weight=torch.randn(hidden_size),
        layer_norm_bias=None,
        eps=1.0e-5,
        skip_norm_and_all_gather=False,
    )
    norm_config = SimpleNamespace(
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        layernorm_zero_centered_gamma=False,
    )
    input_norm = _simplified_indexer_norm_spec(linear_qkv, norm_config)
    normalized_hidden = _simplified_indexer_input(hidden_states, input_norm)

    q_index = indexer.linear_q(normalized_hidden).reshape(seqlen, batch_size, 1, head_dim)
    index_scores = _simplified_index_scores(q_index, key.detach(), indexer.softmax_scale)
    index_scores = index_scores + _causal_mask(seqlen, query.device)
    topk_indices = index_scores.topk(topk, dim=-1).indices
    reference_output = unfused_grouped_dsa_fn(
        query, key, value, topk_indices, score_scale, use_gather=True
    )
    reference_loss = compute_gqa_dsa_indexer_loss(
        None,
        topk_indices,
        query.detach(),
        key.detach(),
        score_scale,
        loss_coeff,
        True,
        indexer.pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, topk_indices),
    )
    reference_grads = torch.autograd.grad(
        reference_output.float().sum() + reference_loss,
        (query, key, value, indexer.linear_q.weight),
    )

    min_query = query.detach().clone().requires_grad_(True)
    min_key = key.detach().clone().requires_grad_(True)
    min_value = value.detach().clone().requires_grad_(True)
    min_output, min_loss = dsa_min_memory_gqa(
        min_query,
        min_key,
        min_value,
        hidden_states.detach(),
        indexer,
        score_scale,
        loss_coeff,
        False,
        query_chunk_size=seqlen,
        key_chunk_size=seqlen,
        use_triton=False,
        simplified_input_norm=input_norm,
    )
    min_grads = torch.autograd.grad(
        min_output.float().sum() + min_loss,
        (min_query, min_key, min_value, indexer.linear_q.weight),
    )

    torch.testing.assert_close(min_output, reference_output)
    torch.testing.assert_close(min_loss, reference_loss)
    for actual, expected in zip(min_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_simplified_learned_k_dense_loss_matches_reference_and_is_detached():
    torch.manual_seed(789)
    seqlen, batch_size, hidden_size = 7, 2, 12
    num_query_heads, attention_dim, index_dim = 4, 3, 5
    attention_scale = attention_dim**-0.5
    loss_coeff = 0.4
    query = torch.randn(seqlen, batch_size, num_query_heads, attention_dim)
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size, requires_grad=True)
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk=3, learned_k=True)

    detached_hidden = hidden_states.detach()
    q_index = indexer.linear_q(detached_hidden).reshape(seqlen, batch_size, 1, index_dim)
    k_index = indexer.linear_k(detached_hidden).reshape(seqlen, batch_size, 1, index_dim)
    index_scores = _simplified_index_scores(q_index, k_index, indexer.softmax_scale)
    index_scores = index_scores + _causal_mask(seqlen, query.device)
    topk_indices = index_scores.topk(indexer.index_topk, dim=-1).indices
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query.detach(),
        key.detach(),
        attention_scale,
        loss_coeff,
        False,
        indexer.pg_collection,
    )
    reference_grads = torch.autograd.grad(
        reference_loss, (indexer.linear_q.weight, indexer.linear_k.weight)
    )

    dense_loss = dsa_dense_indexer_loss(
        query.detach(),
        key.detach(),
        hidden_states.detach(),
        indexer,
        attention_scale,
        loss_coeff,
        False,
        query_chunk_size=3,
        key_chunk_size=4,
        use_triton=False,
    )
    dense_grads = torch.autograd.grad(
        dense_loss,
        (indexer.linear_q.weight, indexer.linear_k.weight, key, hidden_states),
        allow_unused=True,
    )

    torch.testing.assert_close(dense_loss, reference_loss)
    torch.testing.assert_close(dense_grads[0], reference_grads[0], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(dense_grads[1], reference_grads[1], atol=3e-6, rtol=3e-5)
    assert dense_grads[2] is None
    assert dense_grads[3] is None


def test_simplified_learned_k_sparse_min_memory_matches_reference():
    torch.manual_seed(987)
    seqlen, batch_size, hidden_size = 8, 2, 12
    num_query_heads, attention_dim, index_dim, topk = 4, 3, 5, 4
    attention_scale = attention_dim**-0.5
    loss_coeff = 0.3
    query = torch.randn(
        seqlen, batch_size, num_query_heads, attention_dim, requires_grad=True
    )
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    hidden_states = torch.randn(
        seqlen, batch_size, hidden_size, requires_grad=True
    )
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk, learned_k=True)

    q_index = indexer.linear_q(hidden_states.detach()).reshape(
        seqlen, batch_size, 1, index_dim
    )
    k_index = indexer.linear_k(hidden_states.detach()).reshape(
        seqlen, batch_size, 1, index_dim
    )
    index_scores = _simplified_index_scores(q_index, k_index, indexer.softmax_scale)
    index_scores = index_scores + _causal_mask(seqlen, query.device)
    topk_indices = index_scores.topk(topk, dim=-1).indices.sort(dim=-1).values
    reference_output = unfused_grouped_dsa_fn(
        query, key, value, topk_indices, attention_scale, use_gather=True
    )
    reference_loss = compute_gqa_dsa_indexer_loss(
        None,
        topk_indices,
        query.detach(),
        key.detach(),
        attention_scale,
        loss_coeff,
        True,
        indexer.pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, topk_indices),
    )
    reference_grads = torch.autograd.grad(
        reference_output.float().sum() + reference_loss,
        (
            query,
            key,
            value,
            indexer.linear_q.weight,
            indexer.linear_k.weight,
        ),
    )

    min_query = query.detach().clone().requires_grad_(True)
    min_key = key.detach().clone().requires_grad_(True)
    min_value = value.detach().clone().requires_grad_(True)
    min_output, min_loss = dsa_min_memory_gqa(
        min_query,
        min_key,
        min_value,
        hidden_states,
        indexer,
        attention_scale,
        loss_coeff,
        False,
        query_chunk_size=seqlen,
        key_chunk_size=seqlen,
        use_triton=False,
    )
    min_grads = torch.autograd.grad(
        min_output.float().sum() + min_loss,
        (
            min_query,
            min_key,
            min_value,
            indexer.linear_q.weight,
            indexer.linear_k.weight,
            hidden_states,
        ),
        allow_unused=True,
    )

    torch.testing.assert_close(min_output, reference_output)
    torch.testing.assert_close(min_loss, reference_loss)
    for actual, expected in zip(min_grads[:5], reference_grads):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    assert min_grads[5] is None

    cached_query = query.detach().clone().requires_grad_(True)
    cached_key = key.detach().clone().requires_grad_(True)
    cached_value = value.detach().clone().requires_grad_(True)
    cached_output, cached_loss = dsa_min_memory_gqa(
        cached_query,
        cached_key,
        cached_value,
        hidden_states,
        indexer,
        attention_scale,
        loss_coeff,
        False,
        query_chunk_size=seqlen,
        key_chunk_size=seqlen,
        cache_routing=True,
        cache_indexer_k=True,
        cache_selected_scores=True,
        use_triton=False,
    )
    cached_grads = torch.autograd.grad(
        cached_output.float().sum() + cached_loss,
        (
            cached_query,
            cached_key,
            cached_value,
            indexer.linear_q.weight,
            indexer.linear_k.weight,
            hidden_states,
        ),
        allow_unused=True,
    )
    torch.testing.assert_close(cached_output, min_output)
    torch.testing.assert_close(cached_loss, min_loss)
    for actual, expected in zip(cached_grads[:5], min_grads[:5]):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    assert cached_grads[5] is None


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
def test_simplified_train_main_only_zero_loss_produces_no_indexer_update(
    learned_k, freeze_indexer
):
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
    grads = torch.autograd.grad(
        output.float().sum() + indexer_loss,
        grad_inputs,
    )

    torch.testing.assert_close(indexer_loss, torch.zeros_like(indexer_loss))
    assert any(torch.count_nonzero(grad) for grad in grads[:3])
    if freeze_indexer:
        assert all(not weight.requires_grad for weight in indexer_weights)
    else:
        for grad in grads[3:]:
            torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize(
    "cache_routing,cache_indexer_k",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_standard_train_main_only_zero_loss_backpropagates_only_attention(
    cache_routing, cache_indexer_k
):
    torch.manual_seed(655)
    seqlen, batch_size, hidden_size = 6, 1, 8
    num_query_heads, num_query_groups, head_dim = 4, 2, 2
    index_heads, index_dim, topk = 2, 4, 3
    query = torch.randn(seqlen, batch_size, num_query_heads, head_dim, requires_grad=True)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    indexer = SimpleNamespace(
        index_n_heads=index_heads,
        index_head_dim=index_dim,
        index_topk=topk,
        index_rotary_dim=0,
        rotary_pos_emb=None,
        pg_collection=_DummyPGCollection(),
        config=SimpleNamespace(
            dsa_indexer_mode="standard",
            dsa_indexer_use_hadamard=False,
            layernorm_epsilon=1.0e-5,
            rotary_interleaved=False,
        ),
    )
    indexer.linear_q = torch.nn.Linear(
        hidden_size, index_heads * index_dim, bias=False
    )
    indexer.linear_k = torch.nn.Linear(hidden_size, index_dim, bias=False)
    indexer.k_norm = torch.nn.LayerNorm(index_dim, eps=1.0e-5)
    indexer.linear_weights_proj = torch.nn.Linear(
        hidden_size, index_heads, bias=False
    )
    indexer_modules = (
        indexer.linear_q,
        indexer.linear_k,
        indexer.k_norm,
        indexer.linear_weights_proj,
    )
    for module in indexer_modules:
        module.requires_grad_(False)

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
        cache_routing=cache_routing,
        cache_indexer_k=cache_indexer_k,
        use_triton=False,
    )
    grads = torch.autograd.grad(
        output.float().sum() + indexer_loss,
        (query, key, value),
    )

    torch.testing.assert_close(indexer_loss, torch.zeros_like(indexer_loss))
    assert all(torch.count_nonzero(grad) for grad in grads)
    assert all(
        param.grad is None
        for module in indexer_modules
        for param in module.parameters()
    )


def test_simplified_main_q_mean_reset_uses_all_query_heads():
    from megatron.training.training import _reset_simplified_dsa_indexers_from_main_q

    hidden_size, num_query_heads, head_dim = 5, 4, 3

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
            )
            self.pg_collection = SimpleNamespace(tp=None)

    class _FakeCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = _FakeIndexer()

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = torch.nn.Linear(
                hidden_size, (num_query_heads + 2) * head_dim, bias=False
            )
            self.core_attention = _FakeCore()

    attention = _FakeAttention()
    with torch.no_grad():
        attention.linear_qkv.weight.copy_(
            torch.arange(attention.linear_qkv.weight.numel(), dtype=torch.float32).reshape_as(
                attention.linear_qkv.weight
            )
        )
    expected = attention.linear_qkv.weight[: num_query_heads * head_dim].reshape(
        num_query_heads, head_dim, hidden_size
    ).mean(dim=0)

    assert _reset_simplified_dsa_indexers_from_main_q([attention]) == 1
    torch.testing.assert_close(attention.core_attention.indexer.linear_q.weight, expected)


@pytest.mark.parametrize("attention_output_gate", [False, True])
def test_simplified_main_q_reset_also_initializes_learned_k_from_main_k(
    attention_output_gate,
):
    from megatron.training.training import _reset_simplified_dsa_indexers_from_main_q

    hidden_size, num_query_heads, head_dim = 5, 4, 3

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
                attention_output_gate=attention_output_gate,
            )
            self.pg_collection = SimpleNamespace(tp=None)

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = torch.nn.Linear(
                hidden_size,
                ((2 * num_query_heads if attention_output_gate else num_query_heads) + 2)
                * head_dim,
                bias=False,
            )
            self.core_attention = torch.nn.Module()
            self.core_attention.indexer = _FakeIndexer()

    attention = _FakeAttention()
    with torch.no_grad():
        attention.linear_qkv.weight.copy_(
            torch.arange(attention.linear_qkv.weight.numel(), dtype=torch.float32).reshape_as(
                attention.linear_qkv.weight
            )
        )
    expected_q = attention.linear_qkv.weight[: num_query_heads * head_dim].reshape(
        num_query_heads, head_dim, hidden_size
    ).mean(dim=0)
    k_start = num_query_heads * head_dim * (2 if attention_output_gate else 1)
    expected_k = attention.linear_qkv.weight[k_start : k_start + head_dim]

    assert _reset_simplified_dsa_indexers_from_main_q([attention]) == 1
    torch.testing.assert_close(attention.core_attention.indexer.linear_q.weight, expected_q)
    torch.testing.assert_close(attention.core_attention.indexer.linear_k.weight, expected_k)


def test_simplified_main_q_mean_rescaled_reset_gathers_fused_qkv_across_tp(monkeypatch):
    import megatron.training.training as training

    hidden_size, num_query_heads, head_dim, tp_size = 5, 4, 3, 2
    full_rows = (num_query_heads + 2) * head_dim
    full_weight = torch.arange(full_rows * hidden_size, dtype=torch.float32).reshape(
        full_rows, hidden_size
    )
    shards = list(full_weight.chunk(tp_size, dim=0))

    class _WeightModule(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone())

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
                attention_output_gate=False,
            )
            self.pg_collection = SimpleNamespace(tp=object())

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = _WeightModule(shards[0])
            self.core_attention = torch.nn.Module()
            self.core_attention.indexer = _FakeIndexer()

    def _fake_all_gather(outputs, local_weight, group):
        assert torch.equal(local_weight, shards[0])
        for output, shard in zip(outputs, shards):
            output.copy_(shard)

    monkeypatch.setattr(training, "get_pg_size", lambda group: tp_size)
    monkeypatch.setattr(torch.distributed, "all_gather", _fake_all_gather)

    attention = _FakeAttention()
    main_q_heads = full_weight[: num_query_heads * head_dim].reshape(
        num_query_heads, head_dim, hidden_size
    )
    expected = main_q_heads.mean(dim=0)
    expected = expected * torch.sqrt(
        main_q_heads.square().sum(dim=(1, 2)).mean() / expected.square().sum()
    )
    expected_k = full_weight[
        num_query_heads * head_dim : (num_query_heads + 1) * head_dim
    ]
    assert (
        training._reset_simplified_dsa_indexers_from_main_q([attention], rescale=True) == 1
    )
    torch.testing.assert_close(attention.core_attention.indexer.linear_q.weight, expected)
    torch.testing.assert_close(attention.core_attention.indexer.linear_k.weight, expected_k)


@pytest.mark.parametrize("no_load_optim", [False, True])
@pytest.mark.parametrize("reset_method", ["main-q-mean", "main-q-mean-rescaled"])
def test_simplified_main_q_reset_handles_optimizer_load_modes(
    monkeypatch, no_load_optim, reset_method
):
    import megatron.training.training as training

    hidden_size, num_query_heads, head_dim = 5, 4, 3

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
                attention_output_gate=False,
            )
            self.pg_collection = SimpleNamespace(tp=None)

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = torch.nn.Linear(
                hidden_size, (num_query_heads + 2) * head_dim, bias=False
            )
            self.core_attention = torch.nn.Module()
            self.core_attention.indexer = _FakeIndexer()

    class _FakeOptimizer:
        is_stub_optimizer = False

        def __init__(self):
            self.reload_count = 0

        def reload_model_params(self):
            self.reload_count += 1

    attention = _FakeAttention()
    optimizer = _FakeOptimizer()
    clear_calls = []
    group_step_reset_calls = []
    refresh_calls = []
    monkeypatch.setattr(
        training,
        "_clear_dsa_indexer_optimizer_state",
        lambda model, optimizer: clear_calls.append((model, optimizer)) or 1,
    )
    monkeypatch.setattr(
        training,
        "_apply_dsa_indexer_lr_warmup",
        lambda args, optimizer, scheduler: 0.0,
    )
    monkeypatch.setattr(
        training,
        "_reload_dsa_indexer_optimizer_params",
        lambda model, optimizer: refresh_calls.append((model, optimizer)) or 1,
    )
    monkeypatch.setattr(
        training,
        "_reset_dsa_indexer_optimizer_group_steps",
        lambda optimizer: group_step_reset_calls.append(optimizer) or 1,
    )
    monkeypatch.setattr(training, "_broadcast_dsa_indexer_params", lambda model: None)
    args = SimpleNamespace(
        dsa_indexer_reset_seed=None,
        dsa_indexer_reset_method=reset_method,
        no_load_optim=no_load_optim,
        finetune=False,
        dsa_indexer_activation_start_samples=0,
        consumed_train_samples=0,
    )

    training._reset_dsa_indexer_after_load(
        [attention], optimizer, None, args, explicit_start=True
    )

    expected_k = attention.linear_qkv.weight[
        num_query_heads * head_dim : (num_query_heads + 1) * head_dim
    ]
    torch.testing.assert_close(attention.core_attention.indexer.linear_k.weight, expected_k)
    assert optimizer.reload_count == 0
    assert len(refresh_calls) == 1
    assert len(clear_calls) == (0 if no_load_optim else 1)
    assert len(group_step_reset_calls) == (0 if no_load_optim else 1)


def test_dsa_reset_on_load_allows_pipeline_stage_without_local_indexer(monkeypatch):
    import megatron.training.training as training

    monkeypatch.setattr(
        training, "_reset_simplified_dsa_indexers_from_main_q", lambda model, rescale: 0
    )
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda local_count: 4)
    monkeypatch.setattr(training, "_broadcast_dsa_indexer_params", lambda model: None)
    monkeypatch.setattr(
        training, "_apply_dsa_indexer_lr_warmup", lambda args, optimizer, scheduler: None
    )
    args = SimpleNamespace(
        dsa_indexer_reset_method="main-q-mean",
        no_load_optim=True,
        dsa_indexer_activation_start_samples=0,
        consumed_train_samples=0,
    )

    training._reset_dsa_indexer_after_load(
        [], None, None, args, explicit_start=True
    )


@pytest.mark.parametrize("fsdp_arg", ["use_torch_fsdp2", "use_megatron_fsdp"])
def test_dsa_reset_on_load_rejects_fsdp(fsdp_arg):
    import megatron.training.training as training

    args = SimpleNamespace(use_torch_fsdp2=False, use_megatron_fsdp=False)
    setattr(args, fsdp_arg, True)

    with pytest.raises(RuntimeError, match="DDP/distributed-optimizer"):
        training._reset_dsa_indexer_after_load(
            [], None, None, args, explicit_start=True
        )


def test_dsa_reset_on_load_rejects_optimizer_cpu_offload(monkeypatch):
    import megatron.training.training as training

    class _FakeHybridDeviceOptimizer:
        pass

    monkeypatch.setattr(training, "HybridDeviceOptimizer", _FakeHybridDeviceOptimizer)
    optimizer = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=_FakeHybridDeviceOptimizer())
        ]
    )
    args = SimpleNamespace(use_torch_fsdp2=False, use_megatron_fsdp=False)

    with pytest.raises(RuntimeError, match="optimizer CPU offload"):
        training._reset_dsa_indexer_after_load(
            [], optimizer, None, args, explicit_start=True
        )


def test_dsa_train_indexer_only_allows_pipeline_stage_without_local_indexer(monkeypatch):
    import megatron.training.training as training

    model = torch.nn.Linear(3, 2)
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda local_count: 5)

    training._freeze_non_dsa_indexer_parameters([model])

    assert all(not param.requires_grad for param in model.parameters())


def test_dsa_train_indexer_only_freezes_exactly_indexer_submodule_parameters(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2)
            self.block = torch.nn.Module()
            self.block.indexer = torch.nn.Linear(3, 2)
            self.indexer_aux = torch.nn.Linear(3, 2)

    model = _Model()
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda count: count)

    training._freeze_non_dsa_indexer_parameters([model])

    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith("block.indexer.")


def test_dsa_train_main_only_allows_pipeline_stage_without_local_indexer(monkeypatch):
    import megatron.training.training as training

    model = torch.nn.Linear(3, 2)
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda local_count: 5)

    training._freeze_dsa_indexer_parameters([model])

    assert all(param.requires_grad for param in model.parameters())


def test_dsa_train_main_only_freezes_exactly_indexer_and_preserves_other_freezes(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2)
            self.block = torch.nn.Module()
            self.block.indexer = torch.nn.Linear(3, 2)
            self.indexer_aux = torch.nn.Linear(3, 2)
            self.backbone.bias.requires_grad_(False)

    model = _Model()
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda count: count)

    training._freeze_dsa_indexer_parameters([model])

    for name, param in model.named_parameters():
        if name.startswith("block.indexer.") or name == "backbone.bias":
            assert not param.requires_grad
        else:
            assert param.requires_grad


def test_dsa_indexer_optimizer_refresh_preserves_backbone_master_weights(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2, bias=False)
            self.indexer = torch.nn.Linear(3, 2, bias=False)

    model = _Model()
    backbone_master = torch.full_like(model.backbone.weight, 17.0, dtype=torch.float32)
    indexer_master = torch.full_like(model.indexer.weight, -5.0, dtype=torch.float32)
    with torch.no_grad():
        model.backbone.weight.fill_(1.0)
        model.indexer.weight.fill_(3.0)

    optimizer = SimpleNamespace(
        is_stub_optimizer=False,
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
    )
    monkeypatch.setattr(
        training,
        "get_model_to_optimizer_param_map",
        lambda optimizer: {
            model.backbone.weight: backbone_master,
            model.indexer.weight: indexer_master,
        },
    )

    assert training._reload_dsa_indexer_optimizer_params([model], optimizer) == 1
    torch.testing.assert_close(indexer_master, model.indexer.weight.float())
    torch.testing.assert_close(backbone_master, torch.full_like(backbone_master, 17.0))


def test_dsa_indexer_optimizer_group_step_reset_preserves_backbone_clock():
    import megatron.training.training as training

    backbone_group = {"params": [object()], "is_dsa_indexer": False, "step": 123}
    indexer_weight_group = {"params": [object()], "is_dsa_indexer": True, "step": 123}
    indexer_bias_step = torch.tensor(123.0)
    indexer_bias_group = {
        "params": [object()],
        "is_dsa_indexer": True,
        "step": indexer_bias_step,
    }
    empty_indexer_group = {"params": [], "is_dsa_indexer": True}
    optimizer = SimpleNamespace(
        is_stub_optimizer=False,
        optimizer=SimpleNamespace(
            param_groups=[
                backbone_group,
                indexer_weight_group,
                indexer_bias_group,
                empty_indexer_group,
            ]
        ),
    )

    assert training._reset_dsa_indexer_optimizer_group_steps(optimizer) == 3
    assert backbone_group["step"] == 123
    assert indexer_weight_group["step"] == 0
    assert indexer_bias_group["step"] is indexer_bias_step
    assert indexer_bias_group["step"].item() == 0.0
    assert empty_indexer_group["step"] == 0


def test_dsa_indexer_optimizer_refresh_copies_only_owned_distributed_shard(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = torch.nn.Linear(4, 2, bias=False)

    model = _Model()
    with torch.no_grad():
        model.indexer.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    indexer_shard = torch.full((3,), -1.0, dtype=torch.float32)

    class _Optimizer:
        is_stub_optimizer = False
        config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

        @staticmethod
        def _get_model_param_range_map(param):
            assert param is model.indexer.weight
            return {"param": SimpleNamespace(start=2, end=5)}

    optimizer = _Optimizer()
    monkeypatch.setattr(
        training,
        "get_model_to_optimizer_param_map",
        lambda optimizer: {model.indexer.weight: indexer_shard},
    )

    assert training._reload_dsa_indexer_optimizer_params([model], optimizer) == 1
    torch.testing.assert_close(indexer_shard, torch.tensor([2.0, 3.0, 4.0]))


def test_dsa_indexer_random_reset_is_deterministic_and_preserves_global_rng():
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(init_method=lambda tensor: torch.nn.init.normal_(tensor))
            self.backbone = torch.nn.Linear(4, 3)
            self.indexer = torch.nn.Sequential(
                torch.nn.Linear(4, 3),
                torch.nn.LayerNorm(3),
            )

    model_a = _Model()
    model_b = _Model()
    backbone_before = {name: tensor.detach().clone() for name, tensor in model_a.backbone.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()

    assert training._reset_dsa_indexer_modules([model_a], seed=9876) == 1
    rng_after = torch.random.get_rng_state()
    assert training._reset_dsa_indexer_modules([model_b], seed=9876) == 1

    assert torch.equal(rng_before, rng_after)
    for name, expected in backbone_before.items():
        torch.testing.assert_close(model_a.backbone.state_dict()[name], expected)
    for actual, expected in zip(model_a.indexer.parameters(), model_b.indexer.parameters()):
        torch.testing.assert_close(actual, expected)


def test_dsa_indexer_reset_seed_derivation_is_tp_invariant_and_dp_aware(monkeypatch):
    import megatron.training.training as training

    monkeypatch.setattr(training.mpu, "get_pipeline_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(training.mpu, "get_data_parallel_rank", lambda: 3)

    args = SimpleNamespace(
        dsa_indexer_reset_seed=None,
        seed=1234,
        data_parallel_random_init=False,
    )
    assert training._get_dsa_indexer_reset_seed(args) == 1434

    args.data_parallel_random_init = True
    assert training._get_dsa_indexer_reset_seed(args) == 1464

    args.dsa_indexer_reset_seed = 77
    assert training._get_dsa_indexer_reset_seed(args) == 77


def test_dsa_indexer_optimizer_state_clear_preserves_backbone_state(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2)
            self.indexer = torch.nn.Linear(3, 2)

    model = _Model()
    torch_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    for param in model.parameters():
        torch_optimizer.state[param] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.full_like(param, 2.0),
            "exp_avg_sq": torch.full_like(param, 3.0),
        }
    backbone_state = {
        param: {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in torch_optimizer.state[param].items()
        }
        for param in model.backbone.parameters()
    }
    optimizer = SimpleNamespace(is_stub_optimizer=False, optimizer=torch_optimizer)
    monkeypatch.setattr(
        training,
        "get_model_to_optimizer_param_map",
        lambda _optimizer: {param: param for param in model.parameters()},
    )

    assert training._clear_dsa_indexer_optimizer_state([model], optimizer) == 2
    assert all(param not in torch_optimizer.state for param in model.indexer.parameters())
    for param, expected_state in backbone_state.items():
        assert param in torch_optimizer.state
        for name, expected in expected_state.items():
            actual = torch_optimizer.state[param][name]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected


def test_dsa_indexer_reset_broadcasts_only_indexer_params_across_dp(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2, bias=False)
            self.indexer = torch.nn.Linear(3, 2, bias=False)

    model = _Model()
    dp_group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(training.mpu, "get_data_parallel_group", lambda: dp_group)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: 11)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, src, group: calls.append((tensor, src, group)),
    )

    training._broadcast_dsa_indexer_params([model])

    assert len(calls) == 1
    assert calls[0][0].data_ptr() == model.indexer.weight.data_ptr()
    assert calls[0][1:] == (11, dp_group)


def test_transformer_config_min_memory_accepts_sparse_loss_without_topk_only_flag():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_kernel_backend="triton-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_use_hadamard=True,
    )

    assert config.dsa_indexer_use_sparse_loss
    assert not config.dsa_indexer_sparse_loss_use_topk_only




@pytest.mark.parametrize(
    "backend", ["reference", "torch-min-memory", "triton-min-memory"]
)
def test_transformer_config_accepts_dsa_train_main_only(backend):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_kernel_backend=backend,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_hadamard=True,
        dsa_train_main_only=True,
    )

    assert config.dsa_train_main_only
    assert not config.dsa_indexer_use_sparse_loss




@pytest.mark.parametrize(
    "override,match",
    [
        ({"dsa_indexer_loss_coeff": 0.1}, "leave dsa_indexer_loss_coeff"),
        ({"dsa_indexer_use_sparse_loss": True}, "dsa_indexer_use_sparse_loss"),
        ({"dsa_fwd_use_dense_attn": True}, "sparse DSA forward"),
        ({"dsa_fwd_skip_dsa": True}, "sparse DSA forward"),
        ({"dsa_train_indexer_only": True}, "incompatible"),
        ({"dsa_kernel_cache_selected_scores": True}, "selected-score"),
        ({"dsa_reset_indexer_on_load": True}, "incompatible"),
        ({"dsa_indexer_activation_start_samples": 100}, "activation_start_samples"),
        ({"dsa_indexer_activation_warmup_samples": 100}, "warmup_samples"),
        ({"dsa_indexer_topk_recompute": True}, "nondifferentiable frozen routing"),
    ],
)
def test_transformer_config_rejects_incompatible_dsa_train_main_only_modes(
    override, match
):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_kernel_backend="triton-min-memory",
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_hadamard=True,
        dsa_train_main_only=True,
    )
    kwargs.update(override)

    with pytest.raises(AssertionError, match=match):
        TransformerConfig(**kwargs)






def test_transformer_config_accepts_dense_warmup_min_memory_backend():
    for backend in ("triton-min-memory", "torch-min-memory"):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend=backend,
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_hadamard=True,
        )

        assert config.dsa_fwd_use_dense_attn
        assert not config.dsa_indexer_use_sparse_loss


def test_transformer_config_dense_warmup_rejects_sparse_loss_and_caches():
    with pytest.raises(AssertionError, match="dsa_indexer_use_sparse_loss"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="triton-min-memory",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
            dsa_indexer_use_hadamard=True,
        )

    with pytest.raises(AssertionError, match="dsa_kernel_cache_routing"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="triton-min-memory",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_hadamard=True,
            dsa_kernel_cache_routing=True,
        )


def test_transformer_config_dense_warmup_requires_min_memory_backend():
    with pytest.raises(AssertionError, match="dsa_fwd_use_dense_attn"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="reference",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_hadamard=True,
        )


def test_transformer_config_dense_warmup_requires_positive_loss_coeff_and_dsa_variant():
    with pytest.raises(AssertionError, match="dsa_indexer_loss_coeff"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="triton-min-memory",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_hadamard=True,
        )

    with pytest.raises(AssertionError, match="experimental_attention_variant='dsa'"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            dsa_fwd_use_dense_attn=True,
        )


def test_min_memory_backend_supports_no_grad_validation_forward(monkeypatch):
    torch.manual_seed(123)

    calls = []
    indexer_input_norm = SimpleNamespace(
        normalization="RMSNorm",
        weight=torch.randn(8),
        bias=None,
        eps=1.0e-5,
        zero_centered_gamma=False,
    )

    def _fake_forward_only(**kwargs):
        calls.append(kwargs)
        query = kwargs["query"]
        value = kwargs["value"]
        return query.new_empty(query.size(0), query.size(1), query.size(2) * value.size(-1))

    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.dsa_gqa."
        "dsa_min_memory_gqa_forward_only",
        _fake_forward_only,
    )

    for backend in ("torch-min-memory", "triton-min-memory"):
        core = SimpleNamespace(
            config=SimpleNamespace(
                dsa_kernel_backend=backend,
                dsa_sparse_attention_use_gather=False,
                dsa_indexer_use_sparse_loss=True,
                dsa_indexer_use_hadamard=True,
                fp8=None,
                fp8_param=False,
                layernorm_zero_centered_gamma=False,
                dsa_kernel_query_block_size=2,
                dsa_kernel_key_block_size=3,
                dsa_kernel_cache_indexer_k=True,
                dsa_min_memory_profile=False,
                dsa_min_memory_profile_rank=0,
            ),
            indexer=object(),
            softmax_scale=4**-0.5,
            training=False,
            layer_number=1,
        )

        query = torch.randn(4, 2, 4, 4)
        key = torch.randn(4, 2, 2, 4)
        value = torch.randn(4, 2, 2, 4)
        hidden_states = torch.randn(4, 2, 8)

        with torch.no_grad():
            output = DSGQACoreAttention._forward_min_memory(
                core,
                query,
                key,
                value,
                None,
                hidden_states,
                indexer_input_norm=indexer_input_norm,
                attn_mask_type=AttnMaskType.causal,
            )

        assert output.shape == (4, 2, 16)
        assert not output.requires_grad

    assert [call["use_triton"] for call in calls] == [False, True]
    assert all(call["simplified_input_norm"] is indexer_input_norm for call in calls)


@pytest.mark.parametrize("learned_k", [False, True])
def test_min_memory_simplified_no_norm_discards_supplied_norm(monkeypatch, learned_k):
    calls = []
    supplied_norm = SimpleNamespace(
        normalization="RMSNorm",
        weight=torch.randn(8),
        bias=None,
        eps=1.0e-5,
        zero_centered_gamma=False,
    )

    def _fake_forward_only(**kwargs):
        calls.append(kwargs)
        query = kwargs["query"]
        value = kwargs["value"]
        return query.new_empty(query.size(0), query.size(1), query.size(2) * value.size(-1))

    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.dsa_gqa."
        "dsa_min_memory_gqa_forward_only",
        _fake_forward_only,
    )
    core = SimpleNamespace(
        config=SimpleNamespace(
            dsa_kernel_backend="triton-min-memory",
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=learned_k,
            dsa_simplified_indexer_disable_main_input_norm=True,
            dsa_fwd_skip_dsa=False,
            dsa_fwd_use_dense_attn=False,
            dsa_indexer_use_sparse_loss=True,
            dsa_sparse_attention_use_gather=False,
            fp8=None,
            fp8_param=False,
            layernorm_zero_centered_gamma=False,
            dsa_kernel_query_block_size=2,
            dsa_kernel_key_block_size=3,
            dsa_kernel_cache_indexer_k=learned_k,
            dsa_min_memory_profile=False,
            dsa_min_memory_profile_rank=0,
        ),
        indexer=object(),
        softmax_scale=4**-0.5,
        training=False,
        layer_number=1,
    )
    query = torch.randn(4, 2, 4, 4)
    key = torch.randn(4, 2, 1, 4)
    value = torch.randn(4, 2, 1, 4)
    hidden_states = torch.randn(4, 2, 8)

    with torch.no_grad():
        output = DSGQACoreAttention._forward_min_memory(
            core,
            query,
            key,
            value,
            None,
            hidden_states,
            indexer_input_norm=supplied_norm,
            attn_mask_type=AttnMaskType.causal,
        )

    assert output.shape == (4, 2, 16)
    assert len(calls) == 1
    assert calls[0]["simplified_input_norm"] is None




def test_reference_train_main_only_routes_without_constructing_indexer_loss(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa_gqa as dsa_gqa

    class _Indexer:
        def forward_before_topk(self, hidden_states, **_kwargs):
            sq, batch_size, _ = hidden_states.shape
            q_index = hidden_states.new_zeros((sq, batch_size, 1, 4))
            k_index = hidden_states.new_zeros((sq, batch_size, 1, 4))
            weights = hidden_states.new_ones((sq, batch_size, 1))
            return q_index, k_index, weights

    topk_calls = []

    def _fake_topk(q_index, _k_index, _weights, topk, _mask):
        topk_calls.append((q_index.shape, topk))
        sq, batch_size = q_index.shape[:2]
        scores = q_index.new_zeros((batch_size, sq, sq), dtype=torch.float32)
        indices = torch.zeros((batch_size, sq, topk), dtype=torch.long)
        return scores, indices

    def _unexpected_indexer_loss(*_args, **_kwargs):
        raise AssertionError("main-only mode must not construct indexer KL")

    monkeypatch.setattr(dsa_gqa, "fused_qk_topk_naive", _fake_topk)
    monkeypatch.setattr(dsa_gqa, "compute_gqa_dsa_indexer_loss", _unexpected_indexer_loss)
    monkeypatch.setattr(
        dsa_gqa,
        "unfused_grouped_dsa_fn",
        lambda query, *_args, **_kwargs: query,
    )

    core = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=False,
            dsa_kernel_backend="reference",
            dsa_fwd_skip_dsa=False,
            dsa_indexer_mode="standard",
            dsa_sparse_attention_use_gather=False,
            dsa_standard_indexer_use_main_input_norm=False,
            dsa_train_main_only=True,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
            dsa_indexer_sparse_loss_use_topk_only=False,
            dsa_indexer_loss_recompute=False,
            dsa_indexer_topk_key_chunk_size=None,
            dsa_indexer_topk_recompute=False,
            dsa_sparse_attention_recompute=False,
            dsa_sparse_attention_query_chunk_size=None,
        ),
        indexer=_Indexer(),
        softmax_scale=0.5,
        training=True,
        layer_number=1,
    )
    core.indexer.index_topk = 2
    query = torch.randn(4, 1, 4, 4, requires_grad=True)
    key = torch.randn(4, 1, 2, 4, requires_grad=True)
    value = torch.randn(4, 1, 2, 4, requires_grad=True)
    hidden_states = torch.randn(4, 1, 8)

    output = DSGQACoreAttention.forward(
        core,
        query,
        key,
        value,
        None,
        hidden_states,
        attn_mask_type=AttnMaskType.causal,
    )

    assert output is query
    assert topk_calls == [(torch.Size([4, 1, 1, 4]), 2)]












def test_dense_warmup_no_grad_validation_uses_dense_core_attention():
    torch.manual_seed(123)
    calls = []

    class _DenseCore:
        def __call__(self, query, key, value, attention_mask, **kwargs):
            calls.append((query, key, value, attention_mask, kwargs))
            return query.new_empty(query.size(0), query.size(1), query.size(2) * value.size(-1))

    core = SimpleNamespace(
        config=SimpleNamespace(
            dsa_kernel_backend="triton-min-memory",
            dsa_fwd_use_dense_attn=True,
            dsa_sparse_attention_use_gather=False,
            dsa_indexer_use_sparse_loss=False,
            dsa_indexer_use_hadamard=True,
            fp8=None,
            fp8_param=False,
            layernorm_zero_centered_gamma=False,
            dsa_kernel_cache_routing=False,
            dsa_kernel_cache_indexer_k=False,
            dsa_kernel_cache_selected_scores=False,
        ),
        dense_core_attention=_DenseCore(),
        indexer=object(),
        softmax_scale=4**-0.5,
        training=False,
        layer_number=1,
    )

    query = torch.randn(4, 2, 4, 4)
    key = torch.randn(4, 2, 2, 4)
    value = torch.randn(4, 2, 2, 4)
    attention_mask = torch.empty(1)
    hidden_states = torch.randn(4, 2, 8)

    with torch.no_grad():
        output = DSGQACoreAttention._forward_min_memory(
            core,
            query,
            key,
            value,
            attention_mask,
            hidden_states,
            attn_mask_type=AttnMaskType.causal,
        )

    assert output.shape == (4, 2, 16)
    assert len(calls) == 1
    assert calls[0][3] is attention_mask
    assert calls[0][4]["attn_mask_type"] == AttnMaskType.causal


@pytest.mark.parametrize(
    "legacy_flag",
    [
        "dsa_sparse_attention_query_chunk_size",
        "dsa_indexer_loss_query_chunk_size",
        "dsa_indexer_topk_key_chunk_size",
    ],
)
def test_transformer_config_min_memory_rejects_legacy_chunk_flags(legacy_flag):
    with pytest.raises(AssertionError, match="min-memory"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="triton-min-memory",
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
            dsa_indexer_use_hadamard=True,
            **{legacy_flag: 2},
        )


@pytest.mark.parametrize(
    "legacy_flag",
    [
        "dsa_indexer_topk_recompute",
        "dsa_indexer_loss_recompute",
        "dsa_sparse_attention_recompute",
        "dsa_sparse_attention_use_gather",
    ],
)
def test_transformer_config_min_memory_rejects_legacy_backend_flags(legacy_flag):
    kwargs = {legacy_flag: True}
    if legacy_flag == "dsa_indexer_topk_recompute":
        kwargs["dsa_indexer_topk_key_chunk_size"] = 2
    with pytest.raises(AssertionError, match="min-memory"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="triton-min-memory",
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
            dsa_indexer_use_hadamard=True,
            **kwargs,
        )


def test_transformer_config_cache_routing_requires_min_memory_backend():
    with pytest.raises(AssertionError, match="dsa_kernel_cache_routing"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="reference",
            dsa_indexer_loss_coeff=0.1,
            dsa_kernel_cache_routing=True,
        )


@pytest.mark.parametrize(
    "cache_flag",
    ["dsa_kernel_cache_indexer_k", "dsa_kernel_cache_selected_scores"],
)
def test_transformer_config_optional_kernel_caches_require_min_memory_backend(cache_flag):
    with pytest.raises(AssertionError, match=cache_flag):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="reference",
            dsa_indexer_loss_coeff=0.1,
            **{cache_flag: True},
        )


def test_torch_min_memory_forces_full_key_routing_chunk():
    assert _routing_key_chunk_size(None, key_length=8192, use_triton=False) == 8192
    assert _routing_key_chunk_size(1024, key_length=8192, use_triton=False) == 8192
    assert _routing_key_chunk_size(None, key_length=8192, use_triton=True) == 1024
    assert _routing_key_chunk_size(2048, key_length=8192, use_triton=True) == 2048


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
        query_len,
        batch_size,
        index_heads,
        index_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_index = torch.randn(
        key_len, batch_size, index_head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(
        query_len, batch_size, index_heads, device=device, dtype=torch.bfloat16
    )
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
    torch.testing.assert_close(
        tri_scores, ref_scores_at_tri_indices, rtol=5e-3, atol=5e-3
    )
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
    assert torch.all(
        ref_scores_at_tri_indices.amin(dim=-1) >= ref_threshold - allowance
    )
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
        query_len,
        batch_size,
        index_heads,
        index_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_index = torch.randn(
        key_len, batch_size, index_head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(
        query_len, batch_size, index_heads, device=device, dtype=torch.bfloat16
    )
    weights.mul_((index_heads * index_head_dim) ** -0.5)

    reference_scores = torch.einsum(
        "qbhd,tbd->bqht", q_index.float(), k_index.float()
    )
    reference_scores = torch.relu(reference_scores)
    reference_scores = (
        reference_scores * weights.permute(1, 0, 2).unsqueeze(-1).float()
    ).sum(dim=2)
    query_positions = q_start + torch.arange(query_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    reference_scores.masked_fill_(
        key_positions.view(1, 1, key_len) > query_positions.view(1, query_len, 1),
        float("-inf"),
    )

    actual = triton_topk_index_block(
        q_index, weights, k_index, topk, q_start=q_start, k_start=0
    )
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

    q_index = torch.randn(
        query_len, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        sequence_length, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    topk_indices = torch.randint(
        0, sequence_length, (batch_size, query_len, topk), device=device
    )

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

    key = torch.randn(
        sequence_length, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    topk_indices = torch.randint(
        0, sequence_length, (batch_size, query_len, topk), device=device
    )
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
    q_index = torch.randn(
        query_len, batch_size, 1, head_dim, device=device, dtype=dtype
    )
    selected_k = torch.randn(
        batch_size, query_len, topk, head_dim, device=device, dtype=dtype
    )
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
    expected_q = torch.einsum(
        "bqk,bqkd->bqd", masked_grad, selected_k.float()
    ) * score_scale
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
    grad_selected = torch.randn(
        batch_size, query_len, topk, head_dim, device=device, dtype=dtype
    )
    # Deliberately create heavy collisions, including the same key repeated within a row.
    topk_indices = torch.randint(
        0, 7, (batch_size, query_len, topk), device=device, dtype=torch.int64
    )

    actual = triton_scatter_selected_grad_to_sequence(
        grad_selected, topk_indices, sequence_length
    )
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
    hidden = torch.randn(
        sequence_length, batch_size, hidden_size, device=device, dtype=dtype
    )
    grad_output = torch.randn(
        batch_size, query_len, topk, out_features, device=device, dtype=dtype
    )
    topk_indices = torch.randint(
        0, sequence_length, (batch_size, query_len, topk), device=device
    )
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
    normalized = hidden_float * torch.rsqrt(
        hidden_float.square().mean(dim=-1, keepdim=True) + eps
    )
    normalized = normalized * effective_weight
    normalized = normalized.to(hidden.dtype)
    normalized_by_batch = normalized.permute(1, 0, 2)
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    selected_input = normalized_by_batch[batch_indices, topk_indices]
    expected = grad_output.reshape(-1, out_features).float().t().matmul(
        selected_input.reshape(-1, hidden_size).float()
    )
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_simplified_layernorm_wgrad_uses_exact_recompute_fallback(dtype):
    torch.manual_seed(742)
    device = torch.device("cuda")
    sequence_length, batch_size, hidden_size, out_features = 71, 2, 96, 64
    hidden = torch.randn(
        sequence_length, batch_size, hidden_size, device=device, dtype=dtype
    )
    grad_output = torch.randn(
        sequence_length, batch_size, out_features, device=device, dtype=dtype
    )
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
    normalized = F.layer_norm(
        hidden,
        (hidden_size,),
        effective_weight,
        norm_bias,
        eps,
    )
    expected = grad_output.reshape(-1, out_features).float().t().matmul(
        normalized.reshape(-1, hidden_size).float()
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
    expected = grad_output.reshape(-1, out_features).t().matmul(
        normalized.reshape(-1, hidden_size)
    )
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

    q_index = torch.randn(
        query_len, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    key_block = torch.randn(
        key_len, batch_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )

    actual = triton_simplified_index_scores_block(
        q_index, key_block, score_scale, q_start, k_start
    )
    assert actual is not None
    assert actual.dtype == torch.float32

    q_by_batch = q_index[:, :, 0, :].permute(1, 0, 2).float()
    key_by_batch = key_block[:, :, 0, :].permute(1, 0, 2).float()
    expected = (
        q_by_batch.unsqueeze(2) * key_by_batch.unsqueeze(1)
    ).sum(dim=-1) * score_scale
    query_positions = q_start + torch.arange(query_len, device=device)
    key_positions = k_start + torch.arange(key_len, device=device)
    invalid = key_positions.view(1, 1, key_len) > query_positions.view(1, query_len, 1)
    invalid = invalid.expand(batch_size, -1, -1)
    expected = expected.masked_fill(invalid, float("-inf"))

    assert torch.equal(torch.isneginf(actual), invalid)
    torch.testing.assert_close(actual[~invalid], expected[~invalid], rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_selected_index_scores_backward_matches_reference():
    torch.manual_seed(123)
    device = torch.device("cuda")
    batch_size = 2
    query_len = 9
    topk = 11
    index_heads = 3
    index_head_dim = 32
    q_start = 4

    q_index = torch.randn(
        query_len, batch_size, index_heads, index_head_dim, device=device, requires_grad=True
    )
    weights = torch.randn(query_len, batch_size, index_heads, device=device, requires_grad=True)
    selected_k = torch.randn(
        batch_size, query_len, topk, index_head_dim, device=device, requires_grad=True
    )
    topk_indices = torch.stack(
        [
            torch.randint(0, q_start + query_idx + 1, (batch_size, topk), device=device)
            for query_idx in range(query_len)
        ],
        dim=1,
    )
    grad = torch.randn(batch_size, query_len, topk, device=device)

    tri_scores = triton_selected_index_scores(q_index, weights, selected_k, topk_indices, q_start)
    (tri_scores * grad).sum().backward()
    tri_grads = (q_index.grad.clone(), weights.grad.clone(), selected_k.grad.clone())

    q_ref = q_index.detach().clone().requires_grad_(True)
    w_ref = weights.detach().clone().requires_grad_(True)
    sk_ref = selected_k.detach().clone().requires_grad_(True)
    ref_scores = _selected_index_scores_reference(q_ref, w_ref, sk_ref, topk_indices, q_start)
    (ref_scores * grad).sum().backward()

    torch.testing.assert_close(tri_scores, ref_scores, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tri_grads[0], q_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_grads[1], w_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_grads[2], sk_ref.grad, rtol=1e-4, atol=1e-4)


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
@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("use_hadamard", [False, True])
def test_native_indexer_loss_wgrad_matches_autograd(rotary_interleaved, use_hadamard):
    if use_hadamard and hadamard_transform is None:
        pytest.skip("fast_hadamard_transform is not installed")
    torch.manual_seed(1234)
    device = torch.device("cuda")
    # The native chain is a mixed-precision training path regardless of whether Hadamard is
    # enabled: model operands are BF16, reductions/WGRAD accumulation are FP32, and returned
    # parameter gradients are rounded to the parameter dtype. Using FP32 only for the
    # no-Hadamard cases accidentally tested the kernel's TF32 debug behavior instead.
    dtype = torch.bfloat16
    seqlen = 9
    batch_size = 2
    hidden_size = 16
    query_len = 4
    q_start = 3
    index_heads = 2
    index_head_dim = 8
    topk = 5
    rotary_dim = 4

    hidden_states = torch.randn(seqlen, batch_size, hidden_size, device=device, dtype=dtype)
    linear_q_weight = torch.randn(
        index_heads * index_head_dim, hidden_size, device=device, dtype=dtype, requires_grad=True
    )
    linear_k_weight = torch.randn(
        index_head_dim, hidden_size, device=device, dtype=dtype, requires_grad=True
    )
    k_norm_weight = torch.randn(index_head_dim, device=device, dtype=dtype, requires_grad=True)
    k_norm_bias = torch.randn(index_head_dim, device=device, dtype=dtype, requires_grad=True)
    linear_weights_weight = torch.randn(
        index_heads, hidden_size, device=device, dtype=dtype, requires_grad=True
    )
    topk_indices = torch.stack(
        [
            torch.randint(0, q_start + query_idx + 1, (batch_size, topk), device=device)
            for query_idx in range(query_len)
        ],
        dim=1,
    )
    grad_scores = torch.randn(batch_size, query_len, topk, device=device)
    rotary = _DummyRotary(rotary_dim=rotary_dim, rotary_interleaved=rotary_interleaved)

    q_index, weights = _project_q_index_tile(
        hidden_states.detach(),
        q_start,
        q_start + query_len,
        linear_q_weight,
        linear_weights_weight,
        index_heads,
        index_head_dim,
        rotary_dim,
        rotary,
        rotary_interleaved,
        use_indexer_rope=True,
        use_hadamard=use_hadamard,
    )
    selected_scores = _selected_index_scores_tile(
        hidden_states.detach(),
        q_start,
        q_start + query_len,
        topk_indices,
        q_index,
        weights,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        True,
        1.0e-5,
        index_head_dim,
        rotary_dim,
        rotary,
        rotary_interleaved,
        use_indexer_rope=True,
        use_hadamard=use_hadamard,
    )
    ref_grads = torch.autograd.grad(
        selected_scores,
        [
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        ],
        grad_outputs=grad_scores,
    )

    native_grads = [torch.zeros_like(grad, dtype=torch.float32) for grad in ref_grads]
    with torch.no_grad():
        native_done = _native_indexer_loss_wgrad_chunk(
            hidden_states.detach(),
            q_start,
            q_start + query_len,
            topk_indices,
            q_index.detach(),
            weights.detach(),
            grad_scores,
            linear_q_weight.detach(),
            linear_k_weight.detach(),
            k_norm_weight.detach(),
            k_norm_bias.detach(),
            True,
            linear_weights_weight.detach(),
            1.0e-5,
            index_head_dim,
            rotary_dim,
            rotary,
            rotary_interleaved,
            use_indexer_rope=True,
            use_hadamard=use_hadamard,
            grad_linear_q_weight=native_grads[0],
            grad_linear_k_weight=native_grads[1],
            grad_k_norm_weight=native_grads[2],
            grad_k_norm_bias=native_grads[3],
            grad_linear_weights_weight=native_grads[4],
            profile=None,
        )

    assert native_done
    for native_grad, ref_grad in zip(native_grads, ref_grads):
        torch.testing.assert_close(native_grad, ref_grad.float(), rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
def test_triton_k_ln_backward_autotune_clears_partial_reductions():
    """Autotune configs with different row blocks must not leave stale LN partials."""
    torch.manual_seed(123)
    device = torch.device("cuda")
    batch_size, query_len, topk, out_features = 2, 256, 16, 128
    k_linear = torch.randn(
        batch_size,
        query_len,
        topk,
        out_features,
        device=device,
        dtype=torch.bfloat16,
    )
    grad_k_norm = torch.randn_like(k_linear, dtype=torch.float32)
    k_norm_weight = torch.randn(out_features, device=device, dtype=torch.bfloat16)
    grad_weight = torch.zeros(out_features, device=device, dtype=torch.float32)
    grad_bias = torch.zeros_like(grad_weight)

    prepared = triton_k_ln_backward_prepare(
        grad_k_norm,
        k_linear,
        k_norm_weight,
        1.0e-5,
        grad_weight,
        grad_bias,
        torch.bfloat16,
    )
    assert prepared is not None
    _, partial_weight, partial_bias = prepared
    assert triton_k_ln_param_reduce(
        partial_weight, partial_bias, grad_weight, grad_bias
    )

    k_float = k_linear.float()
    mean = k_float.mean(dim=-1, keepdim=True)
    centered = k_float - mean
    rstd = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    normalized = centered * rstd
    expected_weight = (grad_k_norm * normalized).sum(dim=(0, 1, 2))
    expected_bias = grad_k_norm.sum(dim=(0, 1, 2))

    torch.testing.assert_close(grad_weight, expected_weight, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(grad_bias, expected_bias, rtol=5e-3, atol=5e-3)


def test_transformer_config_accepts_min_memory_sparse_forward_dense_loss():
    for backend in ("triton-min-memory", "torch-min-memory"):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend=backend,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_hadamard=True,
        )

        assert not config.dsa_fwd_use_dense_attn
        assert not config.dsa_indexer_use_sparse_loss


def test_transformer_config_sparse_forward_dense_loss_rejects_selected_score_cache():
    with pytest.raises(AssertionError, match="dsa_kernel_cache_selected_scores"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="triton-min-memory",
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_hadamard=True,
            dsa_kernel_cache_selected_scores=True,
        )


def test_min_memory_impl_matches_reference_forward_and_loss():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 6
    hidden_size = 16
    num_heads = 4
    num_query_groups = 2
    head_dim = 8
    index_heads = 2
    index_head_dim = 4
    topk = 3
    loss_coeff = 0.7

    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    query = torch.randn(seqlen, batch_size, num_heads, head_dim)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    linear_q_weight = torch.randn(index_heads * index_head_dim, hidden_size)
    linear_k_weight = torch.randn(index_head_dim, hidden_size)
    k_norm_weight = torch.randn(index_head_dim)
    k_norm_bias = torch.randn(index_head_dim)
    linear_weights_weight = torch.randn(index_heads, hidden_size)
    pg_collection = _DummyPGCollection()

    q_index = F.linear(hidden_states, linear_q_weight).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    k_index = F.layer_norm(
        F.linear(hidden_states, linear_k_weight),
        (index_head_dim,),
        k_norm_weight,
        k_norm_bias,
    )
    weights = F.linear(hidden_states, linear_weights_weight)
    weights = weights * (index_heads**-0.5) * (index_head_dim**-0.5)
    index_scores, topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, topk, _causal_mask(seqlen, hidden_states.device)
    )
    reference_output = unfused_grouped_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        head_dim**-0.5,
        use_gather=True,
    )
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, topk_indices),
    )

    output, loss = _forward_min_memory_impl(
        query,
        key,
        value,
        hidden_states,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        True,
        linear_weights_weight,
        1e-5,
        index_heads,
        index_head_dim,
        topk,
        0,
        None,
        False,
        False,
        head_dim**-0.5,
        loss_coeff,
        2,
        3,
        pg_collection,
    )

    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(loss, reference_loss)


def test_min_memory_impl_matches_reference_rope_interleaved_layout():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 6
    hidden_size = 14
    num_heads = 4
    num_query_groups = 2
    head_dim = 4
    index_heads = 2
    index_head_dim = 6
    rotary_dim = 4
    topk = 3
    loss_coeff = 0.7
    config_rotary_interleaved = True

    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    query = torch.randn(seqlen, batch_size, num_heads, head_dim)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    linear_q_weight = torch.randn(index_heads * index_head_dim, hidden_size)
    linear_k_weight = torch.randn(index_head_dim, hidden_size)
    k_norm_weight = torch.randn(index_head_dim)
    k_norm_bias = torch.randn(index_head_dim)
    linear_weights_weight = torch.randn(index_heads, hidden_size)
    pg_collection = _DummyPGCollection()
    rotary = _DummyRotary(
        rotary_dim, rotary_interleaved=config_rotary_interleaved
    )

    q_index = F.linear(hidden_states, linear_q_weight).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    q_index = _apply_reference_indexer_rope(
        q_index, rotary, config_rotary_interleaved, rotary_dim
    )
    k_index = F.layer_norm(
        F.linear(hidden_states, linear_k_weight),
        (index_head_dim,),
        k_norm_weight,
        k_norm_bias,
    ).reshape(seqlen, batch_size, 1, index_head_dim)
    k_index = _apply_reference_indexer_rope(
        k_index, rotary, config_rotary_interleaved, rotary_dim
    ).reshape(seqlen, batch_size, index_head_dim)
    weights = F.linear(hidden_states, linear_weights_weight)
    weights = weights * (index_heads**-0.5) * (index_head_dim**-0.5)

    index_scores, _ = fused_qk_topk_naive(
        q_index, k_index, weights, topk, _causal_mask(seqlen, hidden_states.device)
    )
    projected_q, projected_weights = _project_q_index_tile(
        hidden_states,
        0,
        seqlen,
        linear_q_weight,
        linear_weights_weight,
        index_heads,
        index_head_dim,
        rotary_dim,
        rotary,
        config_rotary_interleaved,
        use_indexer_rope=True,
        use_hadamard=False,
    )
    projected_k = _project_k_index_block(
        hidden_states,
        0,
        seqlen,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        True,
        1.0e-5,
        index_head_dim,
        rotary_dim,
        rotary,
        config_rotary_interleaved,
        use_indexer_rope=True,
        use_hadamard=False,
    )
    torch.testing.assert_close(projected_q, q_index, msg="interleaved RoPE Q projection")
    torch.testing.assert_close(projected_k, k_index, msg="interleaved RoPE K projection")
    torch.testing.assert_close(projected_weights, weights, msg="routing-weight projection")

    routing_topk_cache = []
    output, loss = _forward_min_memory_impl(
        query,
        key,
        value,
        hidden_states,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        True,
        linear_weights_weight,
        1e-5,
        index_heads,
        index_head_dim,
        topk,
        rotary_dim,
        rotary,
        True,
        False,
        head_dim**-0.5,
        loss_coeff,
        2,
        3,
        pg_collection,
        rotary_interleaved=config_rotary_interleaved,
        routing_topk_cache=routing_topk_cache,
    )

    padded_topk = []
    q_offset = 0
    target_topk = min(topk, seqlen)
    for tile_topk in routing_topk_cache:
        q_end = q_offset + tile_topk.size(1)
        if tile_topk.size(-1) < target_topk:
            assert q_end < seqlen
            tile_topk = F.pad(
                tile_topk, (0, target_topk - tile_topk.size(-1)), value=q_end
            )
        padded_topk.append(tile_topk)
        q_offset = q_end
    min_memory_topk = torch.cat(padded_topk, dim=1)
    reference_output = unfused_grouped_dsa_fn(
        query,
        key,
        value,
        min_memory_topk,
        head_dim**-0.5,
        use_gather=True,
    )
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=min_memory_topk,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, min_memory_topk),
    )

    torch.testing.assert_close(output, reference_output, msg="interleaved RoPE sparse output")
    torch.testing.assert_close(loss, reference_loss, msg="interleaved RoPE sparse KL")


@pytest.mark.parametrize("input_norm_kind", ["none", "rmsnorm", "layernorm"])
def test_min_memory_impl_matches_reference_gradients(input_norm_kind):
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 5
    hidden_size = 12
    num_heads = 4
    num_query_groups = 2
    head_dim = 4
    index_heads = 2
    index_head_dim = 4
    topk = 3
    loss_coeff = 0.7
    pg_collection = _DummyPGCollection()

    def _make_tensors():
        hidden_states = torch.randn(seqlen, batch_size, hidden_size)
        query = torch.randn(seqlen, batch_size, num_heads, head_dim, requires_grad=True)
        key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, requires_grad=True)
        value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, requires_grad=True)
        linear_q_weight = torch.randn(
            index_heads * index_head_dim, hidden_size, requires_grad=True
        )
        linear_k_weight = torch.randn(index_head_dim, hidden_size, requires_grad=True)
        k_norm_weight = torch.randn(index_head_dim, requires_grad=True)
        k_norm_bias = torch.randn(index_head_dim, requires_grad=True)
        linear_weights_weight = torch.randn(index_heads, hidden_size, requires_grad=True)
        return (
            hidden_states,
            query,
            key,
            value,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        )

    min_tensors = _make_tensors()
    ref_tensors = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in min_tensors)
    input_norm = None
    if input_norm_kind != "none":
        linear_qkv = SimpleNamespace(
            layer_norm_weight=torch.randn(hidden_size),
            layer_norm_bias=(
                torch.randn(hidden_size) if input_norm_kind == "layernorm" else None
            ),
            eps=1.0e-5,
        )
        input_norm = _indexer_input_norm_spec(
            linear_qkv,
            SimpleNamespace(
                normalization=(
                    "LayerNorm" if input_norm_kind == "layernorm" else "RMSNorm"
                ),
                layernorm_epsilon=1.0e-5,
                layernorm_zero_centered_gamma=False,
            ),
        )

    (
        hidden_states,
        query,
        key,
        value,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        linear_weights_weight,
    ) = min_tensors
    output, loss = DSAMinMemoryGQAFn.apply(
        query,
        key,
        value,
        hidden_states,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        linear_weights_weight,
        True,
        1e-5,
        index_heads,
        index_head_dim,
        topk,
        0,
        None,
        False,
        False,
        head_dim**-0.5,
        loss_coeff,
        2,
        3,
        pg_collection,
        False,
        False,
        0,
        "",
        False,
        False,
        False,
        True,
        input_norm,
    )
    (output.sum() + loss).backward()

    (
        ref_hidden_states,
        ref_query,
        ref_key,
        ref_value,
        ref_linear_q_weight,
        ref_linear_k_weight,
        ref_k_norm_weight,
        ref_k_norm_bias,
        ref_linear_weights_weight,
    ) = ref_tensors
    ref_indexer_input = _normalized_indexer_input(ref_hidden_states, input_norm)
    q_index = F.linear(ref_indexer_input, ref_linear_q_weight).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    k_index = F.layer_norm(
        F.linear(ref_indexer_input, ref_linear_k_weight),
        (index_head_dim,),
        ref_k_norm_weight,
        ref_k_norm_bias,
    )
    weights = F.linear(ref_indexer_input, ref_linear_weights_weight)
    weights = weights * (index_heads**-0.5) * (index_head_dim**-0.5)
    index_scores, topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, topk, _causal_mask(seqlen, ref_hidden_states.device)
    )
    ref_output = unfused_grouped_dsa_fn(
        ref_query,
        ref_key,
        ref_value,
        topk_indices,
        head_dim**-0.5,
        use_gather=True,
    )
    ref_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=ref_query.detach(),
        key=ref_key.detach(),
        softmax_scale=head_dim**-0.5,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, topk_indices),
    )
    (ref_output.sum() + ref_loss).backward()

    for min_tensor, ref_tensor in zip(min_tensors[1:], ref_tensors[1:]):
        torch.testing.assert_close(min_tensor.grad, ref_tensor.grad)


@pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(),
    reason="CUDA Triton kernels are required for this test.",
)
def test_triton_selected_k_linear_matches_pytorch_projection():
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seqlen, batch_size, query_len, topk = 11, 2, 5, 4
    hidden_size, index_head_dim = 64, 32
    hidden_states = torch.randn(seqlen, batch_size, hidden_size, device=device, dtype=dtype)
    linear_k_weight = torch.randn(index_head_dim, hidden_size, device=device, dtype=dtype)
    topk_indices = torch.randint(0, seqlen, (batch_size, query_len, topk), device=device)

    projected = triton_selected_k_linear(hidden_states, topk_indices, linear_k_weight)
    hidden_by_batch = hidden_states.permute(1, 0, 2)
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    selected_hidden = hidden_by_batch[batch_index, topk_indices]
    reference = F.linear(selected_hidden, linear_k_weight)

    assert projected is not None
    torch.testing.assert_close(projected.float(), reference.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(),
    reason="CUDA Triton kernels are required for this test.",
)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_triton_selected_k_linear_matches_normalized_projection(zero_centered_gamma):
    torch.manual_seed(124)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seqlen, batch_size, query_len, topk = 11, 2, 5, 4
    hidden_size, index_head_dim = 64, 32
    eps = 1.0e-5
    hidden_states = torch.randn(seqlen, batch_size, hidden_size, device=device, dtype=dtype)
    linear_k_weight = torch.randn(index_head_dim, hidden_size, device=device, dtype=dtype)
    norm_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    topk_indices = torch.randint(0, seqlen, (batch_size, query_len, topk), device=device)
    norm_stats = triton_simplified_input_norm_stats(hidden_states, eps, "RMSNorm")

    projected = triton_selected_k_linear(
        hidden_states,
        topk_indices,
        linear_k_weight,
        norm_weight,
        norm_stats,
        zero_centered_gamma,
    )
    effective_weight = norm_weight + 1.0 if zero_centered_gamma else norm_weight
    hidden_float = hidden_states.float()
    normalized_hidden = (
        hidden_float
        * torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + eps)
        * effective_weight.float()
    ).to(dtype)
    hidden_by_batch = normalized_hidden.permute(1, 0, 2)
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    reference = F.linear(hidden_by_batch[batch_index, topk_indices], linear_k_weight)

    assert norm_stats is not None
    assert projected is not None
    torch.testing.assert_close(projected.float(), reference.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(),
    reason="CUDA Triton kernels are required for this test.",
)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_triton_selected_score_fusion_matches_normalized_standard_indexer(
    zero_centered_gamma,
):
    torch.manual_seed(125)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seqlen, batch_size, query_len, topk = 11, 2, 5, 4
    hidden_size, index_heads, index_dim = 64, 3, 32
    eps = 1.0e-5
    hidden_states = torch.randn(seqlen, batch_size, hidden_size, device=device, dtype=dtype)
    linear_k_weight = torch.randn(index_dim, hidden_size, device=device, dtype=dtype)
    input_norm_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    k_norm_weight = torch.randn(index_dim, device=device, dtype=dtype)
    k_norm_bias = torch.randn(index_dim, device=device, dtype=dtype)
    q_index = torch.randn(query_len, batch_size, index_heads, index_dim, device=device, dtype=dtype)
    routing_weights = torch.randn(query_len, batch_size, index_heads, device=device, dtype=dtype)
    topk_indices = torch.stack(
        [
            torch.randint(0, query_idx + 1, (batch_size, topk), device=device)
            for query_idx in range(query_len)
        ],
        dim=1,
    )
    norm_stats = triton_simplified_input_norm_stats(hidden_states, eps, "RMSNorm")
    fused = triton_selected_index_scores_from_hidden(
        hidden_states,
        topk_indices,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        q_index,
        routing_weights,
        torch.empty(1, device=device, dtype=torch.float32),
        0,
        eps,
        0,
        False,
        False,
        False,
        True,
        1.0,
        1.0,
        return_k_linear=True,
        input_norm_weight=input_norm_weight,
        input_norm_stats=norm_stats,
        input_norm_zero_centered_gamma=zero_centered_gamma,
    )

    hidden_float = hidden_states.float()
    effective_input_norm_weight = (
        input_norm_weight + 1.0 if zero_centered_gamma else input_norm_weight
    )
    normalized_hidden = (
        hidden_float
        * torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + eps)
        * effective_input_norm_weight.float()
    ).to(dtype)
    hidden_by_batch = normalized_hidden.permute(1, 0, 2)
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    selected_hidden = hidden_by_batch[batch_index, topk_indices]
    expected_k_linear = F.linear(selected_hidden, linear_k_weight)
    expected_k = F.layer_norm(
        expected_k_linear,
        (index_dim,),
        k_norm_weight,
        k_norm_bias,
        eps,
    )
    expected_scores = torch.einsum(
        "qbhd,bqkd->bqhk", q_index.float(), expected_k.float()
    )
    expected_scores = (
        torch.relu(expected_scores)
        * routing_weights.permute(1, 0, 2).unsqueeze(-1).float()
    ).sum(dim=2)

    assert norm_stats is not None
    assert fused is not None
    scores, k_linear = fused
    torch.testing.assert_close(scores, expected_scores, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_linear.float(), expected_k_linear.float(), atol=2e-2, rtol=2e-2)


def test_compute_gqa_dsa_indexer_loss_dense_and_sparse():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32)
    )
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

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
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

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
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

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
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

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
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
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)

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
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
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
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
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
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
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
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
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
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
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
        torch.sort(chunked_scores, dim=-1).values,
        torch.sort(dense_scores.gather(-1, dense_indices), dim=-1).values,
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

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
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

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
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

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
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


