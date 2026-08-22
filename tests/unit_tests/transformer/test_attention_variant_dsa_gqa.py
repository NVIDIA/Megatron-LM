from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
import pytest

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.transformer.experimental_attention_variant.dsa import (
    fused_qk_topk_chunked,
    fused_qk_topk_naive,
    hadamard_transform,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGroupedSelfAttention,
    DSGQACoreAttention,
    _build_shifted_causal_mask,
    compute_gqa_dsa_indexer_loss,
    unfused_grouped_dsa_fn,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    DSAMinMemoryGQAFn,
    dsa_dense_indexer_loss,
    _forward_min_memory_impl,
    _native_indexer_loss_wgrad_chunk,
    _project_q_index_tile,
    _routing_key_chunk_size,
    _selected_index_scores_backward_torch,
    _selected_index_scores_tile,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory_triton import (
    HAVE_TRITON,
    triton_indexer_loss_grad,
    triton_linear_wgrad,
    triton_selected_k_linear,
    triton_selected_index_scores,
    triton_topk_index_block,
)
from megatron.core.transformer.enums import AttnMaskType
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


def test_mamba_stack_spec_uses_dsa_grouped_self_attention():
    attention_module = mamba_stack_spec.submodules.attention_layer.submodules.self_attention.module
    assert attention_module is DSGroupedSelfAttention


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
                attn_mask_type=AttnMaskType.causal,
            )

        assert output.shape == (4, 2, 16)
        assert not output.requires_grad

    assert [call["use_triton"] for call in calls] == [False, True]


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

    q_index = torch.randn(query_len, batch_size, index_heads, index_head_dim, device=device)
    k_index = torch.randn(key_len, batch_size, index_head_dim, device=device)
    weights = torch.randn(query_len, batch_size, index_heads, device=device)
    scores = torch.einsum("qbhd,tbd->bqht", q_index.float(), k_index.float())
    scores = torch.relu(scores)
    scores = (scores * weights.permute(1, 0, 2).unsqueeze(-1).float()).sum(dim=2)
    query_positions = (q_start + torch.arange(query_len, device=device)).view(query_len, 1)
    key_positions = torch.arange(key_len, device=device).view(1, key_len)
    scores = scores.masked_fill((key_positions > query_positions).unsqueeze(0), float("-inf"))
    ref_scores, ref_indices = scores.topk(topk, dim=-1)

    tri_scores, tri_indices = triton_topk_index_block(
        q_index, weights, k_index, topk, q_start=q_start, k_start=0
    )

    torch.testing.assert_close(tri_scores, ref_scores, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tri_indices, ref_indices)


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

    ref = grad_output.float().t().matmul(input_tensor.float())
    rtol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    atol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(grad_weight, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA Triton only")
@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("use_hadamard", [False, True])
def test_native_indexer_loss_wgrad_matches_autograd(rotary_interleaved, use_hadamard):
    if use_hadamard and hadamard_transform is None:
        pytest.skip("fast_hadamard_transform is not installed")
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if use_hadamard else torch.float32
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
    rtol = 3e-2 if use_hadamard else 2e-3
    atol = 3e-2 if use_hadamard else 2e-3
    for native_grad, ref_grad in zip(native_grads, ref_grads):
        torch.testing.assert_close(native_grad, ref_grad.float(), rtol=rtol, atol=atol)


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
    rotary = _DummyRotary(rotary_dim, rotary_interleaved=False)

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
    )

    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(loss, reference_loss)


def test_min_memory_impl_matches_reference_gradients():
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
    q_index = F.linear(ref_hidden_states, ref_linear_q_weight).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    k_index = F.layer_norm(
        F.linear(ref_hidden_states, ref_linear_k_weight),
        (index_head_dim,),
        ref_k_norm_weight,
        ref_k_norm_bias,
    )
    weights = F.linear(ref_hidden_states, ref_linear_weights_weight)
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
