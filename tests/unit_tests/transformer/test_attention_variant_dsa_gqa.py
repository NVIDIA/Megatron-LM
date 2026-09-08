# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    fused_qk_topk_chunked,
    fused_qk_topk_naive,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGQACoreAttention,
    DSGroupedSelfAttention,
    SimplifiedDSGQAIndexer,
    SimplifiedDSGQAIndexerSubmodules,
    _DSAZeroParamDependency,
    _simplified_index_scores,
    _simplified_indexer_input,
    _simplified_indexer_norm_spec,
    compute_gqa_dsa_indexer_loss,
    unfused_grouped_dsa_fn,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import dsa_stack_spec
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    dsa_dense_indexer_loss,
    dsa_min_memory_gqa,
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


@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
def test_simplified_indexer_uses_fused_main_qkv_normalized_input(normalization):
    torch.manual_seed(123)
    hidden = torch.randn(5, 2, 8, requires_grad=True)
    weight = torch.randn(8)
    bias = torch.randn(8) if normalization == "LayerNorm" else None
    linear_qkv = SimpleNamespace(
        layer_norm_weight=weight, layer_norm_bias=bias, eps=1.0e-5, skip_norm_and_all_gather=False
    )
    config = SimpleNamespace(
        normalization=normalization, layernorm_epsilon=1.0e-5, layernorm_zero_centered_gamma=False
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
    linear_qkv = SimpleNamespace(layer_norm_weight=torch.randn(8), skip_norm_and_all_gather=True)
    config = SimpleNamespace(
        normalization="RMSNorm", layernorm_epsilon=1.0e-5, layernorm_zero_centered_gamma=False
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
def test_simplified_indexer_input_honors_zero_centered_gamma_without_norm_grads(normalization):
    torch.manual_seed(321)
    hidden = torch.randn(4, 2, 6, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(6))
    bias = torch.nn.Parameter(torch.randn(6)) if normalization == "LayerNorm" else None
    linear_qkv = SimpleNamespace(
        layer_norm_weight=weight, layer_norm_bias=bias, eps=2.0e-5, skip_norm_and_all_gather=False
    )
    config = SimpleNamespace(
        normalization=normalization, layernorm_epsilon=2.0e-5, layernorm_zero_centered_gamma=True
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
        expected = F.layer_norm(hidden.detach(), (6,), effective_weight, bias.detach(), 2.0e-5)

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
    upstream_module = hybrid_stack_spec.submodules.attention_layer.submodules.self_attention.module
    assert upstream_module is not DSGroupedSelfAttention


def _causal_mask(seqlen: int, device: torch.device):
    return torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), dtype=torch.float32, device=device), diagonal=1
    )


def _causal_index_scores(index_scores: torch.Tensor):
    masked_scores = index_scores + _causal_mask(index_scores.size(1), index_scores.device).view(
        1, index_scores.size(1), index_scores.size(2)
    )
    return masked_scores.detach().requires_grad_(index_scores.requires_grad)


def _gather_selected_index_scores(index_scores, topk_indices):
    """Gather per-slot index scores, tolerating the indexer's -1 padding."""
    padded_slots = topk_indices < 0
    gathered = index_scores.gather(-1, topk_indices.clamp(min=0))
    return gathered.masked_fill(padded_slots, float("-inf"))


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
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=True,
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend=backend,
            dsa_kernel_cache_routing=True,
            dsa_kernel_cache_indexer_k=True,
            dsa_kernel_cache_selected_scores=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
            dsa_indexer_sparse_loss_use_topk_only=True,
            dsa_kernel_query_block_size=256,
            dsa_kernel_key_block_size=1024,
            dsa_min_memory_profile=True,
            dsa_min_memory_profile_rank=-1,
        )

        assert config.dsa_min_memory_backend == backend
        assert config.dsa_kernel_query_block_size == 256
        assert config.dsa_kernel_key_block_size == 1024
        assert config.dsa_kernel_cache_routing
        assert config.dsa_kernel_cache_indexer_k
        assert config.dsa_kernel_cache_selected_scores
        assert config.dsa_min_memory_profile
        assert config.dsa_min_memory_profile_rank == -1


def test_transformer_config_accepts_disabled_simplified_main_input_norm():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_simplified_indexer_disable_main_input_norm=True,
    )

    assert config.dsa_simplified_indexer_disable_main_input_norm


def test_transformer_config_rejects_standard_main_input_norm_for_simplified_dsa():
    with pytest.raises(AssertionError, match="dsa_indexer_mode='standard'"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            add_bias_linear=False,
            dsa_indexer_mode="simplified",
            dsa_indexer_topk=4,
            dsa_standard_indexer_use_main_input_norm=True,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_attention_wrapper_supplies_fused_norm_to_standard_indexer_only_when_enabled(enabled):
    hidden_states = torch.randn(3, 1, 8)
    linear_qkv = SimpleNamespace(layer_norm_weight=torch.randn(8), layer_norm_bias=None, eps=1.0e-5)
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
def test_attention_wrapper_honors_simplified_main_input_norm_disable(disable_main_input_norm):
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
            layer_norm_weight=torch.randn(8), layer_norm_bias=None, eps=1.0e-5
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
    indexer.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False) if learned_k else None
    return indexer


def test_transformer_config_accepts_simplified_dsa_and_derives_shape():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_min_memory_backend="torch-min-memory",
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
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=True,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=4,
        dsa_min_memory_backend="torch-min-memory",
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
            add_bias_linear=False,
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=True,
            dsa_indexer_head_dim=6,
            dsa_indexer_topk=4,
            dsa_indexer_reset_method="main-q-mean-rescaled",
            dsa_reset_indexer_on_load=True,
            dsa_min_memory_backend="torch-min-memory",
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
        )


def test_transformer_config_rejects_simplified_mode_without_dsa_variant():
    with pytest.raises(AssertionError, match="requires experimental_attention_variant='dsa'"):
        TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, dsa_indexer_mode="simplified"
        )


def test_transformer_config_rejects_reset_while_dsa_is_still_skipped():
    with pytest.raises(AssertionError, match="disabled when resetting"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            add_bias_linear=False,
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
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_min_memory_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    # The indexer reads its RoPE settings as dynamic attributes; megatron/training supplies them
    # from args, so a config built directly has to set them the same way.
    config.rotary_percent = 0.0
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
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=True,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=4,
        dsa_min_memory_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    # The indexer reads its RoPE settings as dynamic attributes; megatron/training supplies them
    # from args, so a config built directly has to set them the same way.
    config.rotary_percent = 0.0
    config.num_query_groups = 2

    class _TP2Group:
        def size(self):
            return 2

    def _build_linear(_spec, input_size, output_size, **_kwargs):
        return torch.nn.Linear(input_size, output_size, bias=False)

    monkeypatch.setattr(dsa_gqa, "build_module", _build_linear)
    indexer = SimplifiedDSGQAIndexer(
        config,
        SimplifiedDSGQAIndexerSubmodules(linear_q=torch.nn.Linear, linear_k=torch.nn.Linear),
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
        rotary_interleaved=True,
        use_cpu_initialization=True,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_min_memory_backend="torch-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    # The indexer reads its RoPE settings as dynamic attributes; megatron/training supplies them
    # from args, so a config built directly has to set them the same way. Setting them here is
    # the point of this test: the indexer's RoPE must track the model's.
    config.rotary_percent = 1.0
    config.rotary_seq_len_interpolation_factor = 2.0
    config.use_rope_scaling = True
    config.rope_scaling_factor = 4.0

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
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_min_memory_backend="torch-min-memory",
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
        normalization="RMSNorm", layernorm_epsilon=1.0e-5, layernorm_zero_centered_gamma=False
    )
    input_norm = _simplified_indexer_norm_spec(linear_qkv, norm_config)
    normalized_hidden = _simplified_indexer_input(hidden_states, input_norm)

    q_index = indexer.linear_q(normalized_hidden).reshape(seqlen, batch_size, 1, head_dim)
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
        dense_loss, (indexer.linear_q.weight, key, hidden_states), allow_unused=True
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
    query = torch.randn(seqlen, batch_size, num_query_heads, head_dim, requires_grad=True)
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
        normalization="RMSNorm", layernorm_epsilon=1.0e-5, layernorm_zero_centered_gamma=False
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
        selected_index_scores=_gather_selected_index_scores(index_scores, topk_indices),
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
    query = torch.randn(seqlen, batch_size, num_query_heads, attention_dim, requires_grad=True)
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    value = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size, requires_grad=True)
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk, learned_k=True)

    q_index = indexer.linear_q(hidden_states.detach()).reshape(seqlen, batch_size, 1, index_dim)
    k_index = indexer.linear_k(hidden_states.detach()).reshape(seqlen, batch_size, 1, index_dim)
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
        selected_index_scores=_gather_selected_index_scores(index_scores, topk_indices),
    )
    reference_grads = torch.autograd.grad(
        reference_output.float().sum() + reference_loss,
        (query, key, value, indexer.linear_q.weight, indexer.linear_k.weight),
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


def test_transformer_config_min_memory_accepts_sparse_loss_without_topk_only_flag():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        add_bias_linear=False,
        dsa_indexer_topk=4,
        dsa_min_memory_backend="triton-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )

    assert config.dsa_indexer_use_sparse_loss
    assert not config.dsa_indexer_sparse_loss_use_topk_only


@pytest.mark.parametrize("backend", ["reference", "torch-min-memory", "triton-min-memory"])
def test_transformer_config_accepts_dsa_train_main_only(backend):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        add_bias_linear=False,
        dsa_indexer_topk=4,
        dsa_min_memory_backend=backend,
        dsa_indexer_loss_coeff=0.0,
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
    ],
)
def test_transformer_config_rejects_incompatible_dsa_train_main_only_modes(override, match):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_min_memory_backend="triton-min-memory",
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
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend=backend,
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
        )

        assert config.dsa_fwd_use_dense_attn
        assert not config.dsa_indexer_use_sparse_loss


def test_transformer_config_dense_warmup_rejects_sparse_loss_and_caches():
    with pytest.raises(AssertionError, match="dsa_indexer_use_sparse_loss"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend="triton-min-memory",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
        )

    with pytest.raises(AssertionError, match="dsa_kernel_cache_routing"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend="triton-min-memory",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_kernel_cache_routing=True,
        )


def test_transformer_config_dense_warmup_requires_min_memory_backend():
    with pytest.raises(AssertionError, match="dsa_fwd_use_dense_attn"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend="reference",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
        )


def test_transformer_config_dense_warmup_requires_positive_loss_coeff_and_dsa_variant():
    with pytest.raises(AssertionError, match="dsa_indexer_loss_coeff"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend="triton-min-memory",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.0,
        )

    with pytest.raises(AssertionError, match="experimental_attention_variant='dsa'"):
        TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, dsa_fwd_use_dense_attn=True
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
                dsa_min_memory_backend=backend,
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
            dsa_min_memory_backend="triton-min-memory",
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
    monkeypatch.setattr(dsa_gqa, "unfused_grouped_dsa_fn", lambda query, *_args, **_kwargs: query)

    core = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=False,
            dsa_min_memory_backend="reference",
            dsa_fwd_skip_dsa=False,
            dsa_indexer_mode="standard",
            dsa_sparse_attention_use_gather=False,
            dsa_standard_indexer_use_main_input_norm=False,
            dsa_train_main_only=True,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
            dsa_indexer_sparse_loss_use_topk_only=False,
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
        core, query, key, value, None, hidden_states, attn_mask_type=AttnMaskType.causal
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
            dsa_min_memory_backend="triton-min-memory",
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


def test_transformer_config_cache_routing_requires_min_memory_backend():
    with pytest.raises(AssertionError, match="dsa_kernel_cache_routing"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend="reference",
            dsa_indexer_loss_coeff=0.1,
            dsa_kernel_cache_routing=True,
        )


@pytest.mark.parametrize(
    "cache_flag", ["dsa_kernel_cache_indexer_k", "dsa_kernel_cache_selected_scores"]
)
def test_transformer_config_optional_kernel_caches_require_min_memory_backend(cache_flag):
    with pytest.raises(AssertionError, match=cache_flag):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=True,
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend="reference",
            dsa_indexer_loss_coeff=0.1,
            **{cache_flag: True},
        )


def test_transformer_config_sparse_forward_dense_loss_rejects_selected_score_cache():
    with pytest.raises(AssertionError, match="dsa_kernel_cache_selected_scores"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend="triton-min-memory",
            dsa_indexer_loss_coeff=0.1,
            dsa_kernel_cache_selected_scores=True,
        )


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
        query=query, key=key, value=value, topk_indices=topk_indices, softmax_scale=head_dim**-0.5
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
        q=q, k=k, weights=weights, index_topk=topk, mask=mask
    )
    chunked_scores, chunked_indices = fused_qk_topk_chunked(
        q=q, k=k, weights=weights, index_topk=topk, mask=mask, key_chunk_size=4
    )

    expected_chunked_scores = dense_scores.gather(-1, chunked_indices)
    torch.testing.assert_close(chunked_scores, expected_chunked_scores)
    torch.testing.assert_close(
        torch.sort(chunked_scores, dim=-1).values,
        torch.sort(dense_scores.gather(-1, dense_indices), dim=-1).values,
    )


def test_transformer_config_accepts_min_memory_sparse_forward_dense_loss():
    for backend in ("triton-min-memory", "torch-min-memory"):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=8,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            add_bias_linear=False,
            dsa_indexer_topk=4,
            dsa_min_memory_backend=backend,
            dsa_indexer_loss_coeff=0.1,
        )

        assert not config.dsa_fwd_use_dense_attn
        assert not config.dsa_indexer_use_sparse_loss
