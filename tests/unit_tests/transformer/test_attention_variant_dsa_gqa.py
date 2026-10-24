# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_gqa_decode_kernel as decode_kernels,
)
from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGQACoreAttention,
    DSGroupedSelfAttention,
    SimplifiedDSGQAIndexer,
    SimplifiedDSGQAIndexerSubmodules,
    _dsa_caches_indexer_k,
    _dsa_caches_routing,
    _dsa_caches_selected_scores,
    _DSAZeroParamDependency,
    _simplified_index_scores,
    _simplified_indexer_input,
    _simplified_indexer_norm_spec,
    compute_gqa_dsa_indexer_loss,
    unfused_grouped_dsa_fn,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import (
    dsa_stack_spec,
    gdp_dsa_stack_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    dsa_dense_indexer_loss,
    dsa_min_memory_gqa,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


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


def test_gdp_dsa_stack_spec_uses_dsa_grouped_self_attention():
    module = gdp_dsa_stack_spec.submodules.attention_layer.submodules.self_attention.module
    assert module is DSGroupedSelfAttention


def test_gdp_dsa_stack_spec_does_not_mutate_upstream_gdp_spec():
    """gated_delta_product_stack_spec shares hybrid_stack_spec's attention_layer object.

    Deriving either DSA stack therefore has to deep copy first, or swapping the module on one
    would swap it on the other and on upstream's spec as well.
    """
    upstream = gated_delta_product_stack_spec.submodules.attention_layer
    assert upstream.submodules.self_attention.module is not DSGroupedSelfAttention


def test_the_two_dsa_stacks_keep_their_own_mamba_mixers():
    """Only the attention layer is DSA's to change; the mixers are what distinguish the stacks."""
    assert gdp_dsa_stack_spec.submodules.mamba_layer != dsa_stack_spec.submodules.mamba_layer


def _causal_mask(seqlen: int, device: torch.device):
    return torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), dtype=torch.float32, device=device), diagonal=1
    )


def _causal_index_scores(index_scores: torch.Tensor):
    masked_scores = index_scores + _causal_mask(index_scores.size(1), index_scores.device).view(
        1, index_scores.size(1), index_scores.size(2)
    )
    return masked_scores.detach().requires_grad_(index_scores.requires_grad)


def _random_topk_indices(batch_size: int, seqlen: int, topk: int):
    return torch.randn(batch_size, seqlen, seqlen).topk(topk, dim=-1).indices


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
    for backend in ("min-memory-triton", "min-memory-torch"):
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
            dsa_kernel_backend=backend,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
        )

        assert config.dsa_kernel_backend == backend


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
        dsa_kernel_backend="min-memory-torch",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )

    assert config.dsa_indexer_n_heads == 1
    assert config.dsa_indexer_head_dim == 8


def test_simplified_dsa_turns_off_indexer_rotation():
    """Simplified DSA scores a plain dot product, so the rotation setting resolves to off.

    dsa_indexer_rotate_activation defaults True for the standard DeepSeek indexer. Rather than
    make every simplified config turn it off, __post_init__ resolves it -- there is no Hadamard
    rotation in the simplified scoring path for the flag to control.
    """
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
        # Required now that the unset backend resolves to min-memory-triton before the
        # min-memory checks run. Incidental to what this test is about.
        dsa_indexer_loss_coeff=0.1,
    )

    assert not config.dsa_indexer_rotate_activation


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
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=4,
        dsa_kernel_backend="min-memory-torch",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )

    assert config.dsa_indexer_n_heads == 1
    assert config.dsa_indexer_head_dim == 6


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dsa_fwd_use_dense_attn": True}, "dsa_fwd_use_dense_attn requires"),
        ({"dsa_reset_indexer_on_load": True}, "dsa_reset_indexer_on_load requires"),
    ],
)
def test_transformer_config_rejects_simplified_only_options_in_standard_mode(overrides, message):
    with pytest.raises(AssertionError, match=message):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            kv_channels=8,
            experimental_attention_variant="dsa",
            add_bias_linear=False,
            dsa_indexer_mode="standard",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_indexer_loss_coeff=0.1,
            **overrides,
        )


def test_transformer_config_rejects_simplified_mode_without_dsa_variant():
    with pytest.raises(AssertionError, match="requires experimental_attention_variant='dsa'"):
        TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, dsa_indexer_mode="simplified"
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
        dsa_kernel_backend="min-memory-torch",
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
        SimplifiedDSGQAIndexerSubmodules(linear_q=torch.nn.Linear, linear_k=torch.nn.Linear),
        pg_collection=pg_collection,
    )

    assert indexer.linear_q.weight.shape == (8, 32)
    assert indexer.linear_k.weight.shape == (8, 32)
    assert set(indexer.state_dict()) == {"linear_q.weight", "linear_k.weight"}
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
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=4,
        dsa_kernel_backend="min-memory-torch",
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
        dsa_kernel_backend="min-memory-torch",
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
        dsa_kernel_backend="min-memory-torch",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    config_kwargs.update(kwargs)
    with pytest.raises(AssertionError, match=message):
        TransformerConfig(**config_kwargs)


def test_simplified_learned_k_dense_loss_matches_reference_and_is_detached():
    torch.manual_seed(789)
    seqlen, batch_size, hidden_size = 7, 2, 12
    num_query_heads, attention_dim, index_dim = 4, 3, 5
    attention_scale = attention_dim**-0.5
    loss_coeff = 0.4
    query = torch.randn(seqlen, batch_size, num_query_heads, attention_dim)
    key = torch.randn(seqlen, batch_size, 1, attention_dim, requires_grad=True)
    hidden_states = torch.randn(seqlen, batch_size, hidden_size, requires_grad=True)
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk=3)

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
    indexer = _simplified_test_indexer(hidden_size, index_dim, topk)

    q_index = indexer.linear_q(hidden_states.detach()).reshape(seqlen, batch_size, 1, index_dim)
    k_index = indexer.linear_k(hidden_states.detach()).reshape(seqlen, batch_size, 1, index_dim)
    index_scores = _simplified_index_scores(q_index, k_index, indexer.softmax_scale)
    index_scores = index_scores + _causal_mask(seqlen, query.device)
    topk_indices = index_scores.topk(topk, dim=-1).indices.sort(dim=-1).values
    reference_output = unfused_grouped_dsa_fn(query, key, value, topk_indices, attention_scale)
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query.detach(),
        key.detach(),
        attention_scale,
        loss_coeff,
        True,
        indexer.pg_collection,
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
        dsa_kernel_backend="min-memory-triton",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )

    assert config.dsa_indexer_use_sparse_loss


def test_transformer_config_accepts_dense_warmup_min_memory_backend():
    for backend in ("min-memory-triton", "min-memory-torch"):
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
            dsa_kernel_backend=backend,
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
            dsa_kernel_backend="min-memory-triton",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=True,
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
            dsa_kernel_backend="reference",
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
            dsa_kernel_backend="min-memory-triton",
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

    for backend in ("min-memory-torch", "min-memory-triton"):
        core = SimpleNamespace(
            config=SimpleNamespace(
                dsa_kernel_backend=backend,
                dsa_indexer_use_sparse_loss=True,
                dsa_indexer_rotate_activation=True,
                fp8=None,
                fp8_param=False,
                layernorm_zero_centered_gamma=False,
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


def test_dense_warmup_no_grad_validation_uses_dense_core_attention():
    torch.manual_seed(123)
    calls = []

    class _DenseCore:
        def __call__(self, query, key, value, attention_mask, **kwargs):
            calls.append((query, key, value, attention_mask, kwargs))
            return query.new_empty(query.size(0), query.size(1), query.size(2) * value.size(-1))

    core = SimpleNamespace(
        config=SimpleNamespace(
            dsa_kernel_backend="min-memory-triton",
            dsa_fwd_use_dense_attn=True,
            dsa_indexer_use_sparse_loss=False,
            dsa_indexer_rotate_activation=True,
            fp8=None,
            fp8_param=False,
            layernorm_zero_centered_gamma=False,
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


def _routing_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        add_bias_linear=False,
        dsa_indexer_topk=4,
        dsa_kernel_backend="min-memory-triton",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_dsa_routing_is_saved_by_default():
    assert _dsa_caches_routing(_routing_config())


def test_dsa_routing_is_recomputed_when_listed_in_recompute_modules():
    config = _routing_config(
        recompute_granularity="selective", recompute_modules=["dsa_simple_routing"]
    )
    assert not _dsa_caches_routing(config)


def test_dsa_routing_is_saved_when_other_modules_are_recomputed():
    config = _routing_config(recompute_granularity="selective", recompute_modules=["core_attn"])
    assert _dsa_caches_routing(config)


def test_dsa_indexer_k_and_selected_scores_are_saved_by_default():
    config = _routing_config()
    assert _dsa_caches_indexer_k(config)
    assert _dsa_caches_selected_scores(config)


def test_dsa_indexer_k_is_recomputed_when_listed_in_recompute_modules():
    config = _routing_config(recompute_granularity="selective", recompute_modules=["dsa_indexer_k"])
    assert not _dsa_caches_indexer_k(config)
    assert _dsa_caches_selected_scores(config)


def test_dsa_selected_scores_are_recomputed_when_listed_in_recompute_modules():
    config = _routing_config(
        recompute_granularity="selective", recompute_modules=["dsa_selected_scores"]
    )
    assert not _dsa_caches_selected_scores(config)
    assert _dsa_caches_indexer_k(config)


def test_dsa_selected_scores_are_unavailable_without_the_sparse_objective():
    config = _routing_config(dsa_indexer_use_sparse_loss=False)
    assert not _dsa_caches_selected_scores(config)
    assert _dsa_caches_indexer_k(config)


def test_dsa_routing_is_not_saved_for_dense_indexer_warmup():
    config = _routing_config(
        dsa_fwd_use_dense_attn=True, dsa_indexer_use_sparse_loss=False, dsa_indexer_loss_coeff=0.1
    )
    assert not _dsa_caches_routing(config)
    assert not _dsa_caches_indexer_k(config)
    assert not _dsa_caches_selected_scores(config)


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


def test_fused_qk_topk_naive_caps_topk_by_key_length():
    torch.manual_seed(123)

    q = torch.randn(2, 1, 4, 8, dtype=torch.float32)
    k = torch.randn(5, 1, 8, dtype=torch.float32)
    weights = torch.randn(2, 1, 4, dtype=torch.float32)

    _, topk_indices = fused_qk_topk_naive(q=q, k=k, weights=weights, index_topk=4)

    assert topk_indices.shape == (1, 2, 4)
    assert torch.all((topk_indices >= 0) & (topk_indices < 5))


def test_transformer_config_accepts_min_memory_sparse_forward_dense_loss():
    for backend in ("min-memory-triton", "min-memory-torch"):
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
            dsa_kernel_backend=backend,
            dsa_indexer_loss_coeff=0.1,
        )

        assert not config.dsa_fwd_use_dense_attn
        assert not config.dsa_indexer_use_sparse_loss


def _dynamic_packed_seq_params(inference_context: DynamicInferenceContext) -> PackedSeqParams:
    request_count = inference_context.get_active_request_count()
    cu_seqlens_q, max_seqlen_q = inference_context.cu_query_lengths()
    cu_seqlens_kv, _, max_seqlen_kv = inference_context.cu_kv_lengths()
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q[: request_count + 1],
        cu_seqlens_kv=cu_seqlens_kv[: request_count + 1],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        total_tokens=inference_context.active_token_count,
    )


@pytest.mark.skipif(not decode_kernels.HAVE_CUTEDSL, reason="CuTeDSL is required")
class TestDSGQADynamicInference:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture
    def decode_compile_calls(self, monkeypatch):
        for name in ("_COMPILED", "_OPERATORS", "_ATTENTION_COMPILED", "_ATTENTION_OPERATORS"):
            monkeypatch.setattr(decode_kernels, name, {})

        compile_calls = []
        original_compile = decode_kernels.cute.compile

        def count_compile(*args, **kwargs):
            compile_calls.append(1)
            return original_compile(*args, **kwargs)

        monkeypatch.setattr(decode_kernels.cute, "compile", count_compile)
        return compile_calls

    @pytest.mark.parametrize("head_dim", [128, 192, 256])
    def test_paged_decode_kernels_match_reference(self, head_dim):
        torch.manual_seed(17)
        device = torch.device("cuda", torch.cuda.current_device())
        batch, pages_per_request, page_size, heads = 4, 3, 64, 4
        lengths = torch.tensor([3, 63, 65, 137], dtype=torch.int32, device=device)
        block_table = (
            torch.randperm(batch * pages_per_request, device=device)
            .reshape(batch, pages_per_request)
            .to(torch.int32)
        )
        keys = torch.randn(
            batch * pages_per_request, page_size, 1, head_dim, device=device, dtype=torch.bfloat16
        )
        values = torch.randn_like(keys)
        index_q = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
        scores = torch.empty(batch, 151, device=device, dtype=torch.float32)
        decode_kernels.dsa_gqa_decode_indexer_score(
            index_q, keys[:, :, 0], block_table, lengths, out=scores
        )
        expected_scores = torch.full_like(scores, -torch.inf)
        for request, length in enumerate(lengths.tolist()):
            positions = torch.arange(length, device=device)
            k = keys[block_table[request, positions // page_size].long(), positions % page_size, 0]
            expected_scores[request, :length] = k.float() @ index_q[request].float()
        torch.testing.assert_close(scores, expected_scores, atol=2e-4, rtol=2e-4)

        indices = scores.topk(16, dim=-1).indices.to(torch.int32)
        # Attention must accept a query view with the pitched token stride produced by QKV.
        q = torch.randn(batch, heads + 2, head_dim, device=device, dtype=torch.bfloat16)[:, :heads]
        output = torch.empty(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
        scale = head_dim**-0.5
        decode_kernels.dsa_gqa_decode_attention(
            q, keys, values, block_table, indices, lengths, scale, out=output
        )
        expected = torch.empty_like(output)
        for request, length in enumerate(lengths.tolist()):
            selected = indices[request, : min(length, indices.size(1))].long()
            pages = block_table[request, selected // page_size].long()
            k = keys[pages, selected % page_size, 0]
            v = values[pages, selected % page_size, 0]
            probs = (q[request].float() @ k.float().T * scale).softmax(-1).to(torch.bfloat16)
            expected[request] = (probs.float() @ v.float()).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("batch", [1, 3])
    def test_decode_indexer_reuses_compilation_when_score_width_changes(
        self, decode_compile_calls, batch
    ):
        torch.manual_seed(79)
        device = torch.device("cuda", torch.cuda.current_device())
        head_dim, page_size, pages_per_request = 192, 64, 3
        num_pages = batch * pages_per_request
        keys = torch.randn(num_pages, page_size, head_dim, device=device, dtype=torch.bfloat16)
        query = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
        table = (
            torch.randperm(num_pages, device=device).view(batch, pages_per_request).to(torch.int32)
        )
        logical_keys = keys[table.long()].reshape(batch, -1, head_dim)

        # Start at width one (ambiguous strides), cross tile/page boundaries, then
        # shrink. Different request lengths also exercise writes to invalid slots.
        for width in (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 7):
            lengths = torch.tensor(
                [max(width - 2 * i, 0) for i in range(batch)], device=device, dtype=torch.int32
            )
            output = torch.full((batch, width), torch.nan, device=device, dtype=torch.float32)
            decode_kernels.dsa_gqa_decode_indexer_score(query, keys, table, lengths, out=output)
            expected = torch.einsum("bd,bkd->bk", query.float(), logical_keys[:, :width].float())
            expected.masked_fill_(
                torch.arange(width, device=device)[None, :] >= lengths[:, None], -torch.inf
            )
            torch.testing.assert_close(output, expected, atol=2e-4, rtol=2e-4)

        assert len(decode_compile_calls) == 1
        assert len(decode_kernels._COMPILED) == 1

    def test_short_decode_reuses_both_kernels(self, monkeypatch, decode_compile_calls):
        from megatron.core.transformer.experimental_attention_variant import dsa_gqa

        decode_steps = 2
        deepselect_calls = []
        if dsa_gqa._deepselect_decode_supported(torch.device("cuda"), 512):
            original_topk = dsa_gqa.deep_select.topk
            original_score = decode_kernels.dsa_gqa_decode_indexer_score
            score_buffers = []

            def capture_score(*args, **kwargs):
                score_buffers.append(kwargs["out"].data_ptr())
                return original_score(*args, **kwargs)

            def capture_topk(scores, *args, **kwargs):
                # The production score kernel must already provide aligned rows.
                assert scores.dtype == torch.float32
                assert scores.data_ptr() == score_buffers[-1]
                assert (
                    scores.stride(0)
                    * scores.element_size()
                    % (dsa_gqa.deep_select.get_stride_requirement()[0])
                    == 0
                )
                deepselect_calls.append(1)
                return original_topk(scores, *args, **kwargs)

            monkeypatch.setattr(decode_kernels, "dsa_gqa_decode_indexer_score", capture_score)
            monkeypatch.setattr(dsa_gqa.deep_select, "topk", capture_topk)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=1,
            kv_channels=192,
            params_dtype=torch.bfloat16,
            bf16=True,
            normalization="RMSNorm",
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_indexer_n_heads=1,
            dsa_indexer_head_dim=192,
            dsa_indexer_topk=512,
            dsa_indexer_loss_coeff=0.1,
        )
        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=128,
                max_requests=1,
                max_tokens=64,
                block_size_tokens=64,
                buffer_size_gb=0.05,
                num_cuda_graphs=None,
            ),
        )
        context.add_request(
            DynamicInferenceRequest(
                request_id=0,
                prompt_tokens=torch.tensor([1]),
                sampling_params=SamplingParams(num_tokens_to_generate=decode_steps + 1),
            )
        )
        context.initialize_attention_state()

        attention = DSGroupedSelfAttention(
            config=config,
            submodules=dsa_stack_spec.submodules.attention_layer.submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).eval()

        # One prefill, then two decodes to check compilation and score-buffer reuse.
        for step in range(decode_steps + 1):
            hidden = torch.randn(
                context.padded_active_token_count, 1, 4096, device="cuda", dtype=torch.bfloat16
            )
            with torch.inference_mode(), InferenceMode.active():
                attention(hidden_states=hidden, attention_mask=None, inference_context=context)
            if step < decode_steps:
                context.update_requests(
                    active_requests_mask=torch.ones(1, dtype=torch.int32),
                    new_tokens=torch.tensor([step + 2]),
                )
                context.initialize_attention_state()

        assert len(decode_compile_calls) == 2
        assert len(decode_kernels._COMPILED) == 1
        assert len(decode_kernels._ATTENTION_COMPILED) == 1
        if dsa_gqa._deepselect_decode_supported(torch.device("cuda"), 512):
            assert len(deepselect_calls) == decode_steps

    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("fused", [False, True])
    def test_dynamic_query_rope_uses_token_positions_with_padding(self, packed, fused):
        config = TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=2, apply_rope_fusion=fused
        )
        device = torch.device("cuda", torch.cuda.current_device())
        query = torch.randn(8, 1, 2, 64, device=device, dtype=torch.bfloat16)
        # Two decodes followed by a three-token prefill and three padding tokens.
        positions = torch.tensor([13, 9, 0, 1, 2, 0, 0, 0], device=device)
        cu_lengths = torch.tensor([0, 1, 2, 5], device=device, dtype=torch.int32)
        cp_group = parallel_state.get_context_parallel_group()
        rope = RotaryEmbedding(64, rotary_percent=1.0, cp_group=cp_group)
        frequencies = rope(16)
        selected = frequencies[positions]
        first, second = query.float().chunk(2, dim=-1)
        expected = (
            query.float() * selected.cos() + torch.cat((-second, first), dim=-1) * selected.sin()
        ).to(torch.bfloat16)
        context = SimpleNamespace(
            padded_active_token_count=8, gpu_view=SimpleNamespace(token_to_pos_ids=positions)
        )
        actual = DynamicInferenceContext.apply_rotary_emb_query(
            context,
            query.squeeze(1) if packed else query,
            frequencies,
            config,
            cu_lengths if packed else None,
            cp_group,
        )
        torch.testing.assert_close(
            actual, expected.squeeze(1) if packed else expected, atol=2e-2, rtol=2e-2
        )

    @pytest.mark.parametrize(
        "prefill_backend,use_rope,provide_packed_metadata,index_topk",
        [
            ("reference", False, False, 8),
            ("reference", False, True, 32),
            ("reference", True, False, 32),
            ("min-memory-triton", False, True, 8),
            ("min-memory-triton", True, False, 8),
            ("min-memory-triton", True, True, 32),
        ],
    )
    def test_mixed_prefill_reuses_training_and_decode_uses_cute(
        self, monkeypatch, prefill_backend, use_rope, provide_packed_metadata, index_topk
    ):
        dtype = torch.bfloat16
        hidden_size = 4096
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            ffn_hidden_size=16384,
            num_attention_heads=16,
            num_query_groups=1,
            kv_channels=256,
            params_dtype=dtype,
            bf16=True,
            normalization="RMSNorm",
            add_bias_linear=False,
            attention_dropout=0.0,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_kernel_backend=prefill_backend,
            dsa_indexer_n_heads=1,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=index_topk,
            dsa_indexer_loss_coeff=0.1,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
        )
        inference_context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=128,
                max_requests=3,
                max_tokens=64,
                block_size_tokens=64,
                buffer_size_gb=0.05,
                unified_memory_level=0,
                num_cuda_graphs=None,
                use_flashinfer_fused_rope=False,
            ),
        )
        token_offset = 0
        for request_id, prompt_length in enumerate((13, 9)):
            inference_context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=torch.arange(
                        token_offset, token_offset + prompt_length, dtype=torch.long, device="cpu"
                    ),
                    sampling_params=SamplingParams(num_tokens_to_generate=2),
                )
            )
            token_offset += prompt_length
        inference_context.initialize_attention_state()

        attention = DSGroupedSelfAttention(
            config=config,
            submodules=dsa_stack_spec.submodules.attention_layer.submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        attention.eval()
        rope = RotaryEmbedding(
            config.kv_channels, rotary_percent=1.0, cp_group=attention.pg_collection.cp
        )

        def rotary(seq_len):
            return rope(seq_len) if use_rope else None

        def run_step() -> tuple[torch.Tensor, torch.Tensor]:
            hidden_states = torch.randn(
                inference_context.padded_active_token_count,
                1,
                hidden_size,
                dtype=dtype,
                device=torch.cuda.current_device(),
            )
            with torch.inference_mode(), InferenceMode.active():
                output, bias = attention(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    inference_context=inference_context,
                    packed_seq_params=(
                        _dynamic_packed_seq_params(inference_context)
                        if provide_packed_metadata
                        else None
                    ),
                    rotary_pos_emb=rotary(128),
                )
            assert bias is None
            return output, hidden_states

        prefill_output, prompt_hidden_states = run_step()
        assert prefill_output.shape == (inference_context.padded_active_token_count, 1, hidden_size)

        inference_context.update_requests(
            active_requests_mask=torch.ones(2, dtype=torch.int32),
            new_tokens=torch.tensor([100, 101], dtype=torch.long),
        )
        inference_context.add_request(
            DynamicInferenceRequest(
                request_id=2,
                prompt_tokens=torch.arange(
                    token_offset, token_offset + 5, dtype=torch.long, device="cpu"
                ),
                sampling_params=SamplingParams(num_tokens_to_generate=2),
            )
        )
        inference_context.initialize_attention_state()
        assert inference_context.num_decode_requests == 2
        assert inference_context.num_prefill_requests == 1

        indexer_calls = []
        attention_calls = []
        original_indexer = decode_kernels.dsa_gqa_decode_indexer_score
        original_attention = decode_kernels.dsa_gqa_decode_attention

        def capture_indexer(*args, **kwargs):
            indexer_calls.append(tuple(args[0].shape))
            return original_indexer(*args, **kwargs)

        def capture_attention(*args, **kwargs):
            attention_calls.append((tuple(args[0].shape), tuple(args[4].shape)))
            return original_attention(*args, **kwargs)

        monkeypatch.setattr(decode_kernels, "dsa_gqa_decode_indexer_score", capture_indexer)
        monkeypatch.setattr(decode_kernels, "dsa_gqa_decode_attention", capture_attention)
        core_attention = attention.core_attention
        training_forward = core_attention.forward
        prefill_calls = []

        def capture_prefill(*args, **kwargs):
            prefill_calls.append(kwargs["query"].size(0))
            return training_forward(*args, **kwargs)

        monkeypatch.setattr(core_attention, "forward", capture_prefill)
        mixed_output, mixed_hidden_states = run_step()

        assert indexer_calls == [(2, 128)]
        assert attention_calls == [((2, 16, 256), (2, index_topk))]
        assert prefill_calls == [5]
        assert mixed_output.shape == (inference_context.padded_active_token_count, 1, hidden_size)
        assert mixed_output.dtype == dtype

        monkeypatch.setattr(core_attention, "forward", training_forward)
        prompt_start = 0
        for request_idx, prompt_length in enumerate((13, 9)):
            request_hidden_states = torch.cat(
                (
                    prompt_hidden_states[prompt_start : prompt_start + prompt_length],
                    mixed_hidden_states[request_idx : request_idx + 1],
                )
            )
            with torch.inference_mode():
                training_output, bias = attention(
                    hidden_states=request_hidden_states,
                    attention_mask=None,
                    rotary_pos_emb=rotary(request_hidden_states.size(0)),
                )
            assert bias is None
            torch.testing.assert_close(
                mixed_output[request_idx], training_output[-1], rtol=2e-2, atol=2e-2
            )
            prompt_start += prompt_length

        # All three requests now decode, exercising cache reuse across a second step.
        inference_context.update_requests(
            active_requests_mask=torch.ones(3, dtype=torch.int32),
            new_tokens=torch.tensor([102, 103, 104], dtype=torch.long),
        )
        inference_context.initialize_attention_state()
        assert inference_context.num_decode_requests == 3
        assert inference_context.num_prefill_requests == 0
        monkeypatch.setattr(core_attention, "forward", capture_prefill)
        prefill_calls.clear()
        decode_output, decode_hidden_states = run_step()
        assert prefill_calls == []
        assert indexer_calls[-1] == (3, 128)
        assert attention_calls[-1] == ((3, 16, 256), (3, index_topk))
        monkeypatch.setattr(core_attention, "forward", training_forward)
        histories = [
            torch.cat((prompt_hidden_states[:13], mixed_hidden_states[:1])),
            torch.cat((prompt_hidden_states[13:22], mixed_hidden_states[1:2])),
            mixed_hidden_states[2:7],
        ]
        for request_idx, history in enumerate(histories):
            full_prefix = torch.cat((history, decode_hidden_states[request_idx : request_idx + 1]))
            with torch.inference_mode():
                expected, _ = attention(
                    hidden_states=full_prefix,
                    attention_mask=None,
                    rotary_pos_emb=rotary(full_prefix.size(0)),
                )
            torch.testing.assert_close(
                decode_output[request_idx], expected[-1], rtol=2e-2, atol=2e-2
            )

    @pytest.mark.parametrize("backend", ["reference", "min-memory-triton"])
    @pytest.mark.parametrize("topk", [8, 512])
    @pytest.mark.parametrize(
        "head_dim,index_head_dim,rotary_base,rotary_percent",
        [(128, 128, 10000, 1.0), (192, 128, 500000, 0.5), (256, 64, 1000000, 0.5)],
    )
    def test_decode_logprobs_match_training(
        self, monkeypatch, backend, topk, head_dim, index_head_dim, rotary_base, rotary_percent
    ):
        """The same weights and token histories give close full-vocabulary logprobs."""
        torch.manual_seed(53)
        vocab_size, decode_steps = 1024, 8
        prompt_lengths = (31, 63, 513)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=512,
            ffn_hidden_size=1024,
            num_attention_heads=4,
            num_query_groups=1,
            kv_channels=head_dim,
            params_dtype=torch.bfloat16,
            bf16=True,
            normalization="RMSNorm",
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            is_hybrid_model=True,
            experimental_attention_variant="dsa",
            dsa_indexer_mode="simplified",
            dsa_kernel_backend=backend,
            dsa_indexer_n_heads=1,
            dsa_indexer_head_dim=index_head_dim,
            dsa_indexer_topk=topk,
            dsa_indexer_loss_coeff=0.1,
        )
        # Match the model-wide settings passed to the indexer by argument_utils.
        config.rope_type = "rope"
        config.rotary_base = rotary_base
        config.rotary_percent = rotary_percent
        model = HybridModel(
            config=config,
            hybrid_stack_spec=dsa_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=640,
            hybrid_layer_pattern="*-*-",
            position_embedding_type="rope",
            rotary_base=rotary_base,
            rotary_percent=rotary_percent,
            parallel_output=False,
        ).cuda()
        tokens = [
            torch.randint(vocab_size, (length + decode_steps + 1,), device="cuda")
            for length in prompt_lengths
        ]

        # Exercise the actual training branch: gradients enabled, dropout disabled,
        # indexer loss enabled, and no inference context. Causal attention lets one
        # teacher-forced forward supply references for every later decode step.
        model.train()
        training_logprobs = []
        with torch.enable_grad():
            for sequence in tokens:
                inputs = sequence[:-1].unsqueeze(0)
                logits = model(
                    input_ids=inputs,
                    position_ids=torch.arange(inputs.size(1), device="cuda").unsqueeze(0),
                    attention_mask=None,
                    runtime_gather_output=True,
                )
                assert logits.requires_grad
                training_logprobs.append(logits.detach().float().log_softmax(-1)[0])
                del logits

        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=640,
                max_requests=3,
                max_tokens=1024,
                block_size_tokens=64,
                buffer_size_gb=0.05,
                unified_memory_level=0,
                num_cuda_graphs=None,
                materialize_only_last_token_logits=True,
                use_flashinfer_fused_rope=False,
            ),
        )
        for request_id, length in enumerate(prompt_lengths):
            context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=tokens[request_id][:length].cpu(),
                    sampling_params=SamplingParams(num_tokens_to_generate=decode_steps + 1),
                )
            )

        decode_calls = []
        original_decode = decode_kernels.dsa_gqa_decode_attention

        def capture_decode(*args, **kwargs):
            decode_calls.append(1)
            return original_decode(*args, **kwargs)

        monkeypatch.setattr(decode_kernels, "dsa_gqa_decode_attention", capture_decode)
        model.eval()
        errors = []
        with torch.inference_mode(), InferenceMode.active():
            for step in range(decode_steps + 1):
                context.initialize_attention_state()
                input_ids, position_ids = context.current_input_and_position_ids()
                # Allocating a new KV page can reorder active requests. Align both
                # references and teacher-forced tokens with the current request IDs.
                assert context.paused_request_count == 0
                request_ids = context.request_ids[: context.total_request_count].tolist()
                if step:
                    expected_positions = torch.tensor(
                        [prompt_lengths[request_id] - 1 + step for request_id in request_ids],
                        device="cuda",
                    )
                    expected_inputs = torch.stack(
                        [
                            tokens[request_id][prompt_lengths[request_id] - 1 + step]
                            for request_id in request_ids
                        ]
                    )
                    torch.testing.assert_close(position_ids[0], expected_positions)
                    torch.testing.assert_close(input_ids[0], expected_inputs)
                logits = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                    inference_context=context,
                    runtime_gather_output=True,
                )
                assert logits.shape == (1, len(prompt_lengths), vocab_size)
                actual = logits[0].float().log_softmax(-1)
                expected = torch.stack(
                    [
                        training_logprobs[request_id][prompt_lengths[request_id] - 1 + step]
                        for request_id in request_ids
                    ]
                )
                # Check every request and vocabulary token at every step. An absolute
                # tolerance of 0.01 nats allows about 1% probability error per element.
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0,
                    atol=1e-2,
                    msg=lambda message: f"Step {step}, request IDs {request_ids}:\n{message}",
                )
                delta = (actual - expected).abs()
                errors.append(delta)
                if step < decode_steps:
                    # Feed the training tokens, so sampling cannot change the histories.
                    targets = torch.stack(
                        [
                            tokens[request_id][prompt_lengths[request_id] + step]
                            for request_id in request_ids
                        ]
                    )
                    context.update_requests(
                        active_requests_mask=torch.ones(len(tokens), dtype=torch.int32),
                        new_tokens=targets.cpu(),
                    )

        assert len(decode_calls) == 2 * decode_steps
        # Keep the aggregate error budget in addition to the elementwise checks above.
        mean_abs_error = torch.stack(errors).mean().item()
        assert mean_abs_error <= 2e-3, f"Mean logprob error: {mean_abs_error} nats"
