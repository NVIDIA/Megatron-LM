# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native CSA2 parity, cross-layer sharing, and training-stack integration.

Reference: deepseek-ai/DeepSeek-V4.1-Flash, revision
dba1be0a40aa45a94ad051997016db3960a90277, inference/model.py.
The oracle uses functional projections, adjacent-pair RoPE, and dense masked attention;
it does not call the implementation's compressor, indexer, or sparse attention helpers.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
    CSA2State,
    select_candidate_blocks,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config


@pytest.fixture
def pg_collection():
    # Native indexer-loss tests below do not need CUDA or Transformer Engine.
    if not torch.cuda.is_available():
        pytest.skip("Megatron RoPE requires CUDA")
    if not HAVE_TE:
        pytest.skip("Transformer Engine is not installed")
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


def _layer(pg_collection, ratio, dtype=torch.float32, **overrides):
    config_values = dict(
        num_layers=1,
        csa_compress_ratios=[ratio],
        csa2_kv_source_layers=[0] if ratio else [],
        csa2_index_source_layers=[0] if ratio else [],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_indexer_topk=2,
    )
    config_values.update(overrides)
    config = _make_config(params_dtype=dtype, **config_values)
    layer = build_module(
        get_experimental_attention_variant_module_spec(config),
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    ).cuda()
    # Nontrivial query magnitudes, gates, and sink expose normalization/softmax mistakes.
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "norm.weight" in name:
                param.uniform_(0.7, 1.3)
            else:
                param.normal_(std=0.2)
    return layer


def _reference_rope(x, config, ratio, position_stride=1, inverse=False):
    dim = config.qk_pos_emb_head_dim
    base = config.csa_compress_rotary_base if ratio else config.rotary_base
    freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    if ratio:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(config.original_max_position_embeddings / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(config.beta_fast)), 0)
        high = min(math.ceil(corrected_dim(config.beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, device=x.device, dtype=torch.float32) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        freq = freq / config.rotary_scaling_factor * ramp + freq * (1 - ramp)
    positions = torch.arange(x.shape[0], device=x.device, dtype=torch.float32) * position_stride
    angles = torch.outer(positions, freq)
    # Preserve V4's RoPE precision: cast trig values to the activation dtype before
    # the two products and their sum, instead of promoting the activation to FP32.
    shape = (x.shape[0], *([1] * (x.ndim - 2)), dim // 2)
    cos = angles.cos().to(x.dtype).reshape(shape)
    sin = angles.sin().to(x.dtype).reshape(shape)
    if inverse:
        sin = -sin
    even, odd = x[..., -dim::2], x[..., -dim + 1 :: 2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((x[..., :-dim], rotated), dim=-1)


def _reference_norm(x, weight, eps):
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt() * weight).to(x.dtype)


def _reference(x, weights, config, ratio):
    """Dense oracle including the grouped output projection and indexer score graph."""

    def linear(tensor, name):
        return F.linear(tensor, weights[name + ".weight"])

    def norm(tensor, name):
        return _reference_norm(
            tensor, weights[name + ".weight"], config.attention_latent_norm_epsilon
        )

    seq_len, batch = x.shape[:2]
    qr = norm(linear(x, "linear_q_down_proj"), "q_layernorm")
    q = linear(qr, "linear_q_up_proj").reshape(
        seq_len, batch, config.num_attention_heads, config.v_head_dim
    )
    q = _reference_rope(q, config, ratio)
    local_kv = _reference_rope(norm(linear(x, "linear_kv_proj"), "kv_layernorm"), config, ratio)
    qi = torch.arange(seq_len, device=x.device)[:, None]
    ki = torch.arange(seq_len, device=x.device)[None, :]
    visible = ((ki <= qi) & (ki > qi - config.csa_window_size)).expand(batch, -1, -1)
    kv = local_kv
    index_scores = None
    if ratio:
        prefix = "core_attention.compressor."
        if ratio == 1:
            latent = linear(x, prefix + "linear_wkv")
        else:
            # Stack whole groups explicitly, without the implementation's reshape/pooling path.
            groups = []
            for start in range(0, seq_len - ratio + 1, ratio):
                tokens = x[start : start + ratio].float()
                projected = linear(tokens, prefix + "linear_wkv")
                logits = linear(tokens, prefix + "linear_wgate")
                groups.append((projected * logits.softmax(dim=0)).sum(dim=0))
            if groups:
                latent = torch.stack(groups).to(x.dtype)
            else:
                # Preserve zero derivatives to both projections when no group is complete.
                latent = (
                    linear(x[:0].float(), prefix + "linear_wkv")
                    + linear(x[:0].float(), prefix + "linear_wgate")
                ).to(x.dtype)
        latent = norm(latent, prefix + "norm")
        prefix = "core_attention.indexer."
        index_q = linear(qr, prefix + "linear_wq_b").reshape(
            seq_len, batch, config.dsa_indexer_n_heads, config.dsa_indexer_head_dim
        )
        index_q = _reference_rope(index_q, config, ratio)
        index_k = norm(linear(latent, prefix + "linear_wk"), prefix + "k_norm")
        index_k = _reference_rope(index_k, config, ratio, position_stride=ratio)
        index_weights = linear(x, prefix + "linear_weights_proj").float()
        index_scores = torch.einsum("sbhd,tbd->bsht", index_q.float(), index_k.float()).relu()
        index_scores = (
            index_scores
            * index_weights.permute(1, 0, 2).unsqueeze(-1)
            * (config.dsa_indexer_head_dim**-0.5 * config.dsa_indexer_n_heads**-0.5)
        ).sum(dim=2)
        global_visible = torch.arange(latent.shape[0], device=x.device) < (qi + 1) // ratio
        index_scores = index_scores.masked_fill(~global_visible, -torch.inf)
        selected = index_scores.topk(min(config.dsa_indexer_topk, latent.shape[0]), dim=-1).indices
        selected_mask = torch.zeros_like(index_scores, dtype=torch.bool).scatter(-1, selected, True)
        visible = torch.cat((visible, selected_mask & global_visible), dim=-1)
        kv = torch.cat((kv, _reference_rope(latent, config, ratio, position_stride=ratio)))
    # K and V are the same latent. Combine both gradient contributions in FP32 before
    # casting back, just as for the FP32 attention computation in the sparse path.
    kv_float = kv.float()
    scores = torch.einsum("sbhd,tbd->bsht", q.float(), kv_float) * config.v_head_dim**-0.5
    scores = scores.masked_fill(~visible.unsqueeze(2), -torch.inf)
    sink = weights["core_attention.attn_sink"].reshape(1, 1, -1, 1).expand(batch, seq_len, -1, 1)
    probs = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
    output = torch.einsum("bsht,tbd->sbhd", probs, kv_float).to(x.dtype)
    output = _reference_rope(output, config, ratio, inverse=True)
    output = output.reshape(seq_len, batch, config.o_groups, -1)
    group_weight = weights["linear_o_group_proj"].reshape(config.o_groups, config.o_lora_rank, -1)
    # One independent projection per group; use matmul instead of the wrapper's einsum.
    output = torch.stack(
        [F.linear(output[:, :, group], group_weight[group]) for group in range(config.o_groups)],
        dim=2,
    ).flatten(-2)
    return linear(output, "linear_proj"), index_scores


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("seq_len", [1, 3, 5, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_and_backward_match_dense_reference(pg_collection, ratio, seq_len, dtype):
    torch.manual_seed(1234)
    layer = _layer(pg_collection, ratio, dtype)
    x = torch.randn(
        seq_len, 2, layer.config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = {
        name: p.detach().clone().requires_grad_() for name, p in layer.named_parameters()
    }
    actual, bias = layer(x, None)
    expected, _ = _reference(ref_x, ref_weights, layer.config, ratio)
    assert bias is None
    tol = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=1.5e-2, rtol=3e-2)
    torch.testing.assert_close(actual, expected, **tol)
    grad = torch.randn_like(actual)
    (actual * grad).sum().backward()
    (expected * grad).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, **tol)
    for name, param in layer.named_parameters():
        if "indexer." in name:
            # Discrete selection intentionally does not imply an indexer training objective.
            assert param.grad is None and ref_weights[name].grad is None
        else:
            assert param.grad is not None, name
            torch.testing.assert_close(
                param.grad, ref_weights[name].grad, msg=lambda msg: f"{name}: {msg}", **tol
            )
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexer_scores_and_auxiliary_gradients(pg_collection, ratio, dtype):
    torch.manual_seed(42)
    layer = _layer(pg_collection, ratio, dtype)
    x = torch.randn(9, 2, layer.config.hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = {
        name: p.detach().clone().requires_grad_() for name, p in layer.named_parameters()
    }
    qr = layer.q_layernorm(layer.linear_q_down_proj(x)[0])
    latent = layer.core_attention.compressor(x)
    scores = layer.core_attention.indexer.forward_before_topk(x, qr, latent, layer.rotary_pos_emb)
    _, ref_scores = _reference(ref_x, ref_weights, layer.config, ratio)
    tol = dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(scores, ref_scores, **tol)
    finite = scores.isfinite()
    scores[finite].square().mean().backward()
    ref_scores[finite].square().mean().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, **tol)
    for name, param in layer.named_parameters():
        if "indexer." in name:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, name
            torch.testing.assert_close(
                param.grad, ref_weights[name].grad, msg=lambda msg: f"{name}: {msg}", **tol
            )


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_causality_and_multiple_live_microbatches(pg_collection, ratio):
    torch.manual_seed(7)
    layer = _layer(pg_collection, ratio)
    x = torch.randn(9, 2, layer.config.hidden_size, device="cuda", requires_grad=True)
    changed = x.detach().clone()
    changed[5:] = torch.randn_like(changed[5:]) * 10
    changed.requires_grad_()
    first, _ = layer(x, None)
    second, _ = layer(changed, None)
    # Query 4 cannot see token 5, which would complete its ratio-2 global group.
    torch.testing.assert_close(first[:5], second[:5], atol=1e-6, rtol=1e-5)
    grad_x, grad_changed = torch.autograd.grad(first[:5].sum() + second[:5].sum(), (x, changed))
    assert torch.count_nonzero(grad_x[5:]) == 0
    assert torch.count_nonzero(grad_changed[5:]) == 0
    assert torch.isfinite(grad_x).all() and torch.isfinite(grad_changed).all()


def test_bf16_conversion_keeps_compressor_and_sink_in_fp32(pg_collection):
    layer = _layer(pg_collection, 2)
    convert_module_to_dtype_except_fp32_marked(layer, torch.bfloat16)
    core = layer.core_attention
    assert core.attn_sink.dtype == torch.float32
    assert core.compressor.linear_wkv.weight.dtype == torch.float32
    assert core.compressor.linear_wgate.weight.dtype == torch.float32
    assert core.compressor.norm.weight.dtype == torch.bfloat16
    assert layer.linear_q_up_proj.weight.dtype == torch.bfloat16
    ratio_one = _layer(pg_collection, 1, torch.bfloat16).core_attention.compressor
    assert ratio_one.linear_wkv.weight.dtype == torch.bfloat16
    assert ratio_one.linear_wgate is None


@pytest.mark.parametrize("layer_number", [3, 5, 6])
def test_cross_layer_modes_require_forward_state(pg_collection, layer_number):
    config = _make_config()
    layer = build_module(
        get_experimental_attention_variant_module_spec(config),
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
    ).cuda()
    x = torch.randn(5, 2, config.hidden_size, device="cuda")
    with pytest.raises(ValueError, match="explicit csa2_state"):
        layer(x, None)


def test_candidate_source_can_run_standalone(pg_collection):
    config = _make_config()
    layer = build_module(
        get_experimental_attention_variant_module_spec(config),
        config=config,
        layer_number=4,
        pg_collection=pg_collection,
    ).cuda()
    x = torch.randn(5, 2, config.hidden_size, device="cuda", requires_grad=True)
    output, _ = layer(x, None)
    output.float().square().mean().backward()
    assert output.shape == x.shape
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()


def test_unimplemented_dropout_fails_clearly(pg_collection):
    with pytest.raises(NotImplementedError, match="dropout"):
        _layer(pg_collection, 2, attention_dropout=0.1)


def test_causal_mask_and_unsupported_packing(pg_collection):
    layer = _layer(pg_collection, 2)
    x = torch.randn(5, 2, layer.config.hidden_size, device="cuda")
    mask = torch.ones(1, 1, 5, 5, device="cuda", dtype=torch.bool).triu(1)
    torch.testing.assert_close(layer(x, mask)[0], layer(x, None)[0])
    mask[..., 3, 0] = True
    with pytest.raises(NotImplementedError, match="ordinary causal"):
        layer(x, mask)
    with pytest.raises(NotImplementedError, match="unpacked"):
        layer(x, None, packed_seq_params=object())


# Cross-layer sharing and hierarchical candidate selection


def _layers(pg_collection, dtype=torch.float32, candidates=True):
    overrides = (
        {}
        if candidates
        else dict(
            csa2_candidate_source_layer=None,
            csa2_candidate_topk_blocks=0,
            csa2_candidate_block_size=0,
        )
    )
    config = _make_config(params_dtype=dtype, dsa_indexer_topk=2, **overrides)
    spec = get_experimental_attention_variant_module_spec(config)
    layers = torch.nn.ModuleList(
        build_module(spec, config=config, layer_number=i + 1, pg_collection=pg_collection)
        for i in range(config.num_layers)
    ).cuda()
    with torch.no_grad():
        for name, parameter in layers.named_parameters():
            if "norm.weight" in name:
                parameter.uniform_(0.7, 1.3)
            else:
                parameter.normal_(std=0.2)
    return layers


def _reference_candidates(scores, ratio, blocks_to_keep, block_size):
    """Rank explicit slices, including the newest partly visible block."""
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    blocks = [
        scores[..., start : start + block_size].amax(-1) for start in range(0, width, block_size)
    ]
    block_scores = torch.stack(blocks, -1)
    visible = torch.arange(1, scores.shape[1] + 1, device=scores.device) // ratio
    newest = (visible - 1) // block_size
    for block in range(len(blocks)):
        block_scores[..., block] = torch.where(newest == block, torch.inf, block_scores[..., block])
    top = block_scores.topk(min(blocks_to_keep, len(blocks)), dim=-1)
    selected = torch.zeros_like(block_scores, dtype=torch.bool).scatter(
        -1, top.indices, top.values > -torch.inf
    )
    return torch.stack([selected[..., position // block_size] for position in range(width)], -1)


def _reference_layer(x, weights, config, layer_idx, state):
    def linear(tensor, name):
        return F.linear(tensor, weights[name + ".weight"])

    def norm(tensor, name):
        return _reference_norm(
            tensor, weights[name + ".weight"], config.attention_latent_norm_epsilon
        )

    seq_len, batch = x.shape[:2]
    ratio = config.csa_compress_ratios[layer_idx]
    qr = norm(linear(x, "linear_q_down_proj"), "q_layernorm")
    q = linear(qr, "linear_q_up_proj").reshape(
        seq_len, batch, config.num_attention_heads, config.v_head_dim
    )
    q = _reference_rope(q, config, ratio)
    kv = _reference_rope(norm(linear(x, "linear_kv_proj"), "kv_layernorm"), config, ratio)
    query_positions = torch.arange(seq_len, device=x.device)[:, None]
    local_positions = torch.arange(seq_len, device=x.device)[None, :]
    visible = (local_positions <= query_positions) & (
        local_positions > query_positions - config.csa_window_size
    )
    visible = visible.expand(batch, -1, -1)
    if ratio:
        if layer_idx in config.csa2_kv_source_layers:
            prefix = "core_attention.compressor."
            if ratio == 1:
                latent = linear(x, prefix + "linear_wkv")
            else:
                groups = []
                for start in range(0, seq_len - ratio + 1, ratio):
                    tokens = x[start : start + ratio].float()
                    projected = linear(tokens, prefix + "linear_wkv")
                    gate = linear(tokens, prefix + "linear_wgate").softmax(0)
                    groups.append((projected * gate).sum(0))
                latent = (
                    torch.stack(groups)
                    if groups
                    else (
                        linear(x[:0].float(), prefix + "linear_wkv")
                        + linear(x[:0].float(), prefix + "linear_wgate")
                    )
                ).to(x.dtype)
            latent = norm(latent, prefix + "norm")
            state["global_kv"] = _reference_rope(latent, config, ratio, position_stride=ratio)
            prefix = "core_attention.indexer."
            index_k = norm(linear(latent, prefix + "linear_wk"), prefix + "k_norm")
            state["indexer_k"] = _reference_rope(index_k, config, ratio, position_stride=ratio)
        global_len = state["global_kv"].shape[0]
        causal = torch.arange(global_len, device=x.device) < (query_positions + 1) // ratio
        if layer_idx in config.csa2_index_source_layers:
            prefix = "core_attention.indexer."
            q_index = linear(qr, prefix + "linear_wq_b").reshape(
                seq_len, batch, config.dsa_indexer_n_heads, config.dsa_indexer_head_dim
            )
            q_index = _reference_rope(q_index, config, ratio)
            scores = torch.einsum(
                "sbhd,tbd->bsht", q_index.float(), state["indexer_k"].float()
            ).relu()
            weights_index = linear(x, prefix + "linear_weights_proj").float()
            weights_index = weights_index * (
                config.dsa_indexer_head_dim**-0.5 * config.dsa_indexer_n_heads**-0.5
            )
            scores = (scores * weights_index.permute(1, 0, 2).unsqueeze(-1)).sum(2)
            scores = scores.masked_fill(~causal, -torch.inf)
            candidate_owner = config.csa2_candidate_source_layer
            if candidate_owner == layer_idx:
                state["candidates"] = _reference_candidates(
                    scores,
                    ratio,
                    config.csa2_candidate_topk_blocks,
                    config.csa2_candidate_block_size,
                )
            elif candidate_owner is not None and candidate_owner < layer_idx:
                scores = scores.masked_fill(~state["candidates"], -torch.inf)
            state["scores"] = scores
            top = scores.topk(min(config.dsa_indexer_topk, global_len), dim=-1)
            state["selected"] = torch.zeros_like(scores, dtype=torch.bool).scatter(
                -1, top.indices, top.values.isfinite()
            )
        visible = torch.cat((visible, state["selected"] & causal), -1)
        kv = torch.cat((kv, state["global_kv"]), 0)
    kv_float = kv.float()
    scores = torch.einsum("sbhd,tbd->bsht", q.float(), kv_float) * config.v_head_dim**-0.5
    scores = scores.masked_fill(~visible.unsqueeze(2), -torch.inf)
    sink = weights["core_attention.attn_sink"].reshape(1, 1, -1, 1).expand(batch, seq_len, -1, 1)
    probabilities = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    output = torch.einsum("bsht,tbd->sbhd", probabilities, kv_float).to(x.dtype)
    output = _reference_rope(output, config, ratio, inverse=True).reshape(
        seq_len, batch, config.o_groups, -1
    )
    group_weights = weights["linear_o_group_proj"].reshape(config.o_groups, config.o_lora_rank, -1)
    output = torch.stack(
        [F.linear(output[:, :, group], group_weights[group]) for group in range(config.o_groups)], 2
    ).flatten(-2)
    return linear(output, "linear_proj")


def _stack(layers, x, state):
    # A residual attention stack isolates sharing from the still-separate mHC work.
    outputs = []
    for layer in layers:
        output, bias = layer(x, None, csa2_state=state)
        assert bias is None
        outputs.append(output)
        x = x + output * 0.2
    return x, outputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 9])
@pytest.mark.parametrize("candidates", [False, True])
def test_shared_stack_matches_dense_reference(pg_collection, dtype, seq_len, candidates):
    torch.manual_seed(1234)
    layers = _layers(pg_collection, dtype, candidates)
    x = torch.randn(
        seq_len, 2, layers[0].config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = [
        {name: p.detach().clone().requires_grad_() for name, p in layer.named_parameters()}
        for layer in layers
    ]
    actual, outputs = _stack(layers, x, CSA2State())
    expected, ref_state = ref_x, {}
    tol = dict(atol=5e-6, rtol=5e-5) if dtype == torch.float32 else dict(atol=3e-2, rtol=6e-2)
    for layer_idx, (layer, weights) in enumerate(zip(layers, ref_weights)):
        ref_output = _reference_layer(expected, weights, layer.config, layer_idx, ref_state)
        torch.testing.assert_close(outputs[layer_idx], ref_output, **tol)
        expected = expected + ref_output * 0.2
    torch.testing.assert_close(actual, expected, **tol)
    gradient = torch.randn_like(actual)
    (actual * gradient).sum().backward()
    (expected * gradient).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, **tol)
    for layer_idx, (layer, weights) in enumerate(zip(layers, ref_weights)):
        for name, parameter in layer.named_parameters():
            if "indexer." in name:
                assert parameter.grad is None and weights[name].grad is None
            else:
                assert parameter.grad is not None, (layer_idx, name)
                torch.testing.assert_close(
                    parameter.grad,
                    weights[name].grad,
                    msg=lambda msg: f"{layer_idx}.{name}: {msg}",
                    **tol,
                )


def test_only_full_owns_kv_and_reindex_owns_only_query(pg_collection):
    layers = _layers(pg_collection)
    for layer_idx, layer in enumerate(layers):
        core = layer.core_attention
        assert (core.compressor is not None) == (layer_idx in [1, 3])
        assert (core.indexer is not None) == (layer_idx in [1, 3, 4])
        if core.indexer is not None:
            names = dict(core.indexer.named_parameters())
            assert ("linear_wk.weight" in names) == (layer_idx in [1, 3])
            assert ("k_norm.weight" in names) == (layer_idx in [1, 3])
            assert "linear_wq_b.weight" in names and "linear_weights_proj.weight" in names
    x = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    state = CSA2State()
    layers[1](x, None, csa2_state=state)
    owner_kv, owner_index_k, owner_indices = state.global_kv, state.indexer_k, state.global_indices
    layers[2](x, None, csa2_state=state)
    assert state.global_kv is owner_kv and state.indexer_k is owner_index_k
    assert state.global_indices is owner_indices
    layers[3](x, None, csa2_state=state)
    decoder_kv, decoder_index_k = state.global_kv, state.indexer_k
    assert decoder_kv is not owner_kv and decoder_index_k is not owner_index_k
    assert decoder_kv.grad_fn is not None and decoder_index_k.grad_fn is not None
    layers[4](x, None, csa2_state=state)
    assert state.global_kv is decoder_kv and state.indexer_k is decoder_index_k
    assert state.kv_source_layer == 3 and state.index_source_layer == 4


def test_reuse_consumer_gradient_reaches_full_owner(pg_collection):
    torch.manual_seed(47)
    layers = _layers(pg_collection, candidates=False)
    source = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    consumer = torch.randn_like(source, requires_grad=True)
    state = CSA2State()
    # Deliberately discard Full's output; the only path to source is the shared global KV.
    layers[1](source, None, csa2_state=state)
    reused, _ = layers[2](consumer, None, csa2_state=state)
    reused.square().sum().backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert consumer.grad is not None and consumer.grad.abs().sum() > 0
    owner = layers[1].core_attention.compressor
    for parameter in owner.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert layers[1].linear_q_up_proj.weight.grad is None
    # Position 8 belongs to an incomplete ratio-2 group and has no source path here.
    assert torch.count_nonzero(source.grad[-1]) == 0


@torch.no_grad()
def test_shared_state_rejects_stale_order_missing_owners_and_shape_changes(pg_collection):
    layers = _layers(pg_collection)
    x = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda")
    state = CSA2State()
    layers[1](x, None, csa2_state=state)
    with pytest.raises(ValueError, match="increasing order.*fresh CSA2State"):
        layers[1](x, None, csa2_state=state)
    # Reindex cannot run before its Full source, even when another Full populated state.
    with pytest.raises(ValueError, match="KV and indexer K from Full layer 3"):
        layers[4](x, None, csa2_state=CSA2State())
    with pytest.raises(ValueError, match="KV and indexer K from Full layer 3"):
        layers[4](x, None, csa2_state=state)
    layers[3](x, None, csa2_state=state)
    # The decoder Reuse layer needs the latest Reindex selection, not Full's selection.
    with pytest.raises(ValueError, match="requires indices from layer 4"):
        layers[5](x, None, csa2_state=state)
    with pytest.raises(ValueError, match="sequence length, batch size, device and dtype"):
        layers[4](x[:-1], None, csa2_state=state)


def test_two_live_forwards_preserve_causality_and_independent_graphs(pg_collection):
    torch.manual_seed(17)
    layers = _layers(pg_collection)
    x = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    changed = x.detach().clone()
    changed[5:] = torch.randn_like(changed[5:]) * 10
    changed.requires_grad_()
    state, changed_state = CSA2State(), CSA2State()
    first, _ = _stack(layers, x, state)
    second, _ = _stack(layers, changed, changed_state)
    assert state.global_kv is not changed_state.global_kv
    assert state.global_indices is not changed_state.global_indices
    torch.testing.assert_close(first[:5], second[:5], atol=5e-6, rtol=5e-5)
    (first[:5].sum() + second[:5].sum()).backward()
    assert torch.count_nonzero(x.grad[5:]) == 0
    assert torch.count_nonzero(changed.grad[5:]) == 0
    # TE linears release backward context after one traversal. Compare with fresh, isolated
    # forwards so every graph is differentiated once, including the two live graphs above.
    for original in (x, changed):
        isolated = original.detach().clone().requires_grad_()
        output, _ = _stack(layers, isolated, CSA2State())
        expected_gradient = torch.autograd.grad(output[:5].sum(), isolated)[0]
        torch.testing.assert_close(original.grad, expected_gradient)


def test_reindex_scores_preserve_owner_key_graph_and_candidate_mask(pg_collection):
    torch.manual_seed(24)
    layers = _layers(pg_collection)
    source = torch.randn(9, 2, layers[0].config.hidden_size, device="cuda", requires_grad=True)
    consumer = torch.randn_like(source, requires_grad=True)
    ref_source = source.detach().clone().requires_grad_()
    ref_consumer = consumer.detach().clone().requires_grad_()
    owner, reindex = layers[3], layers[4]
    ref_owner_weights = {
        name: p.detach().clone().requires_grad_() for name, p in owner.named_parameters()
    }
    ref_reindex_weights = {
        name: p.detach().clone().requires_grad_() for name, p in reindex.named_parameters()
    }
    state, ref_state = CSA2State(), {}
    owner(source, None, csa2_state=state)
    _reference_layer(ref_source, ref_owner_weights, owner.config, 3, ref_state)
    _reference_layer(ref_consumer, ref_reindex_weights, reindex.config, 4, ref_state)
    qr = reindex.q_layernorm(reindex.linear_q_down_proj(consumer)[0])
    indexer = reindex.core_attention.indexer
    scores = indexer.forward_before_topk(
        consumer,
        qr,
        None,
        reindex.rotary_pos_emb,
        indexer_k=state.indexer_k,
        candidates=state.candidates,
    )
    torch.testing.assert_close(scores, ref_state["scores"], atol=3e-6, rtol=3e-5)
    assert (~state.candidates[:, -1]).any(), "The case must actually exclude old global positions."
    assert torch.isneginf(scores.masked_select(~state.candidates)).all()
    selected = indexer.select_indices(scores)
    valid = selected >= 0
    assert state.candidates.gather(-1, selected.clamp_min(0).long())[valid].all()
    finite = scores.isfinite()
    scores[finite].square().mean().backward()
    ref_state["scores"][finite].square().mean().backward()
    torch.testing.assert_close(source.grad, ref_source.grad, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(consumer.grad, ref_consumer.grad, atol=3e-6, rtol=3e-5)
    for layer, ref_weights in [(owner, ref_owner_weights), (reindex, ref_reindex_weights)]:
        for name, parameter in layer.named_parameters():
            expected_gradient = ref_weights[name].grad
            if expected_gradient is None:
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                torch.testing.assert_close(parameter.grad, expected_gradient, atol=3e-6, rtol=3e-5)
    assert owner.core_attention.indexer.linear_wk.weight.grad.abs().sum() > 0
    assert reindex.core_attention.indexer.linear_wq_b.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Candidate tensors use CUDA")
def test_candidate_blocks_exclude_old_blocks_and_force_newest():
    # Latest position has the lowest score; pinning must displace the second-best old block.
    scores = torch.tensor([[[9.0, 7.0, 8.0, 6.0, 5.0, 4.0, -100.0]]], device="cuda")
    selected = select_candidate_blocks(scores, torch.tensor([[7]], device="cuda"), 2, 2)
    expected = torch.tensor([[[True, True, False, False, False, False, True]]], device="cuda")
    torch.testing.assert_close(selected, expected)
    # No completed ratio-2 group means no candidate, despite having storage for later keys.
    empty = select_candidate_blocks(
        torch.full_like(scores, -torch.inf), torch.tensor([[0]], device="cuda"), 2, 2
    )
    assert not empty.any()
    partial = scores.masked_fill(torch.arange(7, device="cuda") >= 3, -torch.inf)
    selected = select_candidate_blocks(partial, torch.tensor([[3]], device="cuda"), 2, 2)
    expected = torch.tensor([[[True, True, True, True, False, False, False]]], device="cuda")
    torch.testing.assert_close(selected, expected)


# Integration with the existing TransformerBlock and HybridStack


def _build_stack(stack_kind, enable_hyper_connections, pg_collection):
    # Dense MLPs keep this test focused on cross-layer attention state. The Hybrid
    # stack uses its existing attention-only D layers, including its mHC wrapper.
    config = _make_config(
        enable_hyper_connections=enable_hyper_connections,
        num_moe_experts=None,
        moe_ffn_hidden_size=None,
        moe_shared_expert_intermediate_size=None,
        moe_router_enable_expert_bias=False,
        ffn_hidden_size=48,
        activation_func_clamp_value=None,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
    )
    if stack_kind == "transformer":
        stack = TransformerBlock(
            config,
            get_transformer_block_with_experimental_attention_variant_spec(config),
            pg_collection=pg_collection,
        )
        expected_layer = (
            HyperConnectionTransformerLayer if enable_hyper_connections else TransformerLayer
        )
    else:
        stack = HybridStack(
            config,
            hybrid_dsv4_stack_spec(config).submodules,
            layer_type_list=[Symbols.DS_ATTENTION] * config.num_layers,
            pg_collection=pg_collection,
        )
        expected_layer = (
            HyperConnectionHybridLayer if enable_hyper_connections else TransformerLayer
        )
    assert all(isinstance(layer, expected_layer) for layer in stack.layers)
    return stack.cuda().train()


@pytest.mark.parametrize("stack_kind", ["transformer", "hybrid"])
@pytest.mark.parametrize("enable_hyper_connections", [False, True], ids=["residual", "mhc"])
def test_stack_owns_fresh_csa2_state_for_each_forward(
    pg_collection, monkeypatch, stack_kind, enable_hyper_connections
):
    stack = _build_stack(stack_kind, enable_hyper_connections, pg_collection)
    observed = []
    original_forward = CompressedSparseAttention2.forward

    def observe_forward(module, *args, **kwargs):
        state = kwargs.get("csa2_state")
        assert isinstance(state, CSA2State)
        if module.layer_idx == 0:
            assert state.last_layer is None
            assert state.global_kv is None
            assert state.global_indices is None
            assert state.candidates is None
        output = original_forward(module, *args, **kwargs)
        # Snapshot tensor references now: Full/Reindex will overwrite state fields.
        observed.append((module.layer_idx, state, state.global_kv, state.global_indices))
        return output

    monkeypatch.setattr(CompressedSparseAttention2, "forward", observe_forward)
    inputs, outputs, states = [], [], []
    for sequence_length in (5, 7):
        x = torch.randn(
            sequence_length, 2, stack.config.hidden_size, device="cuda", requires_grad=True
        )
        start = len(observed)
        output = stack(hidden_states=x, attention_mask=None)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        records = observed[start:]
        assert [record[0] for record in records] == list(range(stack.config.num_layers))
        state = records[0][1]
        assert all(record[1] is state for record in records)
        assert state.last_layer == 5
        assert state.sequence_length == sequence_length
        assert state.kv_source_layer == 3
        assert state.index_source_layer == 4
        assert state.candidate_source_layer == 3
        assert state.global_kv.requires_grad
        # SWA, Full2, Reuse2, Full1, Reindex1, Reuse1 pass through the real wrappers.
        assert records[0][2] is None
        assert records[1][2] is records[2][2]
        assert records[1][3] is records[2][3]
        assert records[3][2] is records[4][2] is records[5][2]
        assert records[3][2] is not records[1][2]
        assert records[3][3] is not records[4][3]
        assert records[4][3] is records[5][3]
        inputs.append(x)
        outputs.append(output)
        states.append(state)

    assert states[0] is not states[1]
    assert states[0].global_kv is not states[1].global_kv
    # Both graphs are live. Finish the later microbatch first, then the earlier one.
    (outputs[1] * torch.randn_like(outputs[1])).sum().backward()
    assert inputs[0].grad is None
    assert inputs[1].grad is not None and torch.isfinite(inputs[1].grad).all()
    (outputs[0] * torch.randn_like(outputs[0])).sum().backward()
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()
    for module in stack.modules():
        assert not any(isinstance(value, CSA2State) for value in vars(module).values())
        if isinstance(module, CompressedSparseAttention2) and module.is_kv_source:
            grad = module.compressor.linear_wkv.weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert grad.abs().sum() > 0
    for name, parameter in stack.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


# Native CPU indexer-loss supervision and shared autograd graphs.
# The teacher oracle normalizes joint global, window, and sink logits. Native
# projection/norm adapters retain the production compressor, indexer, RoPE, and loss.


def _groups():
    # A real singleton ProcessGroup supplies size/rank without starting collectives.
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    return groups


def _math_core(sparse=False, per_token=False, coefficient=0.3):
    core = CompressedSparseAttention2.__new__(CompressedSparseAttention2)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(
        dsa_indexer_loss_coeff=coefficient,
        dsa_indexer_use_sparse_loss=sparse,
        calculate_per_token_loss=per_token,
    )
    core.pg_collection = _groups()
    core.softmax_scale = 0.5
    core.attn_sink = nn.Parameter(torch.tensor([-1.0, 2.0, 4.0]))
    return core


def _inputs(dtype=torch.float32, candidates=False, empty=False):
    torch.manual_seed(102)
    length, batch, heads, dim = (1 if empty else 7), 2, 3, 4
    width = length // 2
    query = torch.randn(length, batch, heads, dim, dtype=dtype, requires_grad=True)
    local_kv = torch.randn(length, batch, dim, dtype=dtype, requires_grad=True)
    global_kv = torch.randn(width, batch, dim, dtype=dtype, requires_grad=True)
    raw_scores = torch.randn(batch, length, width, requires_grad=True)
    visible = torch.arange(width)[None, :] < torch.arange(1, length + 1)[:, None] // 2
    visible = visible.expand(batch, -1, -1).clone()
    if candidates:
        visible[0, 5:, 1] = False
        visible[1, 3:, 0] = False
    scores = raw_scores.masked_fill(~visible, -torch.inf)
    topk = scores.detach().topk(min(2, width), dim=-1).indices
    topk = topk.masked_fill(~scores.detach().gather(-1, topk).isfinite(), -1)
    window = torch.full((batch, length, 2), -1, dtype=torch.int64)
    for row in range(length):
        positions = torch.arange(max(row - 1, 0), row + 1)
        window[:, row, : positions.numel()] = positions
    return (query, local_kv, global_kv, window, topk, scores), raw_scores, visible


def _teacher_oracle(core, query, local_kv, global_kv, window, topk, scores):
    """Independent per-query KL, including window/sink mass before summing heads."""
    losses = []
    for batch in range(scores.shape[0]):
        for row in range(scores.shape[1]):
            positions = scores[batch, row].detach().isfinite().nonzero().flatten()
            if core.config.dsa_indexer_use_sparse_loss:
                positions = positions[torch.isin(positions, topk[batch, row])]
            if positions.numel() == 0:
                losses.append(scores[batch, row, :0].sum() * 0)
                continue
            q = query[row, batch].detach().float()
            selected = global_kv[positions, batch].detach().float()
            global_logits = (q @ selected.T) * core.softmax_scale
            local_positions = window[batch, row]
            local_positions = local_positions[local_positions >= 0]
            local = local_kv[local_positions, batch].detach().float()
            window_logits = (q @ local.T) * core.softmax_scale
            all_logits = torch.cat(
                (global_logits, window_logits, core.attn_sink.detach()[:, None]), dim=-1
            )
            teacher = all_logits.softmax(-1)[:, : positions.numel()].sum(0)
            teacher = teacher / teacher.sum()
            log_prediction = scores[batch, row, positions].float().log_softmax(-1)
            losses.append((teacher * (teacher.clamp_min(1e-10).log() - log_prediction)).sum())
    loss = torch.stack(losses).sum() * core.config.dsa_indexer_loss_coeff
    if not core.config.calculate_per_token_loss:
        loss = loss / (scores.shape[0] * scores.shape[1])
    return loss


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("candidates", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loss_and_score_gradients_match_joint_teacher(sparse, per_token, candidates, dtype):
    core = _math_core(sparse, per_token)
    args, raw_scores, visible = _inputs(dtype, candidates)
    scores_before = args[-1].detach().clone()
    reference_raw = raw_scores.detach().clone().requires_grad_()
    reference_scores = reference_raw.masked_fill(~visible, -torch.inf)
    expected = _teacher_oracle(core, *args[:-1], reference_scores)
    actual = core._compute_indexer_loss(*args)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(args[-1], scores_before)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(raw_scores.grad, reference_raw.grad, atol=2e-7, rtol=2e-6)
    assert torch.isfinite(raw_scores.grad).all()
    assert raw_scores.grad.abs().sum() > 0
    assert torch.count_nonzero(raw_scores.grad.masked_select(~visible)) == 0
    assert all(tensor.grad is None for tensor in args[:3])
    assert core.attn_sink.grad is None


@pytest.mark.parametrize("sparse", [False, True])
def test_teacher_depends_on_window_and_sink_mass(sparse):
    core = _math_core(sparse)
    args, _, _ = _inputs()
    baseline = core._compute_indexer_loss(*args)
    changed_window = (args[0], args[1] * 5, *args[2:])
    changed = core._compute_indexer_loss(*changed_window)
    assert not torch.isclose(baseline, changed, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        core.attn_sink.copy_(torch.tensor([9.0, -7.0, 0.5]))
    changed = core._compute_indexer_loss(*args)
    assert not torch.isclose(baseline, changed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("empty_width", [False, True])
def test_no_visible_global_positions_have_finite_zero_loss(sparse, empty_width):
    core = _math_core(sparse)
    args, raw_scores, _ = _inputs(empty=empty_width)
    if not empty_width:
        args = (
            *args[:-2],
            torch.full_like(args[-2], -1),
            raw_scores.masked_fill(torch.ones_like(raw_scores, dtype=torch.bool), -torch.inf),
        )
    loss = core._compute_indexer_loss(*args)
    assert loss.item() == 0
    loss.backward()
    assert raw_scores.grad is not None
    assert torch.isfinite(raw_scores.grad).all()
    assert torch.count_nonzero(raw_scores.grad) == 0


def test_coefficient_and_token_sum_scaling():
    args, _, _ = _inputs()
    base = _math_core()._compute_indexer_loss(*args)
    scaled = _math_core(coefficient=0.9)._compute_indexer_loss(*args)
    token_sum = _math_core(per_token=True)._compute_indexer_loss(*args)
    torch.testing.assert_close(scaled, base * 3)
    torch.testing.assert_close(token_sum, base * 14)


def test_autoscaler_preserves_output_and_scales_only_auxiliary_backward():
    args, raw_scores, _ = _inputs()
    loss = _math_core()._compute_indexer_loss(*args)
    expected_gradient = torch.autograd.grad(loss, raw_scores, retain_graph=True)[0]
    output = torch.randn(3, requires_grad=True)
    previous = DSAIndexerLossAutoScaler.main_loss_backward_scale
    try:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(0.25))
        attached = DSAIndexerLossAutoScaler.apply(output, loss)
        torch.testing.assert_close(attached, output)
        (attached.sum() * 3).backward()
        torch.testing.assert_close(output.grad, torch.full_like(output, 3))
        torch.testing.assert_close(raw_scores.grad, expected_gradient * 0.25)
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = previous


class _NativeLinear(nn.Module):
    def __init__(self, input_size, output_size, config, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        nn.init.normal_(self.weight, std=0.2)

    def forward(self, x):
        return F.linear(x, self.weight), None


class _NativeNorm(nn.RMSNorm):
    def __init__(self, hidden_size, eps, config):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


class _CPURotaryTable(nn.Module):
    """Ordinary CPU frequency table; the production RoPE application remains unchanged."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("frequencies", 10000 ** (-torch.arange(0, dim, 2).float() / dim))

    def forward(self, length, packed_seq=False):
        angles = torch.outer(torch.arange(length).float(), self.frequencies)
        return torch.cat((angles, angles), dim=-1)[:, None, None, :]


def _sharing_cores(per_token=False, reuse=False):
    config = _make_config(
        num_layers=4 if reuse else 3,
        csa_compress_ratios=[1] * (4 if reuse else 3),
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 1, 2],
        csa2_candidate_source_layer=0,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=0.3,
        calculate_per_token_loss=per_token,
    )
    modules = CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(
            module=CSA2Compressor,
            submodules=CompressorSubmodules(_NativeLinear, _NativeLinear, _NativeNorm),
        ),
        indexer=ModuleSpec(
            module=CSA2Indexer,
            submodules=CSA2IndexerSubmodules(
                _NativeLinear, _NativeLinear, _NativeNorm, _NativeLinear
            ),
        ),
    )
    return [
        CompressedSparseAttention2(
            config,
            modules,
            layer_number=layer + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=_groups(),
            rotary_pos_emb=_CPURotaryTable(config.qk_pos_emb_head_dim),
        )
        for layer in range(config.num_layers)
    ]


def test_shared_keys_accumulate_consumer_auxiliary_gradients_without_backbone_gradients():
    torch.manual_seed(91)
    owner, first, second = _sharing_cores()
    state = CSA2State()
    hidden = [torch.randn(7, 2, 32, requires_grad=True) for _ in range(3)]
    query_latents = [torch.randn(7, 2, 16, requires_grad=True) for _ in range(3)]
    records = [
        core._shared_global_attention(x, qr, state, use_indexer_loss=True)
        for core, x, qr in zip((owner, first, second), hidden, query_latents)
    ]
    assert records[0][0] is records[1][0] is records[2][0]
    assert state.indexer_k.requires_grad
    assert state.candidates is not None and not state.candidates.requires_grad
    losses = []
    for core, (global_kv, indices, scores) in zip((owner, first, second), records):
        query = torch.randn(7, 2, 4, 16, requires_grad=True)
        local = torch.randn(7, 2, 16, requires_grad=True)
        window = torch.arange(7).view(1, 7, 1).expand(2, -1, -1)
        losses.append(core._compute_indexer_loss(query, local, global_kv, window, indices, scores))
    weight = owner.indexer.linear_wk.weight
    individual = [torch.autograd.grad(loss, weight, retain_graph=True)[0] for loss in losses]
    assert all(gradient.abs().sum() > 0 for gradient in individual)
    sum(losses).backward()
    torch.testing.assert_close(weight.grad, sum(individual), atol=2e-6, rtol=2e-5)
    assert all(x.grad is None for x in hidden + query_latents)
    assert all(parameter.grad is None for parameter in owner.compressor.parameters())
    for core in (owner, first, second):
        for parameter in core.indexer.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    # Detaching the indexer branch must preserve the independent LM/global-KV graph.
    records[0][0].square().sum().backward()
    assert hidden[0].grad is not None and hidden[0].grad.abs().sum() > 0
    assert owner.compressor.linear_wkv.weight.grad.abs().sum() > 0


def _forward_inputs(count):
    return [
        (
            torch.randn(7, 2, 4, 16, requires_grad=True),
            torch.randn(7, 2, 1, 16, requires_grad=True),
            torch.randn(7, 2, 32, requires_grad=True),
            torch.randn(7, 2, 16, requires_grad=True),
        )
        for _ in range(count)
    ]


def _forward_cores(cores, inputs):
    state = CSA2State()
    return [
        core(q, kv, kv, None, x=x, qr=qr, csa2_state=state)
        for core, (q, kv, x, qr) in zip(cores, inputs)
    ]


def _record_losses(monkeypatch):
    records = []

    def record(**kwargs):
        assert not kwargs["loss"].requires_grad
        records.append(kwargs)

    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", record)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    return records


@pytest.mark.parametrize("per_token", [False, True])
def test_forward_auxiliary_loss_preserves_main_gradients_and_excludes_reuse(monkeypatch, per_token):
    torch.manual_seed(307)
    records = _record_losses(monkeypatch)
    cores = _sharing_cores(per_token, reuse=True)
    baseline = _sharing_cores(per_token, reuse=True)
    for core, reference in zip(cores, baseline):
        reference.load_state_dict(core.state_dict())
    baseline[0].config.dsa_indexer_loss_coeff = 0
    inputs = _forward_inputs(len(cores))
    reference_inputs = [
        tuple(tensor.detach().clone().requires_grad_() for tensor in row) for row in inputs
    ]
    outputs = _forward_cores(cores, inputs)
    expected = _forward_cores(baseline, reference_inputs)
    assert [record["layer_number"] for record in records] == [1, 2, 3]
    assert all(record["num_layers"] == 4 for record in records)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    sum(output.square().mean() for output in outputs).backward()
    sum(output.square().mean() for output in expected).backward()
    for core, reference in zip(cores, baseline):
        parameters = dict(reference.named_parameters())
        for name, parameter in core.named_parameters():
            if name.startswith("indexer."):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameters[name].grad is None
            else:
                torch.testing.assert_close(parameter.grad, parameters[name].grad, atol=0, rtol=0)
    for row, reference_row in zip(inputs, reference_inputs):
        for tensor, reference in zip(row, reference_row):
            torch.testing.assert_close(tensor.grad, reference.grad, atol=0, rtol=0)


def test_token_sum_mode_keeps_same_logged_metric_and_scales_gradients(monkeypatch):
    torch.manual_seed(81)
    records = _record_losses(monkeypatch)
    average_cores = _sharing_cores()
    sum_cores = _sharing_cores(per_token=True)
    for core, summed in zip(average_cores, sum_cores):
        summed.load_state_dict(core.state_dict())
    inputs = _forward_inputs(3)
    average_outputs = _forward_cores(average_cores, inputs)
    sum_outputs = _forward_cores(sum_cores, inputs)
    assert len(records) == 6
    for average, summed in zip(records[:3], records[3:]):
        torch.testing.assert_close(average["loss"], summed["loss"])
    sum(output.sum() for output in average_outputs).backward()
    sum(output.sum() for output in sum_outputs).backward()
    for average, summed in zip(average_cores, sum_cores):
        for parameter, summed_parameter in zip(
            average.indexer.parameters(), summed.indexer.parameters()
        ):
            torch.testing.assert_close(
                summed_parameter.grad, parameter.grad * 14, atol=3e-6, rtol=3e-5
            )


@pytest.mark.parametrize("evaluation", [False, True])
def test_evaluation_and_no_grad_do_not_attach_auxiliary_loss(monkeypatch, evaluation):
    records = _record_losses(monkeypatch)
    cores = _sharing_cores(reuse=True)
    inputs = _forward_inputs(len(cores))
    if evaluation:
        for core in cores:
            core.eval()
        outputs = _forward_cores(cores, inputs)
        sum(output.square().mean() for output in outputs).backward()
        assert all(
            parameter.grad is None
            for core in cores
            if core.indexer is not None
            for parameter in core.indexer.parameters()
        )
    else:
        with torch.no_grad():
            outputs = _forward_cores(cores, inputs)
        assert all(not output.requires_grad for output in outputs)
    assert not records
