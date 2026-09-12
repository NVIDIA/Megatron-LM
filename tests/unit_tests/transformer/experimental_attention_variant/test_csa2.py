# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native and fused CSA2 parity, cross-layer sharing, and training-stack integration.

Reference: deepseek-ai/DeepSeek-V4.1-Flash, revision
dba1be0a40aa45a94ad051997016db3960a90277, inference/model.py.
The oracle uses functional projections, adjacent-pair RoPE, and dense masked attention;
it does not call the implementation's compressor, indexer, or sparse attention helpers.
"""

import math
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
    CSA2State,
    apply_csa2_thd_rope,
    select_candidate_blocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    csa2_candidates,
    csa2_indexer,
    fused_compressor,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
    candidate_blocks_from_scores,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
    fused_csa2_indexer_loss,
    prepare_csa2_indexer_inputs,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.module import (
    MegatronModule,
    convert_module_to_dtype_except_fp32_marked,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config
from tests.unit_tests.transformer.experimental_attention_variant.test_indexer_quantization import (
    _reference_quantize,
)


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
                tokens = x[start : start + ratio]
                projected = linear(tokens, prefix + "linear_wkv").float()
                logits = linear(tokens, prefix + "linear_wgate").float()
                groups.append((projected * logits.softmax(dim=0)).sum(dim=0))
            if groups:
                latent = torch.stack(groups).to(x.dtype)
            else:
                # Preserve zero derivatives to both projections when no group is complete.
                latent = (
                    linear(x[:0], prefix + "linear_wkv") + linear(x[:0], prefix + "linear_wgate")
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
        selected = index_scores.argsort(dim=-1, descending=True, stable=True)[
            ..., : min(config.dsa_indexer_topk, latent.shape[0])
        ]
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


def test_bf16_conversion_keeps_only_sink_in_fp32(pg_collection):
    layer = _layer(pg_collection, 2)
    convert_module_to_dtype_except_fp32_marked(layer, torch.bfloat16)
    core = layer.core_attention
    assert core.attn_sink.dtype == torch.float32
    assert core.compressor.linear_wkv.weight.dtype == torch.bfloat16
    assert core.compressor.linear_wgate.weight.dtype == torch.bfloat16
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


def test_attention_mask_and_unsupported_packing_format(pg_collection):
    layer = _layer(pg_collection, 2)
    x = torch.randn(5, 2, layer.config.hidden_size, device="cuda")
    mask = torch.ones(1, 1, 5, 5, device="cuda", dtype=torch.bool).triu(1)
    torch.testing.assert_close(layer(x, mask)[0], layer(x, None)[0])
    mask[..., 3, 0] = True
    # Match V4: the sparse indices, rather than the caller's dense mask,
    # determine visibility.
    torch.testing.assert_close(layer(x, mask)[0], layer(x, None)[0])
    with pytest.raises((ValueError, NotImplementedError), match="[Tt][Hh][Dd]"):
        layer(
            x,
            None,
            packed_seq_params=PackedSeqParams(qkv_format="sbhd", max_seqlen_q=5, max_seqlen_kv=5),
        )


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
    top_indices = block_scores.argsort(dim=-1, descending=True, stable=True)[
        ..., : min(blocks_to_keep, len(blocks))
    ]
    selected = torch.zeros_like(block_scores, dtype=torch.bool).scatter(
        -1, top_indices, block_scores.gather(-1, top_indices) > -torch.inf
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
                    tokens = x[start : start + ratio]
                    projected = linear(tokens, prefix + "linear_wkv").float()
                    gate = linear(tokens, prefix + "linear_wgate").float().softmax(0)
                    groups.append((projected * gate).sum(0))
                latent = (
                    torch.stack(groups)
                    if groups
                    else (
                        linear(x[:0], prefix + "linear_wkv")
                        + linear(x[:0], prefix + "linear_wgate")
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
            top_indices = scores.argsort(dim=-1, descending=True, stable=True)[
                ..., : min(config.dsa_indexer_topk, global_len)
            ]
            state["selected"] = torch.zeros_like(scores, dtype=torch.bool).scatter(
                -1, top_indices, scores.gather(-1, top_indices).isfinite()
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
    candidate_mask = state.candidates.to_mask(state.indexer_k.shape[0])
    assert (~candidate_mask[:, -1]).any(), "The case must actually exclude old global positions."
    assert torch.isneginf(scores.masked_select(~candidate_mask)).all()
    selected = indexer.select_indices(scores)
    valid = selected >= 0
    assert candidate_mask.gather(-1, selected.clamp_min(0).long())[valid].all()
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
    assert state.candidates is not None and not state.candidates.indices.requires_grad
    assert state.candidates.indices.dtype == state.candidates.lengths.dtype == torch.int32
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


# Packed layout and padding-producer contracts.


def _params(valid, physical=None, max_seqlen=None):
    cu = torch.tensor(valid, dtype=torch.int32)
    padded = None if physical is None else torch.tensor(physical, dtype=torch.int32)
    lengths = cu.diff() if padded is None else padded.diff()
    if max_seqlen is None:
        max_seqlen = max(lengths.tolist(), default=0)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=None if padded is None else padded.clone(),
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        pad_between_seqs=physical is not None,
    )


def _pad(alignment, maximum, max_num_seqs, static, tail_policy):
    # Real sequences have lengths 5 and 3. The first has one physical padding
    # token. Logical prefixes are cumulative REAL lengths, not physical starts.
    params = _params([0, 5, 8], [0, 6, 9])
    mask = torch.tensor([[False] * 5 + [True] + [False] * 3])
    pad_alignment, target, sequence_capacity = get_thd_padding_kwargs(
        alignment, maximum, max_num_seqs, static
    )
    tokens, _, _, _, params, mask = pad_sequence_for_thd(
        tokens=torch.arange(9).unsqueeze(0),
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=params,
        alignment=pad_alignment,
        target_len=target,
        max_num_seqs=sequence_capacity,
        tail_padding_policy=tail_policy,
        padding_mask=mask,
        cp_size=1,
        cp_rank=0,
    )
    return tokens, params, mask


def _check_compression(layout, ratio):
    """Independent per-sequence enumeration checks addresses and validity."""
    compressed = layout.for_compression(ratio)
    valid = layout.valid_tokens.tolist()
    physical = layout.cu_seqlens_padded.tolist()
    source_rows, valid_groups, positions, seq_ids = [], [], [], []
    physical_prefix, valid_prefix = [0], [0]
    for seq_id, (start, end) in enumerate(zip(physical, physical[1:])):
        group_count = (end - start) // ratio
        seq_valid = 0
        for local_group in range(group_count):
            sources = list(range(start + ratio * local_group, start + ratio * (local_group + 1)))
            is_valid = all(valid[source] for source in sources)
            source_rows.append(sources)
            valid_groups.append(is_valid)
            positions.append(ratio * local_group if is_valid else 0)
            seq_ids.append(seq_id)
            seq_valid += is_valid
        physical_prefix.append(physical_prefix[-1] + group_count)
        valid_prefix.append(valid_prefix[-1] + seq_valid)
    expected_capacity = min(
        layout.total_tokens // ratio, (len(physical) - 1) * (layout.max_seqlen // ratio)
    )
    while len(source_rows) < expected_capacity:
        source_rows.append([0] * ratio)
        valid_groups.append(False)
        positions.append(0)
        seq_ids.append(-1)
    assert compressed.capacity == expected_capacity
    assert compressed.max_seqlen == layout.max_seqlen // ratio
    assert compressed.source_indices.shape == (compressed.capacity, ratio)
    assert compressed.source_indices.tolist() == source_rows
    assert compressed.valid_groups.tolist() == valid_groups
    assert compressed.position_ids.tolist() == positions
    assert compressed.sequence_ids.tolist() == seq_ids
    assert compressed.cu_seqlens_padded.tolist() == physical_prefix
    assert compressed.cu_seqlens.tolist() == valid_prefix
    assert compressed.cu_seqlens.dtype == layout.cu_seqlens.dtype
    return compressed


@pytest.mark.parametrize("alignment", [4, 8, "max"])
@pytest.mark.parametrize("max_num_seqs", [None, 6])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("tail_policy", ["append_dummy_seq", "extend_last"])
def test_actual_padding_configuration_layout(alignment, max_num_seqs, static, tail_policy):
    tokens, params, _ = _pad(alignment, 16, max_num_seqs, static, tail_policy)
    total = tokens.shape[-1]
    expected_total = 16 if static or alignment in (8, "max") else 12
    assert total == expected_total
    if max_num_seqs is not None:
        assert params.cu_seqlens_q.numel() == max_num_seqs + 1
        assert params.cu_seqlens_q[-1] == params.cu_seqlens_q[-2]
    layout = build_csa2_thd_layout(params, total)
    dummy_length = total - 9 if tail_policy == "append_dummy_seq" else 0
    assert layout.valid_tokens.tolist() == (
        [True] * 5 + [False] + [True] * (3 + dummy_length) + [False] * (total - 9 - dummy_length)
    )
    assert layout.position_ids.tolist() == (
        [0, 1, 2, 3, 4, 0, 0, 1, 2] + list(range(dummy_length)) + [0] * (total - 9 - dummy_length)
    )
    assert layout.valid_tokens.sum() == 8 + dummy_length
    for ratio, expected_valid in ((1, 8), (2, 3)):
        compressed = _check_compression(layout, ratio)
        assert compressed.valid_groups.sum() == expected_valid + dummy_length // ratio


def test_integer_alignment_capacity_comes_from_padded_tensor():
    # Integer alignment rounds the complete pack, even when rounding exceeds
    # max_seqlen_per_dp_cp_rank. That maximum is a target only for "max"/static.
    tokens, params, _ = _pad(8, 10, 6, False, "append_dummy_seq")
    assert tokens.shape[-1] == 16
    layout = build_csa2_thd_layout(params, tokens.shape[-1])
    compressed = _check_compression(layout, 2)
    assert compressed.capacity == 8
    assert compressed.cu_seqlens_padded[-1] == 7
    assert compressed.sequence_ids[-1] == -1


def test_dummy_sequence_is_logically_valid_as_in_dsv4():
    tokens, params, mask = _pad("max", 16, 6, False, "append_dummy_seq")
    layout = build_csa2_thd_layout(params, tokens.shape[-1])
    # The producer mask still excludes dummies for consumers such as MoE.
    # Attention follows DSv4's cu metadata, where dummy sequences are ordinary sequences.
    assert (~mask).sum() == 8
    assert layout.valid_tokens.sum() == 15
    assert layout.for_compression(2).valid_groups.sum() == 6


def test_metadata_capacity_includes_dummy_sequence():
    with pytest.raises(AssertionError, match="thd_max_packed_sequences"):
        _pad("max", 16, 2, False, "append_dummy_seq")
    tokens, params, _ = _pad("max", 16, 2, False, "extend_last")
    layout = build_csa2_thd_layout(params, tokens.shape[-1])
    assert layout.cu_seqlens.numel() == 3
    assert _check_compression(layout, 2).valid_groups.sum() == 3


def test_fixed_target_rejects_oversize_without_truncation():
    with pytest.raises(AssertionError, match="exceeds"):
        _pad("max", 8, 6, False, "append_dummy_seq")


@pytest.mark.parametrize(
    "valid,physical,total",
    [
        ([0], [0], 0),
        ([0], [0], 4),
        ([0, 0, 0], [0, 0, 0], 0),
        ([0, 0, 0], [0, 2, 4], 4),
        ([0, 1], [0, 1], 1),
        ([0, 1, 2], [0, 1, 2], 2),
        ([0, 0, 3, 3], [0, 0, 3, 3], 5),
    ],
)
def test_empty_repeated_and_unassigned_capacity(valid, physical, total):
    layout = build_csa2_thd_layout(_params(valid, physical), total)
    for ratio in (1, 2):
        _check_compression(layout, ratio)


@pytest.mark.parametrize(
    "valid,physical,total,max_seqlen,expected_capacities",
    [
        pytest.param([0, 3], [0, 4], 64, 4, (4, 2), id="tail-exceeds-sequence-bound"),
        pytest.param([0, 1, 2], [0, 1, 2], 32, 1, (2, 0), id="max-shorter-than-r2"),
        pytest.param([0], [0], 16, 8, (0, 0), id="no-sequences-with-token-capacity"),
        pytest.param(
            [0, 2, 2, 2],
            [0, 2, 2, 2],
            64,
            4,
            (12, 6),
            id="metadata-capacity-includes-empty-entries",
        ),
    ],
)
def test_compressed_capacity_respects_dsv4_sequence_bound(
    valid, physical, total, max_seqlen, expected_capacities
):
    layout = build_csa2_thd_layout(_params(valid, physical, max_seqlen), total)
    for ratio, expected in zip((1, 2), expected_capacities):
        compressed = _check_compression(layout, ratio)
        assert compressed.capacity == expected
        assert compressed.capacity < total // ratio


def test_layout_snapshot_detects_reused_metadata_changes():
    params = _params([0, 2, 4], max_seqlen=4)
    layout = build_csa2_thd_layout(params, 4)
    layout.validate_layout(layout)
    layout.validate_compatible(params, 4)
    params.cu_seqlens_q[1] = 1
    params.cu_seqlens_kv[1] = 1
    assert layout.cu_seqlens.tolist() == [0, 2, 4]
    with pytest.raises(ValueError, match="different THD layout"):
        layout.validate_compatible(params, 4)


@pytest.mark.parametrize(
    "valid,physical,total,max_seqlen,message",
    [
        ([1, 4], [1, 4], 4, 4, "start at zero"),
        ([0, 3, 2], [0, 3, 4], 4, 4, "monotonic"),
        ([0, 5], [0, 5], 4, 5, "exceeds physical token"),
        ([0, 4], [0, 3], 4, 4, "exceeds its physical segment"),
        ([0, 3], [0, 4], 4, 3, "exceeds max_seqlen"),
    ],
)
def test_invalid_prefix_values(valid, physical, total, max_seqlen, message):
    with pytest.raises(ValueError, match=message):
        build_csa2_thd_layout(_params(valid, physical, max_seqlen), total)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "values", "physical_values"])
def test_query_kv_metadata_must_agree(mismatch):
    params = _params([0, 2, 4], [0, 2, 4])
    if mismatch == "shape":
        params.cu_seqlens_kv = params.cu_seqlens_kv[:2]
    elif mismatch == "dtype":
        params.cu_seqlens_kv = params.cu_seqlens_kv.to(torch.int64)
    elif mismatch == "values":
        params.cu_seqlens_kv[1] = 1
    else:
        params.cu_seqlens_kv_padded[1] = 1
    with pytest.raises(ValueError, match="query/KV"):
        build_csa2_thd_layout(params, 4)


@pytest.mark.parametrize("ratio", [0, 3, True])
def test_unsupported_compression_ratio(ratio):
    layout = build_csa2_thd_layout(_params([0, 4]), 4)
    with pytest.raises(ValueError, match="ratio must be 1 or 2"):
        layout.for_compression(ratio)


def test_nonlocal_cp_metadata_is_rejected():
    params = _params([0, 4])
    params.local_cp_size = 2
    with pytest.raises(ValueError, match="requires CP=1"):
        build_csa2_thd_layout(params, 4)


# Packed compressor/RoPE parity and shared DSv4 indexing regressions.


class _Linear(nn.Linear):
    """Ordinary PyTorch projection exposing Megatron's output/bias interface."""

    def __init__(self, input_size, output_size, config, **kwargs):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)

    def forward(self, x):
        return super().forward(x), None


class _RMSNorm(nn.Module):
    """Native RMSNorm with FP32 statistics and activation-dtype output."""

    def __init__(self, hidden_size, eps, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


def _config(dtype):
    return SimpleNamespace(
        attention_backend="unfused",
        dsa_kernel_backend="none",
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        hidden_size=6,
        q_lora_rank=4,
        v_head_dim=8,
        qk_pos_emb_head_dim=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=2,
        attention_latent_norm_epsilon=1e-20,
        init_method=lambda weight: nn.init.normal_(weight, std=0.2),
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mrope_section=None,
    )


def _compressor(ratio, dtype, **overrides):
    config = _config(dtype)
    for name, value in overrides.items():
        setattr(config, name, value)
    module = CSA2Compressor(
        config=config,
        submodules=CompressorSubmodules(_Linear, _Linear, _RMSNorm),
        compress_ratio=ratio,
        pg_collection=_groups(),
    )
    with torch.no_grad():
        module.norm.weight.uniform_(0.7, 1.3)
    return module


def _packed(real_lengths, physical_lengths, *, tail=0, dummy_sequences=(), device="cpu"):
    """Build metadata and a producer mask; DSv4 attention includes logical dummy rows."""
    real_cu = torch.tensor([0, *accumulate(real_lengths)], dtype=torch.int32, device=device)
    physical_cu = torch.tensor([0, *accumulate(physical_lengths)], dtype=torch.int32, device=device)
    total = sum(physical_lengths) + tail
    valid = torch.zeros(total, dtype=torch.bool, device=device)
    producer_valid = valid.clone()
    offset = 0
    for sequence, (real, physical) in enumerate(zip(real_lengths, physical_lengths)):
        valid[offset : offset + real] = True
        if sequence not in dummy_sequences:
            producer_valid[offset : offset + real] = True
        offset += physical
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real_cu,
        cu_seqlens_kv=real_cu,
        cu_seqlens_q_padded=physical_cu,
        cu_seqlens_kv_padded=physical_cu,
        max_seqlen_q=max(physical_lengths, default=0),
        max_seqlen_kv=max(physical_lengths, default=0),
        total_tokens=total,
        pad_between_seqs=True,
    )
    return params, (~producer_valid).unsqueeze(0), valid


def _reference_compressor(x, weights, ratio, physical_lengths, valid):
    """Project each complete physical group separately; padded groups have no derivative."""
    safe_x = x.masked_fill(~valid[:, None, None], 0)
    zero = safe_x.sum() * 0
    for weight in weights.values():
        zero = zero + weight.sum() * 0
    zero_row = zero.expand(1, weights["norm.weight"].shape[0]).to(x.dtype)
    outputs = []
    offset = 0
    for physical in physical_lengths:
        for position in range(0, physical - ratio + 1, ratio):
            start = offset + position
            if not valid[start : start + ratio].all():
                outputs.append(zero_row)
                continue
            tokens = x[start : start + ratio]
            if ratio == 1:
                latent = F.linear(tokens[0], weights["linear_wkv.weight"].to(x.dtype))
            else:
                projected = F.linear(tokens, weights["linear_wkv.weight"].to(x.dtype)).float()
                gate = F.linear(tokens, weights["linear_wgate.weight"].to(x.dtype)).float()
                # Each output channel has its own distribution over this sequence's group.
                latent = (projected * gate.softmax(dim=0)).sum(dim=0).to(x.dtype)
            xf = latent.float()
            normalized = xf / (xf.square().mean(dim=-1, keepdim=True) + 1e-20).sqrt()
            outputs.append((normalized * weights["norm.weight"].float()).to(x.dtype))
        offset += physical
    expected_capacity = min(
        x.shape[0] // ratio, len(physical_lengths) * (max(physical_lengths, default=0) // ratio)
    )
    outputs.extend([zero_row] * (expected_capacity - len(outputs)))
    if not outputs:
        return zero_row.unsqueeze(0)[:0]
    return torch.stack(outputs)


_PACK_CASES = [
    pytest.param([1, 3, 4, 5], [1, 3, 4, 5], 0, (), id="odd-sequences"),
    pytest.param([1, 3, 4], [4, 4, 6], 3, (), id="inter-sequence-and-tail-padding"),
    pytest.param([3, 4, 2], [4, 4, 2], 3, (2,), id="dummy-present-in-both-cu-arrays"),
    pytest.param([1, 1, 1], [3, 3, 3], 1, (), id="no-complete-r2-group"),
    pytest.param([0, 0], [4, 2], 2, (), id="entirely-padding"),
    pytest.param([1], [1], 0, (), id="zero-r2-capacity"),
    pytest.param([3], [4], 60, (), id="capacity-bounded-by-sequence-length"),
    pytest.param([1, 1], [1, 1], 30, (), id="capacity-zero-when-max-shorter-than-r2"),
    pytest.param([], [], 8, (), id="no-sequences-with-token-capacity"),
]


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("real_lengths,physical_lengths,tail,dummies", _PACK_CASES)
@pytest.mark.parametrize("backend", ["native", "fused"])
def test_compressor_packed_outputs_and_gradients_match_independent_groups(
    ratio, dtype, real_lengths, physical_lengths, tail, dummies, backend, monkeypatch
):
    torch.manual_seed(320)
    device = "cuda" if backend == "fused" else "cpu"
    if backend == "fused":
        _require_csa2_compressor_kernels()
    calls = _record_compressor_kernels(monkeypatch)
    params, _, valid = _packed(
        real_lengths, physical_lengths, tail=tail, dummy_sequences=dummies, device=device
    )
    module = _compressor(ratio, dtype).to(device)
    module.use_fused_compressor = backend == "fused"
    x = torch.randn(valid.numel(), 1, 6, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = {
        name: parameter.detach().float().clone().requires_grad_()
        for name, parameter in module.named_parameters()
    }
    latent, layout = module(x, packed_seq_params=params)
    expected = _reference_compressor(reference_x, weights, ratio, physical_lengths, valid)
    tolerance = dict(atol=3e-6, rtol=3e-5)
    if dtype == torch.bfloat16:
        tolerance = dict(atol=2e-2, rtol=3e-2)
    expected_capacity = min(
        valid.numel() // ratio, len(physical_lengths) * (max(physical_lengths, default=0) // ratio)
    )
    assert latent.shape == (expected_capacity, 1, 8)
    assert latent.dtype == dtype
    engaged = backend == "fused" and ratio == 2 and expected_capacity > 0
    assert calls == {
        "prepare": int(engaged and dtype == torch.float32),
        "pool": 0,
        "thd": int(engaged and dtype == torch.bfloat16),
    }
    torch.testing.assert_close(latent, expected, **tolerance)
    assert torch.count_nonzero(latent[~layout.valid_groups]) == 0

    # Multiple consumers exercise accumulation into the same compressor projections.
    probe_a = torch.randn_like(latent)
    probe_b = torch.randn_like(latent)
    actual_loss = (latent.float() * probe_a).sum() + (latent.float() * probe_b).sum() * 0.3
    expected_loss = (expected.float() * probe_a).sum() + (expected.float() * probe_b).sum() * 0.3
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
    assert torch.count_nonzero(x.grad[~valid]) == 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad.float(),
            weights[name].grad,
            **tolerance,
            msg=lambda detail: f"{name}: {detail}",
        )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["native", "fused"])
def test_nonfinite_padding_cannot_poison_compressor_forward_or_backward(ratio, dtype, backend):
    torch.manual_seed(321)
    device = "cuda" if backend == "fused" else "cpu"
    if backend == "fused":
        _require_csa2_compressor_kernels()
    params, _, valid = _packed([3, 1, 2], [4, 4, 2], tail=2, dummy_sequences=(2,), device=device)
    module = _compressor(ratio, dtype).to(device)
    module.use_fused_compressor = backend == "fused"
    x = torch.randn(valid.numel(), 1, 6, device=device, dtype=dtype)
    contaminated = x.clone()
    consumed = valid.clone()
    if ratio == 2:
        # Real trailing tokens also have no compressor representation. They must
        # not enter the Linear weight-gradient reduction with a zero upstream grad.
        consumed[2] = consumed[4] = False
    invalid_rows = (~consumed).nonzero().flatten()
    for index, row in enumerate(invalid_rows):
        contaminated[row] = [float("nan"), float("inf"), -float("inf")][index % 3]
    x.requires_grad_()
    contaminated.requires_grad_()
    expected, _ = module(x, packed_seq_params=params)
    actual, layout = module(contaminated, packed_seq_params=params)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[~layout.valid_groups]) == 0
    probe = torch.randn_like(actual)
    parameters = tuple(module.parameters())
    expected_grads = torch.autograd.grad((expected * probe).sum(), (x, *parameters))
    actual_grads = torch.autograd.grad((actual * probe).sum(), (contaminated, *parameters))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert torch.count_nonzero(actual_grads[0][~consumed]) == 0


@pytest.mark.parametrize("ratio", [1, 2])
def test_prebuilt_layout_and_ordinary_compressor_calls_preserve_contract(ratio):
    params, _, _ = _packed([3, 5], [4, 6], tail=2)
    module = _compressor(ratio, torch.float32)
    x = torch.randn(12, 1, 6)
    layout = build_csa2_thd_layout(params, x.shape[0])
    direct, _ = module(x, packed_seq_params=params)
    reused, _ = module(x, thd_layout=layout)
    torch.testing.assert_close(reused, direct, atol=0, rtol=0)
    ordinary = module(x[:5])
    assert isinstance(ordinary, torch.Tensor)
    assert ordinary.shape == (5 // ratio, 1, 8)


def _require_csa2_compressor_kernels():
    if not fused_compressor.fused_compressor_available():
        pytest.skip("CSA2 cuDNN compressor requires SM100+ and cuDNN frontend")


def _record_compressor_kernels(monkeypatch):
    calls = {"prepare": 0, "pool": 0, "thd": 0}
    for operation in calls:
        name = (
            "maybe_compress_csa2_thd_fused"
            if operation == "thd"
            else f"maybe_{operation}_csa2_r2_fused"
        )
        function = getattr(fused_compressor, name)

        def record(*args, op=operation, fn=function, **kwargs):
            output = fn(*args, **kwargs)
            calls[op] += output is not None
            return output

        monkeypatch.setattr(
            f"megatron.core.transformer.experimental_attention_variant.csa2.{name}", record
        )
    return calls


@pytest.mark.parametrize("dim", [7, 128, 512])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("saturated", [False, True])
def test_fused_compressor_r2_pool_bf16_projections_and_gradients(dim, batch, dtype, saturated):
    """Test cuDNN BF16 projections with FP32 pooling, strides, and extreme gates."""
    _require_csa2_compressor_kernels()
    torch.manual_seed(541)
    # Slicing the channel dimension also verifies the dispatch handles strided projections.
    kv = (
        torch.randn(34, batch, 2 * dim, device="cuda", dtype=dtype)[..., ::2]
        .detach()
        .requires_grad_()
    )
    gate = torch.randn_like(kv) * (1000 if saturated else 1)
    gate[:2] = 0  # Equal gates, including an exact tie, share mass equally.
    gate.requires_grad_()
    ref_kv = kv.detach().clone().requires_grad_()
    ref_gate = gate.detach().clone().requires_grad_()
    output = fused_compressor.maybe_pool_csa2_r2_fused(kv, gate, dtype)
    assert output is not None and output.dtype == dtype
    shape = (17, 2, batch, dim)
    expected = (
        (ref_kv.float().reshape(shape) * ref_gate.float().reshape(shape).softmax(1))
        .sum(1)
        .to(dtype)
    )
    tolerance = dict(atol=5e-7, rtol=3e-6)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=8e-3)
    else:
        torch.testing.assert_close(output, expected, **tolerance)
    probe = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (kv, gate), probe)
    expected_grads = torch.autograd.grad(expected, (ref_kv, ref_gate), probe)
    for actual, reference in zip(actual_grads, expected_grads):
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, reference, atol=2e-3, rtol=8e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [0, 1, 2, 17])
def test_fused_compressor_sbhd_module_and_weight_gradients(monkeypatch, dtype, seq_len):
    """Keep the pre-RMSNorm rounding boundary, dropped tail, and shared projection gradients."""
    _require_csa2_compressor_kernels()
    torch.manual_seed(542)
    calls = _record_compressor_kernels(monkeypatch)
    module = _compressor(
        2,
        dtype,
        hidden_size=64,
        v_head_dim=512,
        attention_backend="auto",
        dsa_kernel_backend="cudnn",
    ).cuda()
    reference = _compressor(2, dtype, hidden_size=64, v_head_dim=512).cuda()
    reference.load_state_dict(module.state_dict())
    # Non-contiguous input and an incomplete trailing token with no global representation.
    x = torch.randn(seq_len, 3, 128, dtype=dtype, device="cuda")[..., ::2]
    if seq_len % 2:
        x[-1] = float("nan")
    x.requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    norm_inputs = []
    module.norm.register_forward_pre_hook(lambda _, args: norm_inputs.append(args[0]))
    output, expected = module(x), reference(ref_x)
    assert calls == {"prepare": 0, "pool": int(seq_len >= 2 and dtype == torch.bfloat16), "thd": 0}
    if seq_len >= 2:
        assert norm_inputs[0].dtype == dtype
    tolerance = dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(output, expected, **tolerance)
    probe_a, probe_b = torch.randn_like(output), torch.randn_like(output)
    ((output * probe_a).sum() + (output * probe_b).sum()).backward()
    ((expected * probe_a).sum() + (expected * probe_b).sum()).backward()
    torch.testing.assert_close(x.grad, ref_x.grad, **tolerance)
    if seq_len % 2:
        assert torch.count_nonzero(x.grad[-1]) == 0
    for (name, param), (_, ref) in zip(module.named_parameters(), reference.named_parameters()):
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, ref.grad, **tolerance)
        if "linear" in name:
            assert param.dtype == param.grad.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_compressor_thd_preparation_strides_and_nonfinite_gradients(dtype):
    """Padding, incomplete groups, and unused capacity have exactly zero input gradients."""
    _require_csa2_compressor_kernels()
    params, _, valid = _packed([3, 1, 4], [5, 3, 6], tail=9, device="cuda")
    layout = build_csa2_thd_layout(params, valid.numel()).for_compression(2)
    x = torch.randn(valid.numel(), 1, 130, device="cuda", dtype=dtype)[..., ::2]
    x[~valid] = float("nan")
    x.requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    actual = fused_compressor.maybe_prepare_csa2_r2_fused(x, layout)
    expected = ref_x[layout.source_indices].masked_fill(
        ~layout.valid_groups[:, None, None, None], 0
    )
    expected = expected.reshape(layout.capacity * 2, 1, 65)
    assert actual is not None and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    probe.reshape(layout.capacity, 2, 1, 65)[~layout.valid_groups] = float("nan")
    actual_grad = torch.autograd.grad(actual, x, probe)[0]
    expected_grad = torch.autograd.grad(expected, ref_x, probe)[0]
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_fused_compressor_fallback_keeps_native_reference(monkeypatch):
    """Optional kernels never change CPU or disabled-backend semantics."""
    x = torch.randn(8, 1, 6, requires_grad=True)
    params, _, _ = _packed([3, 4], [4, 4])
    layout = build_csa2_thd_layout(params, x.shape[0]).for_compression(2)
    assert fused_compressor.maybe_prepare_csa2_r2_fused(x, layout) is None
    assert fused_compressor.maybe_pool_csa2_r2_fused(x, x, torch.float32) is None
    module = _compressor(2, torch.float32, attention_backend="auto", dsa_kernel_backend="cudnn")
    output, _ = module(x, packed_seq_params=params)
    monkeypatch.setattr(fused_compressor, "HAVE_TRITON", False)
    actual, _ = module(x, packed_seq_params=params)
    torch.testing.assert_close(actual, output, atol=0, rtol=0)
    actual_grads = torch.autograd.grad(actual.sum(), (x, *module.parameters()))
    expected_grads = torch.autograd.grad(output.sum(), (x, *module.parameters()))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_compressor_pool_ignores_nonfinite_padding_and_gradients():
    _require_csa2_compressor_kernels()
    torch.manual_seed(543)
    valid = torch.tensor([True, False, True, False], device="cuda")
    kv = torch.randn(8, 1, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(kv)
    kv.reshape(4, 2, 1, 128)[~valid] = float("nan")
    gate.reshape(4, 2, 1, 128)[~valid] = float("inf")
    kv.requires_grad_()
    gate.requires_grad_()
    output = fused_compressor.maybe_pool_csa2_r2_fused(kv, gate, torch.bfloat16, valid_groups=valid)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~valid]) == 0
    probe = torch.randn_like(output)
    probe[~valid] = float("nan")
    gradients = torch.autograd.grad(output, (kv, gate), probe)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient.reshape(4, 2, 1, 128)[~valid]) == 0
        assert gradient.reshape(4, 2, 1, 128)[valid].abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_compressor_cuda_graph_replay_and_determinism(dtype):
    """Replay forward/backward with changed THD groups at fixed buffer capacity."""
    _require_csa2_compressor_kernels()
    torch.manual_seed(544)
    params, _, valid = _packed([3, 5, 0], [5, 6, 2], tail=3, device="cuda")
    layout = build_csa2_thd_layout(params, valid.numel()).for_compression(2)
    x = torch.randn(valid.numel(), 1, 64, device="cuda", dtype=dtype, requires_grad=True)
    token_cu = params.cu_seqlens_q_padded.clone()
    module = _compressor(2, dtype, hidden_size=64, v_head_dim=128).cuda()
    module.use_fused_compressor = True
    reference = _compressor(2, dtype, hidden_size=64, v_head_dim=128).cuda()
    reference.load_state_dict(module.state_dict())
    probe = torch.randn(layout.capacity, 1, 128, device="cuda", dtype=dtype)

    def run():
        output, _ = module._forward_thd(x, layout, token_cu)
        return output, torch.autograd.grad(output, (x, *module.parameters()), probe)

    # Compile both directions and initialize GEMM handles on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, gradients = run()
    tolerance = dict(atol=4e-6, rtol=4e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=3e-2)
    for lengths, physical in [
        ([3, 5, 0], [5, 6, 2]),
        ([4, 2, 1], [4, 7, 2]),
        ([0, 0, 0], [5, 6, 2]),
    ]:
        next_params, _, next_valid = _packed(lengths, physical, tail=3, device="cuda")
        next_layout = build_csa2_thd_layout(next_params, next_valid.numel()).for_compression(2)
        with torch.no_grad():
            x.normal_()
            x[~next_valid] = float("nan")
            layout.source_indices.copy_(next_layout.source_indices)
            layout.valid_groups.copy_(next_layout.valid_groups)
            layout.cu_seqlens_padded.copy_(next_layout.cu_seqlens_padded)
            token_cu.copy_(next_params.cu_seqlens_q_padded)
        graph.replay()
        ref_x = x.detach().clone().requires_grad_()
        expected, _ = reference(ref_x, packed_seq_params=next_params)
        ref_grads = torch.autograd.grad(expected, (ref_x, *reference.parameters()), probe)
        torch.testing.assert_close(output, expected, **tolerance)
        for actual, ref in zip(gradients, ref_grads):
            torch.testing.assert_close(actual, ref, **tolerance)
        assert torch.count_nonzero(gradients[0][~next_valid]) == 0

    # The frontend still computes dAPE; strict deterministic mode must use native pooling.
    enabled, warn_only = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        original_layout = build_csa2_thd_layout(params, valid.numel()).for_compression(2)
        with torch.no_grad():
            x.normal_()
            layout.source_indices.copy_(original_layout.source_indices)
            layout.valid_groups.copy_(original_layout.valid_groups)
        # Run the kernel region alone so the test does not depend on cuBLAS workspace settings.
        kv = torch.randn(32, 2, 128, device="cuda", dtype=dtype, requires_grad=True)
        gate = torch.randn_like(kv, requires_grad=True)
        probe = torch.randn(16, 2, 128, device="cuda", dtype=dtype)
        snapshots = []
        for _ in range(2):
            assert fused_compressor.maybe_pool_csa2_r2_fused(kv, gate, dtype) is None
            pooled = (
                (kv.float().reshape(16, 2, 2, 128) * gate.float().reshape(16, 2, 2, 128).softmax(1))
                .sum(1)
                .to(dtype)
            )
            grads = torch.autograd.grad(pooled, (kv, gate), probe)
            prepared = fused_compressor.maybe_prepare_csa2_r2_fused(x, layout)
            dx = torch.autograd.grad(prepared, x, torch.ones_like(prepared))[0]
            snapshots.append((pooled, *grads, dx))
        for first, second in zip(*snapshots):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
def test_fused_compressor_transformer_engine_projections_and_norm(
    pg_collection, monkeypatch, dtype, packed
):
    """Actual TE linears use the model dtype around cuDNN BF16 pooling."""
    _require_csa2_compressor_kernels()
    layer = _layer(
        pg_collection,
        2,
        dtype,
        dsa_kernel_backend="cudnn" if dtype == torch.bfloat16 else "none",
        num_attention_heads=64,
        v_head_dim=512,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
    )
    module = layer.core_attention.compressor
    # The full sparse-attention backend requires BF16; exercise the standalone
    # compressor's FP32 reference dtype without enabling that attention backend.
    module.use_fused_compressor = True
    calls = _record_compressor_kernels(monkeypatch)
    if packed:
        params, _, valid = _packed([3, 5, 1], [5, 6, 2], tail=3, device="cuda")
        x = torch.randn(valid.numel(), 1, layer.config.hidden_size, device="cuda", dtype=dtype)
        x[~valid] = float("nan")
    else:
        params = None
        x = torch.randn(9, 3, layer.config.hidden_size, device="cuda", dtype=dtype)
    x.requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    weights = {name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()}
    actual = module(x, packed_seq_params=params)
    if packed:
        actual, _ = actual
        expected = _reference_compressor(ref_x, weights, 2, [5, 6, 2], valid)
    else:
        latent = F.linear(ref_x[:8], weights["linear_wkv.weight"]).float()
        gate = F.linear(ref_x[:8], weights["linear_wgate.weight"]).float()
        latent = (latent.reshape(4, 2, 3, 512) * gate.reshape(4, 2, 3, 512).softmax(1)).sum(1)
        expected = _reference_norm(latent.to(dtype), weights["norm.weight"], 1e-20)
    assert calls == {
        "prepare": int(packed and dtype == torch.float32),
        "pool": int(not packed and dtype == torch.bfloat16),
        "thd": int(packed and dtype == torch.bfloat16),
    }
    tolerance = dict(atol=4e-6, rtol=4e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(actual, expected, **tolerance)
    probe = torch.randn_like(actual) * 0.1
    actual.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(x.grad, ref_x.grad, **tolerance)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, weights[name].grad, **tolerance)
        if "linear" in name:
            assert parameter.dtype == parameter.grad.dtype == dtype


def _rotary_module(config, use_yarn, cp_group):
    if use_yarn:
        return YarnRotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_base=160000,
            scaling_factor=16,
            original_max_position_embeddings=128,
            beta_fast=32,
            beta_slow=1,
            mscale=0,
            mscale_all_dim=0,
            cp_group=cp_group,
        )
    return RotaryEmbedding(config.qk_pos_emb_head_dim, 1.0, rotary_base=10000, cp_group=cp_group)


def _rope_reference(x, config, rotary, positions, valid):
    """Adjacent-pair rotation from the module's mathematical frequency table."""
    dim = config.qk_pos_emb_head_dim
    frequencies = rotary(64, packed_seq=True)
    if isinstance(frequencies, tuple):
        frequencies = frequencies[0]
    # Frequency generation is independent of packing. Apply explicit local positions here.
    angles = frequencies[positions, 0, 0, : dim // 2]
    shape = (x.shape[0], *([1] * (x.ndim - 2)), dim // 2)
    cos = angles.cos().to(x.dtype).reshape(shape)
    sin = angles.sin().to(x.dtype).reshape(shape)
    safe_x = x.masked_fill(~valid.reshape(-1, *([1] * (x.ndim - 1))), 0)
    even, odd = safe_x[..., -dim::2], safe_x[..., -dim + 1 :: 2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((safe_x[..., :-dim], rotated), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron rotary tables require CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("stream", ["token-rope", "token-yarn", "compressed-r1", "compressed-r2"])
@pytest.mark.parametrize("heads", [None, 2])
@pytest.mark.parametrize("fused", [False, True])
def test_packed_rope_resets_positions_and_preserves_gradients(dtype, stream, heads, fused):
    torch.manual_seed(322)
    config = _config(dtype)
    config.apply_rope_fusion = fused
    params, _, token_valid = _packed(
        [3, 5, 2], [4, 6, 2], tail=3, dummy_sequences=(2,), device="cuda"
    )
    token_layout = build_csa2_thd_layout(params, token_valid.numel())
    ratio = 2 if stream == "compressed-r2" else 1
    is_compressed = stream.startswith("compressed")
    layout = token_layout.for_compression(ratio) if is_compressed else token_layout
    # Derive the expected physical row positions separately from layout.position_ids.
    positions, validity = [], []
    for real, physical in zip([3, 5, 2], [4, 6, 2]):
        step = ratio if is_compressed else 1
        for position in range(0, physical - step + 1, step):
            positions.append(position)
            validity.append(position + step <= real)
    rows = (
        min(token_valid.numel() // ratio, 3 * (6 // ratio))
        if is_compressed
        else token_valid.numel()
    )
    positions.extend([0] * (rows - len(positions)))
    validity.extend([False] * (rows - len(validity)))
    positions = torch.tensor(positions, dtype=torch.long, device="cuda")
    valid = torch.tensor(validity, dtype=torch.bool, device="cuda")
    shape = (rows, 1, config.v_head_dim) if heads is None else (rows, 1, heads, config.v_head_dim)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    x[~valid] = float("nan")
    x.requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    cp_group = _groups().cp
    rotary = _rotary_module(config, stream != "token-rope", cp_group)
    actual = apply_csa2_thd_rope(x, rotary, config, layout, cp_group)
    expected = _rope_reference(reference_x, config, rotary, positions, valid)
    tolerance = dict(atol=2e-6, rtol=2e-6)
    if dtype == torch.bfloat16:
        tolerance = dict(atol=2e-2, rtol=2e-2)
    assert torch.isfinite(actual).all()
    assert actual.shape == x.shape
    assert torch.isnan(x[~valid]).all(), "RoPE must not mutate shared input storage"
    torch.testing.assert_close(actual, expected, **tolerance)
    assert torch.count_nonzero(actual[~valid]) == 0
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
    assert torch.count_nonzero(x.grad[~valid]) == 0


class _CPUFrequencyTable(nn.Module):
    """Deterministic frequency inputs; the production RoPE operation remains unchanged."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("frequencies", 160000 ** (-torch.arange(0, dim, 2).float() / dim))

    def forward(self, length, packed_seq=False):
        angles = torch.outer(
            torch.arange(length, device=self.frequencies.device).float(), self.frequencies
        )
        return torch.cat((angles, angles), dim=-1)[:, None, None, :]


def _complex_rotation(x, dim, positions, valid, frequency_table):
    """Independent complex-number multiplication, with padding removed before arithmetic."""
    safe_x = x.masked_fill(~valid.reshape(-1, *([1] * (x.ndim - 1))), 0)
    pairs = safe_x[..., -dim:].float().reshape(*safe_x.shape[:-1], dim // 2, 2)
    values = torch.view_as_complex(pairs.contiguous())
    angles = torch.outer(positions.float(), frequency_table.frequencies)
    phase = torch.polar(torch.ones_like(angles), angles)
    phase = phase.reshape(x.shape[0], *([1] * (x.ndim - 2)), dim // 2)
    rotated = torch.view_as_real(values * phase).flatten(-2).to(x.dtype)
    return torch.cat((safe_x[..., :-dim], rotated), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ratio", [0, 1, 2], ids=["token", "compressed-r1", "compressed-r2"])
@pytest.mark.parametrize("heads", [None, 2])
def test_native_cpu_packed_rope_matches_complex_rotation(dtype, ratio, heads):
    params, _, token_valid = _packed([3, 5], [4, 6], tail=2)
    token_layout = build_csa2_thd_layout(params, 12)
    layout = token_layout.for_compression(ratio) if ratio else token_layout
    if ratio == 2:
        positions = torch.tensor([0, 2, 0, 2, 4, 0])
        valid = torch.tensor([True, False, True, True, False, False])
    else:
        positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 0])
        valid = token_valid
    shape = (len(positions), 1, 8) if heads is None else (len(positions), 1, heads, 8)
    torch.manual_seed(323)
    x = torch.randn(shape, dtype=dtype)
    x[~valid] = float("inf")
    x.requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    rotary = _CPUFrequencyTable(4)
    actual = apply_csa2_thd_rope(x, rotary, _config(dtype), layout, _groups().cp)
    expected = _complex_rotation(reference_x, 4, positions, valid, rotary)
    tolerance = dict(atol=2e-6, rtol=2e-6)
    if dtype == torch.bfloat16:
        # Native RoPE rounds each product to BF16; the complex oracle evaluates in FP32.
        tolerance = dict(atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual, expected, **tolerance)
    assert torch.isinf(x[~valid]).all()
    assert torch.count_nonzero(actual[~valid]) == 0
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[~valid]) == 0


@pytest.mark.parametrize("changed", ["physical-cu", "logical-cu"])
def test_shared_state_rejects_changed_packing_even_when_total_shape_matches(changed):
    params, _, _ = _packed([3, 5], [4, 6])
    original = build_csa2_thd_layout(params, 10)
    query = torch.randn(10, 2, 8)
    state = CSA2State()
    state.validate_forward(0, query, thd_layout=original)
    state.last_layer = 0
    # An equal independently built snapshot is valid; object identity is not required.
    equivalent = build_csa2_thd_layout(params, 10)
    state.validate_forward(1, query, thd_layout=equivalent)
    if changed == "physical-cu":
        different_params, _, _ = _packed([3, 5], [3, 7])
    else:
        different_params, _, _ = _packed([4, 4], [4, 6])
    different = build_csa2_thd_layout(different_params, 10)
    with pytest.raises(ValueError):
        state.validate_forward(1, query, thd_layout=different)
    with pytest.raises(ValueError, match="packed and unpacked"):
        state.validate_forward(1, query.unsqueeze(1))
    ordinary_state = CSA2State()
    ordinary_state.validate_forward(0, query.unsqueeze(1))
    ordinary_state.last_layer = 0
    with pytest.raises(ValueError, match="packed and unpacked"):
        ordinary_state.validate_forward(1, query, thd_layout=original)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_indexer_key_projection_preserves_owner_gradients_and_input(ratio, dtype):
    torch.manual_seed(324)
    params, _, _ = _packed([3, 5], [4, 6], tail=2)
    token_layout = build_csa2_thd_layout(params, 12)
    layout = token_layout.for_compression(ratio)
    config = _config(dtype)
    module = CSA2Indexer(
        config, CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear), ratio, _groups()
    )
    latent = torch.randn(layout.capacity, 1, 8, dtype=dtype)
    latent[~layout.valid_groups] = float("nan")
    latent.requires_grad_()
    reference_latent = latent.detach().clone().requires_grad_()
    wk = module.linear_wk.weight.detach().float().clone().requires_grad_()
    gamma = module.k_norm.weight.detach().float().clone().requires_grad_()
    rotary = _CPUFrequencyTable(4)
    actual = module.project_keys(latent, rotary, thd_layout=layout)
    safe_latent = reference_latent.masked_fill(~layout.valid_groups[:, None, None], 0)
    projected = F.linear(safe_latent.float(), wk).to(dtype)
    xf = projected.float()
    normalized = (xf * (xf.square().mean(-1, keepdim=True) + 1e-20).rsqrt() * gamma).to(dtype)
    expected = _rope_reference(normalized, config, rotary, layout.position_ids, layout.valid_groups)
    tolerance = dict(atol=3e-6, rtol=3e-5)
    if dtype == torch.bfloat16:
        tolerance = dict(atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(actual, expected, **tolerance)
    assert torch.isnan(latent[~layout.valid_groups]).all()
    assert torch.count_nonzero(actual[~layout.valid_groups]) == 0
    # Shared keys receive the sum of gradients from two independent downstream consumers.
    probes = [torch.randn_like(actual), torch.randn_like(actual)]
    actual_gradients = None
    if dtype == torch.float32:
        # In BF16, separate backwards round the upstream sum at different points.
        # Compare that combined backward to the FP32 oracle below instead.
        actual_gradients = [
            torch.autograd.grad((actual * probe).sum(), module.linear_wk.weight, retain_graph=True)[
                0
            ]
            for probe in probes
        ]
    sum((actual * probe).sum() for probe in probes).backward()
    sum((expected * probe).sum() for probe in probes).backward()
    torch.testing.assert_close(latent.grad, reference_latent.grad, **tolerance)
    torch.testing.assert_close(module.linear_wk.weight.grad.float(), wk.grad, **tolerance)
    torch.testing.assert_close(module.k_norm.weight.grad.float(), gamma.grad, **tolerance)
    if actual_gradients is not None:
        torch.testing.assert_close(module.linear_wk.weight.grad, sum(actual_gradients), **tolerance)
    assert torch.isfinite(latent.grad).all()
    assert torch.count_nonzero(latent.grad[~layout.valid_groups]) == 0


def _dsv4_cpu_compressor(ratio):
    """Install real CPU parameters without the constructor's CUDA-only APE allocation."""
    config = _config(torch.float32)
    config.fp8 = config.fp4 = False
    config.layernorm_epsilon = 1e-5
    module = Compressor.__new__(Compressor)
    MegatronModule.__init__(module, config)
    module.compress_ratio = ratio
    module.head_dim = 8
    module.overlap = ratio == 4
    module.coff = 2 if module.overlap else 1
    module.use_fused_compressor = False
    module.rotate = False
    module.qk_pos_emb_head_dim = 4
    module.pg_collection = _groups()
    module.rotary_pos_emb = _CPUFrequencyTable(4)
    module.linear_wkv = _Linear(6, module.coff * 8, config)
    module.linear_wgate = _Linear(6, module.coff * 8, config)
    module.ape = nn.Parameter(torch.randn(ratio, module.coff * 8) * 0.2)
    module.norm = _RMSNorm(8, config.layernorm_epsilon, config)
    return module


def _dsv4_reference_groups(x, weights, ratio, lengths, rotary):
    """Enumerate DSv4 groups, including r4's previous-group overlap and APE."""
    result, positions = [], []
    offset = 0
    for length in lengths:
        previous_values = previous_logits = None
        for group in range(length // ratio):
            tokens = x[offset + group * ratio : offset + (group + 1) * ratio]
            values = F.linear(tokens, weights["linear_wkv.weight"])
            logits = F.linear(tokens, weights["linear_wgate.weight"]) + weights["ape"][:, None]
            if ratio == 4:
                # The current group's second half and previous group's first half
                # are pooled. A sequence's first group never consumes its neighbour.
                selected_values, selected_logits = values[..., 8:], logits[..., 8:]
                if previous_values is not None:
                    selected_values = torch.cat((previous_values, selected_values))
                    selected_logits = torch.cat((previous_logits, selected_logits))
                previous_values, previous_logits = values[..., :8], logits[..., :8]
            else:
                selected_values, selected_logits = values, logits
            latent = (selected_values * selected_logits.softmax(dim=0)).sum(dim=0)
            normalized = latent * (latent.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
            result.append(normalized * weights["norm.weight"])
            positions.append(group * ratio)
        offset += length
    result = torch.stack(result)
    return _complex_rotation(
        result, 4, torch.tensor(positions), torch.ones(len(positions), dtype=torch.bool), rotary
    )


@pytest.mark.parametrize("ratio", [4, 128])
def test_dsv4_native_compressor_preserves_per_sequence_forward_and_backward(ratio):
    """Shared indexing preserves DSv4 overlap, APE, odd tails, and packed capacity."""
    torch.manual_seed(325)
    lengths = [ratio + 1, 0, 2 * ratio + 3]
    params, _, valid = _packed(lengths, lengths, tail=4 * ratio)
    module = _dsv4_cpu_compressor(ratio)
    x = torch.randn(valid.numel(), 1, 6, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in module.named_parameters()
    }
    actual, compressed_cu = module(x, packed_seq_params=params)
    expected = _dsv4_reference_groups(reference_x, weights, ratio, lengths, module.rotary_pos_emb)
    # Three metadata slots (including the empty sequence) reserve at most two
    # compressed rows each, even though the token buffer permits more than six.
    assert actual.shape == (6, 1, 8)
    assert compressed_cu.tolist() == [0, 1, 1, 3]
    torch.testing.assert_close(actual[:3], expected, atol=3e-6, rtol=3e-5)
    # DSv4's extra capacity rows are ignored by downstream masking, not defined
    # as zeros. Exclude them from the loss when checking the shared gather graph.
    probe = torch.randn_like(expected)
    (actual[:3] * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=3e-6, rtol=3e-5)
    assert torch.count_nonzero(x.grad[~valid]) == 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad,
            weights[name].grad,
            atol=4e-6,
            rtol=4e-5,
            msg=lambda detail: f"{name}: {detail}",
        )


# Complete THD attention and indexer supervision, compared to independent sequences.


def _packed_attention_cores(
    dtype=torch.float32,
    *,
    candidates=True,
    coefficient=0,
    sparse=False,
    per_token=False,
    **overrides,
):
    config_values = dict(
        params_dtype=dtype,
        csa2_candidate_source_layer=3 if candidates else None,
        csa2_candidate_topk_blocks=1 if candidates else 0,
        csa2_candidate_block_size=2 if candidates else 0,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=coefficient,
        dsa_indexer_use_sparse_loss=sparse,
        calculate_per_token_loss=per_token,
    )
    config_values.update(overrides)
    config = _make_config(**config_values)
    modules = CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(
            CSA2Compressor, submodules=CompressorSubmodules(_Linear, _Linear, _RMSNorm)
        ),
        indexer=ModuleSpec(
            CSA2Indexer, submodules=CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear)
        ),
    )
    cores = nn.ModuleList(
        CompressedSparseAttention2(
            config,
            modules,
            layer_number=index + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=_groups(),
            rotary_pos_emb=_CPUFrequencyTable(config.qk_pos_emb_head_dim),
        )
        for index in range(config.num_layers)
    )
    with torch.no_grad():
        for core in cores:
            core.attn_sink.uniform_(-1, 1)
    return cores


def _packed_core_inputs(cores, total):
    config = cores[0].config
    dtype = config.params_dtype
    return [
        (
            torch.randn(
                total,
                config.num_attention_heads,
                config.v_head_dim,
                dtype=dtype,
                requires_grad=True,
            ),
            torch.randn(total, 1, config.v_head_dim, dtype=dtype, requires_grad=True),
            torch.randn(total, 1, config.hidden_size, dtype=dtype, requires_grad=True),
            torch.randn(total, 1, config.q_lora_rank, dtype=dtype, requires_grad=True),
        )
        for _ in cores
    ]


def _run_packed_cores(cores, inputs, params):
    state = CSA2State()
    outputs = [
        core(
            # DSv4's QKV wrapper returns packed qr without the dummy batch axis.
            q,
            kv,
            kv,
            None,
            x=x,
            qr=qr.squeeze(1),
            packed_seq_params=params,
            csa2_state=state,
        )
        for core, (q, kv, x, qr) in zip(cores, inputs)
    ]
    return outputs, state


def _run_separate_cores(cores, inputs, real_lengths, physical_lengths):
    """Run independent SBHD forwards; no THD metadata/helpers enter this reference."""
    total = inputs[0][0].shape[0]
    contributions = [[] for _ in cores]
    start = 0
    for length, physical in zip(real_lengths, physical_lengths):
        if length:
            state = CSA2State()
            for index, (core, (q, kv, x, qr)) in enumerate(zip(cores, inputs)):
                selected = slice(start, start + length)
                output = core(
                    q[selected].unsqueeze(1),
                    kv[selected].unsqueeze(2),
                    kv[selected].unsqueeze(2),
                    None,
                    x=x[selected],
                    qr=qr[selected],
                    csa2_state=state,
                ).squeeze(1)
                contributions[index].append(F.pad(output, (0, 0, start, total - start - length)))
        start += physical
    return [sum(values) for values in contributions]


def _copy_core_inputs(inputs):
    return [tuple(tensor.detach().clone().requires_grad_() for tensor in row) for row in inputs]


def _assert_optional_gradients(actual, expected, *, dtype=torch.float32):
    if actual is None or expected is None:
        assert actual is expected
        return
    assert torch.isfinite(actual).all()
    tolerance = dict(atol=3e-6, rtol=3e-5)
    if dtype == torch.bfloat16:
        tolerance = dict(atol=2e-4, rtol=4e-2)
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("mask_kind", ["causal", "all-masked", "placeholder"])
def test_attention_mask_does_not_change_sparse_visibility(layout, mask_kind):
    """Like V4, dense masks do not override the sparse/packed attention layout."""
    torch.manual_seed(329)
    cores = _packed_attention_cores(torch.bfloat16)
    params, _, valid = _packed([3, 5], [4, 6], tail=3)
    inputs = _packed_core_inputs(cores, valid.numel())
    if layout == "sbhd":
        inputs = [(q.unsqueeze(1), kv.unsqueeze(1), x, qr) for q, kv, x, qr in inputs]
        params = None
    if mask_kind == "placeholder":
        mask = torch.tensor(float("nan"))
    else:
        mask = torch.ones(1, 1, valid.numel(), valid.numel(), dtype=torch.bool)
        if mask_kind == "causal":
            mask = mask.triu(1)
    masked_state, reference_state = CSA2State(), CSA2State()
    outputs, expected = [], []
    for core, (q, kv, x, qr) in zip(cores, inputs):
        kwargs = dict(x=x, qr=qr, packed_seq_params=params)
        outputs.append(core(q, kv, kv, mask, csa2_state=masked_state, **kwargs))
        expected.append(core(q, kv, kv, None, csa2_state=reference_state, **kwargs))
        torch.testing.assert_close(outputs[-1], expected[-1], rtol=0, atol=0)
    leaves = tuple(t for row in inputs for t in row) + tuple(cores.parameters())
    probes = [torch.randn_like(output) for output in outputs]
    actual_grads = torch.autograd.grad(
        sum((out * probe).sum() for out, probe in zip(outputs, probes)), leaves, allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        sum((out * probe).sum() for out, probe in zip(expected, probes)), leaves, allow_unused=True
    )
    for actual, reference in zip(actual_grads, expected_grads):
        if actual is None or reference is None:
            assert actual is reference
        else:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("candidates", [False, True])
def test_thd_attention_stack_matches_separate_sequence_outputs_and_gradients(dtype, candidates):
    # This FP32 seed ties several Full1 indexer scores at zero. Packing must
    # retain the same smaller-position tie break despite the wider KV capacity.
    torch.manual_seed(326)
    cores = _packed_attention_cores(dtype, candidates=candidates)
    reference = _packed_attention_cores(dtype, candidates=candidates)
    reference.load_state_dict(cores.state_dict())
    real, physical, dummies = [1, 5, 7, 3], [3, 7, 8, 4], (3,)
    params, _, valid = _packed(real, physical, tail=3, dummy_sequences=dummies)
    inputs = _packed_core_inputs(cores, valid.numel())
    reference_inputs = _copy_core_inputs(inputs)
    outputs, state = _run_packed_cores(cores, inputs, params)
    expected = _run_separate_cores(reference, reference_inputs, real, physical)
    assert state.thd_layout is not None and state.compressed_layout is not None
    actual_loss = expected_loss = 0
    for output, ref_output in zip(outputs, expected):
        tolerance = (
            dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=1e-2, rtol=2e-2)
        )
        torch.testing.assert_close(output, ref_output, **tolerance)
        assert torch.count_nonzero(output[~valid]) == 0
        probe = torch.randn_like(output)
        normalization = valid.sum() * output.shape[-1]
        actual_loss = actual_loss + (output.float() * probe).sum() / normalization
        expected_loss = expected_loss + (ref_output.float() * probe).sum() / normalization
    actual_loss.backward()
    expected_loss.backward()
    for row, reference_row in zip(inputs, reference_inputs):
        for tensor, reference_tensor in zip(row, reference_row):
            _assert_optional_gradients(tensor.grad, reference_tensor.grad, dtype=dtype)
            if tensor.grad is not None:
                assert torch.count_nonzero(tensor.grad[~valid]) == 0
    for parameter, reference_parameter in zip(cores.parameters(), reference.parameters()):
        _assert_optional_gradients(parameter.grad, reference_parameter.grad, dtype=dtype)


def test_thd_candidate_score_ties_are_invariant_to_masked_capacity():
    scores = torch.zeros(1, 1, 6)
    padded = F.pad(scores, (0, 7), value=-torch.inf)
    visible = torch.tensor([[[6]]])
    expected = torch.tensor([[[True, True, False, False, True, True]]])
    actual = select_candidate_blocks(scores, visible, 2, 2)
    with_capacity = select_candidate_blocks(padded, visible, 2, 2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(with_capacity[..., :6], expected)
    assert not with_capacity[..., 6:].any()


def _capture_core_indexer_losses(monkeypatch, cores):
    records = {}
    for core in cores:
        if core.indexer is None:
            continue
        losses = records.setdefault(core.layer_idx, [])
        original = core._compute_indexer_loss

        def record(*args, _original=original, _losses=losses, **kwargs):
            loss = _original(*args, **kwargs)
            _losses.append(loss)
            return loss

        monkeypatch.setattr(core, "_compute_indexer_loss", record)
    return records


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_thd_indexer_loss_matches_token_weighted_separate_sequences(monkeypatch, sparse, per_token):
    torch.manual_seed(327)
    logged = _record_losses(monkeypatch)
    cores = _packed_attention_cores(coefficient=0.3, sparse=sparse, per_token=per_token)
    reference = _packed_attention_cores(coefficient=0.3, sparse=sparse, per_token=per_token)
    reference.load_state_dict(cores.state_dict())
    packed_losses = _capture_core_indexer_losses(monkeypatch, cores)
    separate_losses = _capture_core_indexer_losses(monkeypatch, reference)
    real, physical, dummies = [1, 5, 7, 3], [3, 7, 8, 4], (3,)
    params, _, valid = _packed(real, physical, tail=3, dummy_sequences=dummies)
    inputs = _packed_core_inputs(cores, valid.numel())
    reference_inputs = _copy_core_inputs(inputs)
    _, state = _run_packed_cores(cores, inputs, params)
    packed_logged = list(logged)
    _run_separate_cores(reference, reference_inputs, real, physical)
    assert list(packed_losses) == [1, 3, 4]
    assert [record["layer_number"] for record in packed_logged] == [2, 4, 5]
    # The dummy sequence is included exactly as in DSv4; a one-token sequence
    # also counts in the mean even though it has no complete r2 group.
    assert valid.sum() == 16
    actual_terms, expected_terms = [], []
    for index, losses in packed_losses.items():
        assert len(losses) == 1
        actual = losses[0]
        separate = separate_losses[index]
        assert len(separate) == 4
        if per_token:
            expected = sum(separate)
        else:
            expected = sum(loss * length for loss, length in zip(separate, real)) / 16
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        metric = next(
            record["loss"] for record in packed_logged if record["layer_number"] == index + 1
        )
        torch.testing.assert_close(metric, actual.detach() / (16 if per_token else 1))
        actual_terms.append(actual)
        expected_terms.append(expected)
    # Owner K receives both the Full and Reindex auxiliary gradients.
    owner_weight = cores[3].indexer.linear_wk.weight
    individual = [
        torch.autograd.grad(loss, owner_weight, retain_graph=True)[0] for loss in actual_terms[1:]
    ]
    sum(actual_terms).backward()
    sum(expected_terms).backward()
    torch.testing.assert_close(owner_weight.grad, sum(individual), atol=3e-6, rtol=3e-5)
    assert owner_weight.grad.abs().sum() > 0
    for core, reference_core in zip(cores, reference):
        for name, parameter in core.named_parameters():
            reference_parameter = dict(reference_core.named_parameters())[name]
            if name.startswith("indexer."):
                _assert_optional_gradients(parameter.grad, reference_parameter.grad)
            else:
                assert parameter.grad is None, name
    assert all(tensor.grad is None for row in inputs for tensor in row)
    # The main compressed KV remains graph-connected independently of indexer supervision.
    state.global_kv.float().square().sum().backward()
    assert cores[3].compressor.linear_wkv.weight.grad.abs().sum() > 0


def test_thd_padding_and_other_sequences_cannot_contaminate_attention_or_gradients():
    torch.manual_seed(328)
    cores = _packed_attention_cores()
    real, physical, dummies = [1, 5, 7, 3], [3, 7, 8, 4], (3,)
    params, _, valid = _packed(real, physical, tail=3, dummy_sequences=dummies)
    inputs = _packed_core_inputs(cores, valid.numel())
    contaminated = _copy_core_inputs(inputs)
    with torch.no_grad():
        for row in contaminated:
            for tensor in row:
                tensor[~valid] = float("nan")
                tensor[10:17] = torch.randn_like(tensor[10:17]) * 10
    outputs, _ = _run_packed_cores(cores, inputs, params)
    changed, _ = _run_packed_cores(cores, contaminated, params)
    for output, changed_output in zip(outputs, changed):
        assert torch.isfinite(changed_output).all()
        torch.testing.assert_close(changed_output[3:8], output[3:8], atol=0, rtol=0)
        assert torch.count_nonzero(changed_output[~valid]) == 0
    actual_grads = torch.autograd.grad(
        sum(output[3:8].sum() for output in changed),
        [tensor for row in contaminated for tensor in row],
        allow_unused=True,
    )
    reference_grads = torch.autograd.grad(
        sum(output[3:8].sum() for output in outputs),
        [tensor for row in inputs for tensor in row],
        allow_unused=True,
    )
    for actual, expected in zip(actual_grads, reference_grads):
        _assert_optional_gradients(actual, expected)
        if actual is not None:
            assert torch.count_nonzero(actual[:3]) == 0
            assert torch.count_nonzero(actual[8:]) == 0


@pytest.mark.parametrize(
    "real,physical,tail,dummies",
    [
        pytest.param([], [], 0, (), id="zero-tokens"),
        pytest.param([0, 0], [3, 4], 3, (), id="only-padding"),
        pytest.param([1, 1], [1, 1], 2, (), id="no-complete-r2-groups"),
    ],
)
def test_thd_empty_groups_and_padding_have_finite_outputs_and_losses(
    monkeypatch, real, physical, tail, dummies
):
    logged = _record_losses(monkeypatch)
    cores = _packed_attention_cores(coefficient=0.3)
    params, _, valid = _packed(real, physical, tail=tail, dummy_sequences=dummies)
    inputs = _packed_core_inputs(cores, valid.numel())
    outputs, _ = _run_packed_cores(cores, inputs, params)
    for output in outputs:
        assert output.shape == (valid.numel(), 64)
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[~valid]) == 0
    assert all(torch.isfinite(record["loss"]) and record["loss"] == 0 for record in logged)
    sum(output.float().sum() for output in outputs).backward()
    for row in inputs:
        for tensor in row:
            if tensor.grad is not None:
                assert torch.isfinite(tensor.grad).all()
                assert torch.count_nonzero(tensor.grad[~valid]) == 0
    for parameter in cores.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_thd_reuse_forward_rejects_changed_sequence_boundaries():
    cores = _packed_attention_cores()
    params, _, valid = _packed([3, 5], [4, 6])
    changed_params, _, _ = _packed([4, 4], [4, 6])
    inputs = _packed_core_inputs(cores, valid.numel())
    state = CSA2State()
    q, kv, x, qr = inputs[1]
    cores[1](q, kv, kv, None, x=x, qr=qr, packed_seq_params=params, csa2_state=state)
    q, kv, x, qr = inputs[2]
    with pytest.raises(ValueError, match="different THD layout"):
        cores[2](q, kv, kv, None, x=x, qr=qr, packed_seq_params=changed_params, csa2_state=state)


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_thd_attention_wrapper_matches_independent_sequences(pg_collection, ratio, dtype):
    layer = _layer(pg_collection, ratio, dtype)
    params, _, valid = _packed([3, 5], [4, 6], tail=2, device="cuda")
    x = torch.randn(12, 1, layer.config.hidden_size, dtype=dtype, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    output, _ = layer(x, None, packed_seq_params=params)
    first = layer(reference_x[:3], None)[0]
    second = layer(reference_x[4:9], None)[0]
    expected = F.pad(first, (0, 0, 0, 0, 0, 9)) + F.pad(second, (0, 0, 0, 0, 4, 3))
    tolerance = dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=3e-2)
    # DSv4's wrapper leaves padding output values unspecified; only logical
    # sequence rows contribute to the reference objective.
    torch.testing.assert_close(output[valid], expected[valid], **tolerance)
    probe = (torch.randn_like(output) / output.numel()).masked_fill(~valid[:, None, None], 0)
    parameters = tuple(layer.parameters())
    actual_grads = torch.autograd.grad((output * probe).sum(), (x, *parameters), allow_unused=True)
    expected_grads = torch.autograd.grad(
        (expected * probe).sum(), (reference_x, *parameters), allow_unused=True
    )
    for actual, expected_gradient in zip(actual_grads, expected_grads):
        _assert_optional_gradients(actual, expected_gradient, dtype=dtype)


# Fused attention dispatch, shared gradients, and RoPE storage ownership.


@pytest.fixture
def reset_fused_indexer_loss_state(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})


def _require_sparse_kernels():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("CSA2 sparse attention requires SM90+")
    # Match the existing V4 real-kernel tests: the CI FlashMLA build ships
    # SM100 sparse kernels only, although an SM90 build is supported by CSA.
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("CI FlashMLA sparse kernels require SM100+")
    pytest.importorskip("flash_mla")
    frontend = pytest.importorskip("cudnn")
    if not hasattr(frontend, "DSA"):
        pytest.skip("cuDNN Frontend DSA is unavailable")


def _run_sbhd_cores(cores, inputs):
    state = CSA2State()
    outputs = [
        core(q, kv, kv, None, x=x, qr=qr, csa2_state=state)
        for core, (q, kv, x, qr) in zip(cores, inputs)
    ]
    return outputs, state


def _fused_core_inputs(cores, layout, device):
    # Allocate only the metric buffer on the test device; supervision and
    # the production logger/autoscaler still execute unchanged.
    DSAIndexerLossLoggingHelper.tracker["values"] = torch.zeros(len(cores), device=device)
    params = valid = None
    if layout == "sbhd":
        # B=2 detects accidental batch-major flattening and missing batch offsets.
        rows = 18
    else:
        real, physical, tail = {
            "thd": ([1, 5, 7, 2], [3, 6, 8, 2], 3),
            "empty": ([0, 0], [0, 0], 0),
            "padding": ([0, 0], [3, 4], 2),
        }[layout]
        params, _, valid = _packed(real, physical, tail=tail, device=device)
        rows = valid.numel()
    inputs = _packed_core_inputs(cores, rows)
    inputs = [tuple(t.to(device).detach().requires_grad_() for t in row) for row in inputs]
    if layout == "sbhd":
        inputs = [
            tuple(
                (
                    t.reshape(9, 2, *t.shape[2:]).detach().requires_grad_()
                    if i in (2, 3)
                    else t.reshape(9, 2, *t.shape[1:]).detach().requires_grad_()
                )
                for i, t in enumerate(row)
            )
            for row in inputs
        ]
    else:
        with torch.no_grad():
            for row in inputs:
                for tensor in row:
                    tensor[~valid] = float("nan")
    return inputs, params, valid


def _assert_fused_close(actual, expected, *, exact, name):
    if actual is None or expected is None:
        assert actual is expected, name
        return
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(expected).all(), name
    if exact:
        torch.testing.assert_close(
            actual, expected, atol=2e-5, rtol=2e-4, msg=lambda detail: f"{name}: {detail}"
        )
    else:
        # BF16 fused GEMMs round at different points from the FP32 native
        # reference. Check both scale and direction, including near-zero grads.
        error = (actual.float() - expected.float()).norm()
        scale = expected.float().norm().clamp_min(1e-5)
        assert error / scale < 0.04, f"{name}: relative L2 error {error / scale}"


@pytest.mark.parametrize("backend", ["adapter", "real"])
@pytest.mark.parametrize("layout", ["sbhd", "thd", "empty", "padding"])
@pytest.mark.parametrize("coefficient", [0, 0.3])
@pytest.mark.usefixtures("reset_fused_indexer_loss_state")
def test_fused_sparse_attention_stack_forward_backward(monkeypatch, backend, layout, coefficient):
    """Six-layer stack, r2/r1, native supervision, padding, and shared KV gradients."""
    torch.manual_seed(330)
    real = backend == "real"
    if real:
        _require_sparse_kernels()
    device = "cuda" if real else "cpu"
    options = (
        dict(
            num_attention_heads=64, v_head_dim=512, dsa_indexer_n_heads=32, dsa_indexer_head_dim=128
        )
        if real
        else {}
    )
    reference = _packed_attention_cores(torch.bfloat16, coefficient=coefficient, **options).to(
        device
    )
    cores = _packed_attention_cores(
        torch.bfloat16,
        coefficient=coefficient,
        dsa_kernel_backend="cudnn" if real else "none",
        **options,
    ).to(device)
    cores.load_state_dict(reference.state_dict())
    # Keep this regression focused on main attention and native supervision.
    # Fused indexer selection is checked separately against its rounded-input oracle.
    for core in cores:
        if core.indexer is not None:
            core.indexer.use_fused_kernels = False
    calls = []
    if not real:
        for core in cores:
            core.use_fused_kernels = True

        def flat_attention(
            q,
            kv,
            sink,
            indices,
            scale,
            *,
            topk_length,
            is_thd=False,
            kv_reconstruction_parts=None,
            q_padding_mask=None,
        ):
            assert q.dtype == kv.dtype == torch.bfloat16
            assert sink.dtype == torch.float32
            assert indices.dtype == torch.int32 and indices.is_contiguous()
            assert indices.ndim == 2
            assert topk_length.dtype == torch.int32 and topk_length.is_contiguous()
            prefix = torch.arange(indices.shape[-1])[None, :] < topk_length[:, None]
            assert torch.equal(indices >= 0, prefix)
            if q_padding_mask is not None:
                assert torch.equal(q_padding_mask, topk_length == 0)
            calls.append(indices.clone())
            # Simulate the existing kernel's flat-coordinate contract only.
            # The independent path below uses native SBHD or packed coordinates.
            flat_q = q.reshape(-1, q.shape[-2], q.shape[-1])
            output = unfused_compressed_sparse_attn(
                flat_q, kv.reshape(-1, kv.shape[-1]).float(), sink, indices, scale
            )
            return output if is_thd else output.reshape(q.shape[0], q.shape[1], -1)

        monkeypatch.setattr(
            "megatron.core.transformer.experimental_attention_variant.csa2.csa_sparse_attn",
            flat_attention,
        )

    inputs, params, valid = _fused_core_inputs(cores, layout, device)
    ref_inputs = _copy_core_inputs(inputs)
    run = _run_sbhd_cores if layout == "sbhd" else lambda c, x: _run_packed_cores(c, x, params)
    outputs, state = run(cores, inputs)
    expected, ref_state = run(reference, ref_inputs)
    if not real:
        assert len(calls) == (0 if layout == "empty" else 6)
    torch.testing.assert_close(state.global_indices, ref_state.global_indices)
    loss = ref_loss = 0
    for i, (out, ref) in enumerate(zip(outputs, expected)):
        _assert_fused_close(out, ref, exact=not real, name=f"layer {i} output")
        if valid is not None:
            assert torch.count_nonzero(out[~valid]) == 0
        probe = torch.randn_like(out) * 0.01
        loss = loss + (out * probe).sum()
        ref_loss = ref_loss + (ref * probe).sum()
    loss.backward()
    ref_loss.backward()
    for i, (row, ref_row) in enumerate(zip(inputs, ref_inputs)):
        for j, (tensor, ref_tensor) in enumerate(zip(row, ref_row)):
            _assert_fused_close(
                tensor.grad, ref_tensor.grad, exact=not real, name=f"layer {i} input {j}"
            )
            if valid is not None and tensor.grad is not None:
                assert torch.count_nonzero(tensor.grad[~valid]) == 0
    ref_parameters = dict(reference.named_parameters())
    for name, parameter in cores.named_parameters():
        _assert_fused_close(parameter.grad, ref_parameters[name].grad, exact=not real, name=name)
    if layout not in ("empty", "padding"):
        for owner in (1, 3):
            assert cores[owner].compressor.linear_wkv.weight.grad.abs().sum() > 0
            if coefficient:
                assert cores[owner].indexer.linear_wk.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton RoPE requires CUDA")
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.usefixtures("reset_fused_indexer_loss_state")
def test_fused_rope_preserves_shared_latent_and_indexer_backward(layout, dtype):
    torch.manual_seed(331)
    reference = _packed_attention_cores(dtype, coefficient=0.3).cuda()
    cores = _packed_attention_cores(dtype, coefficient=0.3, apply_rope_fusion=True).cuda()
    cores.load_state_dict(reference.state_dict())
    saved = []

    def retain_output(module, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        saved.append((tensor, tensor.detach().clone()))

    for stack in (cores, reference):
        for core in stack:
            core.rotary_pos_emb = _rotary_module(
                core.config, bool(core.compress_ratio), _groups().cp
            )
    for core in cores:
        if core.is_kv_source:
            core.compressor.register_forward_hook(retain_output)
            core.indexer.k_norm.register_forward_hook(retain_output)
    inputs, params, _ = _fused_core_inputs(cores, layout, "cuda")
    ref_inputs = _copy_core_inputs(inputs)
    run = _run_sbhd_cores if layout == "sbhd" else lambda c, x: _run_packed_cores(c, x, params)
    outputs, _ = run(cores, inputs)
    expected, _ = run(reference, ref_inputs)
    for tensor, before in saved:
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
    loss = ref_loss = 0
    for out, ref in zip(outputs, expected):
        _assert_fused_close(out, ref, exact=dtype == torch.float32, name="RoPE output")
        probe = torch.randn_like(out) * 0.01
        loss = loss + (out * probe).sum()
        ref_loss = ref_loss + (ref * probe).sum()
    loss.backward()
    ref_loss.backward()
    for row, ref_row in zip(inputs, ref_inputs):
        for tensor, ref_tensor in zip(row, ref_row):
            _assert_fused_close(
                tensor.grad, ref_tensor.grad, exact=dtype == torch.float32, name="RoPE input"
            )
    for (name, p), (_, ref) in zip(cores.named_parameters(), reference.named_parameters()):
        _assert_fused_close(p.grad, ref.grad, exact=dtype == torch.float32, name=name)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.usefixtures("reset_fused_indexer_loss_state")
def test_fused_attention_wrapper_with_rope(pg_collection, packed, ratio):
    """Real TE projections/norms, Q/local-KV RoPE, sparse attention and inverse RoPE."""
    _require_sparse_kernels()
    options = dict(
        hidden_size=128,
        num_attention_heads=64,
        v_head_dim=512,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
    )
    reference = _layer(pg_collection, ratio, torch.bfloat16, **options)
    layer = _layer(
        pg_collection,
        ratio,
        torch.bfloat16,
        dsa_kernel_backend="cudnn",
        apply_rope_fusion=True,
        **options,
    )
    layer.load_state_dict(reference.state_dict())
    if layer.core_attention.indexer is not None:
        layer.core_attention.indexer.use_fused_kernels = False
    saved = []
    layer.kv_layernorm.register_forward_hook(
        lambda module, args, output: saved.append((output, output.detach().clone()))
    )
    params = valid = None
    if packed:
        params, _, valid = _packed([1, 5, 7], [3, 6, 8], tail=3, device="cuda")
        shape = (valid.numel(), 1, 128)
    else:
        shape = (9, 2, 128)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    output = layer(x, attention_mask=None, packed_seq_params=params)[0]
    expected = reference(ref_x, attention_mask=None, packed_seq_params=params)[0]
    for tensor, before in saved:
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
    _assert_fused_close(output, expected, exact=False, name="wrapper output")
    probe = torch.randn_like(output) * 0.01
    (output * probe).sum().backward()
    (expected * probe).sum().backward()
    _assert_fused_close(x.grad, ref_x.grad, exact=False, name="wrapper input gradient")
    if valid is not None:
        assert torch.count_nonzero(output[~valid]) == 0
        assert torch.count_nonzero(x.grad[~valid]) == 0
    for (name, p), (_, ref) in zip(layer.named_parameters(), reference.named_parameters()):
        _assert_fused_close(p.grad, ref.grad, exact=False, name=name)


def _sorted_valid_indices(scores, topk):
    """Independent selection oracle, with invalid slots after ascending valid keys."""
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk]
    ids = ids.masked_fill(~scores.gather(-1, ids).isfinite(), torch.iinfo(torch.int32).max)
    ids = ids.sort(dim=-1).values
    return ids.masked_fill(ids == torch.iinfo(torch.int32).max, -1).int()


def _reference_fused_indexer_scores(
    indexer, q, k, weights, *, candidates=None, thd_layout=None, compressed_layout=None
):
    """Dense oracle for rounded-input selection, independent of fused dispatch/layout lowering."""
    if indexer.precision == "mxfp8":
        q_data, q_scale = _reference_quantize(q)
        k_data, k_scale = _reference_quantize(k)
        q = (q_data.float().unflatten(-1, (-1, 32)) * q_scale.unsqueeze(-1)).flatten(-2)
        k = (k_data.float().unflatten(-1, (-1, 32)) * k_scale.unsqueeze(-1)).flatten(-2)
    scale = indexer.head_dim**-0.5 * indexer.n_heads**-0.5
    w = (weights.float() * scale).to(weights.dtype).float()
    if thd_layout is None:
        dots = torch.einsum("sbhd,cbd->bshc", q.float(), k.float()).relu()
        scores = (dots * w.permute(1, 0, 2).unsqueeze(-1)).sum(-2)
        visible = torch.arange(1, q.shape[0] + 1, device=q.device) // indexer.compress_ratio
        valid = torch.arange(k.shape[0], device=k.device)[None, :] < visible[:, None]
    else:
        dots = torch.einsum("thd,cd->thc", q.float(), k.float()).relu()
        scores = (dots * w.unsqueeze(-1)).sum(-2)
        valid = (
            thd_layout.valid_tokens[:, None]
            & compressed_layout.valid_groups[None, :]
            & (thd_layout.sequence_ids[:, None] == compressed_layout.sequence_ids[None, :])
            & (
                compressed_layout.position_ids[None, :] + indexer.compress_ratio - 1
                <= thd_layout.position_ids[:, None]
            )
        )
    if candidates is not None:
        valid = valid & candidates.to_mask(
            k.shape[0], thd_layout=thd_layout, compressed_layout=compressed_layout
        )
    return scores.masked_fill(~valid, -torch.inf)


def _reference_fused_indexer(indexer, q, k, weights, **kwargs):
    scores = _reference_fused_indexer_scores(indexer, q, k, weights, **kwargs)
    return _sorted_valid_indices(scores, indexer.topk)


def _reference_prepared_indexer(indexer, inputs, *, return_candidates=False):
    """Restore oracle coordinates as views of the prepared projections."""
    q, k, w = inputs.q, inputs.k, inputs.weights
    if inputs.thd_layout is None:
        batch, seq_len = inputs.output_shape
        q = q.view(batch, seq_len, *q.shape[-2:]).transpose(0, 1)
        k = k.view(batch, inputs.key_capacity, k.shape[-1]).transpose(0, 1)
        w = w.view(batch, seq_len, w.shape[-1]).transpose(0, 1)
    kwargs = dict(thd_layout=inputs.thd_layout, compressed_layout=inputs.compressed_layout)
    indices = _reference_fused_indexer(indexer, q, k, w, candidates=inputs.candidates, **kwargs)
    if return_candidates:
        return indices, _reference_candidate_blocks(indexer, q, k, w, **kwargs)
    return indices


def _emulate_v4_indexer_topk_core(q, k, weights, topk, ratio, **kwargs):
    """CPU adapter for V4 core's prepared-layout and already-scaled weight contract."""
    assert q.is_contiguous() and k.is_contiguous() and weights.is_contiguous()
    assert q.dtype == k.dtype == weights.dtype == torch.bfloat16
    assert kwargs["deterministic"] is True
    assert kwargs["use_compact"] is True
    if kwargs["precision"] == "mxfp8":
        assert q.shape[-2:] == (64, 128) and weights.shape[-1] == 64
    w = weights.float()
    if "cu_seqlens_q" in kwargs:
        cu_q, cu_k = kwargs["cu_seqlens_q"], kwargs["cu_seqlens_kv"]
        assert cu_q.dtype == cu_k.dtype == torch.int32
        assert cu_q[-1] == q.shape[0] and cu_k[-1] == k.shape[0]
        assert cu_q.diff().max() <= kwargs["max_seqlen_q"]
        assert cu_k.diff().max() <= kwargs["max_seqlen_kv"]
        assert kwargs["max_seqlen_q"] <= kwargs["max_seqlen_kv"] * ratio
        scores = torch.full((q.shape[0], max(topk, kwargs["max_seqlen_kv"])), -torch.inf)
        for start, end, k_start, k_end in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
            dots = torch.einsum(
                "thd,cd->thc", q[start:end].float(), k[k_start:k_end].float()
            ).relu()
            local_scores = (dots * w[start:end, :, None]).sum(-2)
            visible = torch.arange(1, end - start + 1) // ratio
            valid = torch.arange(k_end - k_start)[None, :] < visible[:, None]
            scores[start:end, : k_end - k_start] = local_scores.masked_fill(~valid, -torch.inf)
    else:
        assert q.shape[1] <= k.shape[1] * ratio
        dots = torch.einsum("bshd,bcd->bshc", q.float(), k.float()).relu()
        scores = (dots * w.unsqueeze(-1)).sum(-2)
        visible = torch.arange(1, q.shape[1] + 1) // ratio
        valid = torch.arange(k.shape[1])[None, :] < visible[:, None]
        scores = scores.masked_fill(~valid, -torch.inf)
        scores = F.pad(scores, (0, max(0, topk - k.shape[1])), value=-torch.inf)
    indices = _sorted_valid_indices(scores, topk).flip(-1).contiguous()
    return indices, (indices >= 0).sum(-1).int(), None, None


def _kernel_indexer(ratio, precision, heads, device):
    config = _config(torch.bfloat16)
    config.dsa_indexer_n_heads, config.dsa_indexer_head_dim = heads, 128
    config.dsa_indexer_topk = 7
    indexer = CSA2Indexer(
        config, CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear), ratio, _groups()
    ).to(device)
    # Exercise the adapter independently of config validation and main attention.
    indexer.use_fused_kernels, indexer.precision = True, precision
    return indexer


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize(
    "real, physical, tail, bound",
    [
        ([3, 5], [4, 6], 0, 6),
        ([1, 0, 5], [3, 0, 7], 3, 7),
        ([1, 1], [1, 1], 31, 3),
        ([0, 0], [0, 0], 0, 0),
        ([0, 0], [0, 0], 9, 4),
        ([], [], 16, 0),
    ],
)
def test_packed_indexer_metadata_uses_host_bounds(monkeypatch, ratio, real, physical, tail, bound):
    params, _, valid = _packed(real, physical, tail=tail)
    params.max_seqlen_q = params.max_seqlen_kv = bound
    layout = build_csa2_thd_layout(params, valid.numel())
    compressed = layout.for_compression(ratio)
    inputs = prepare_csa2_indexer_inputs(
        torch.empty(valid.numel(), 2, 8, dtype=torch.bfloat16),
        torch.empty(compressed.capacity, 8, dtype=torch.bfloat16),
        torch.empty(valid.numel(), 2, dtype=torch.bfloat16),
        ratio,
        thd_layout=layout,
        compressed_layout=compressed,
    )

    def reject_host_read(*args, **kwargs):
        pytest.fail("Packed indexer metadata must not read tensor values on the host")

    with monkeypatch.context() as patch:
        for method in ("tolist", "item", "cpu", "numpy", "__bool__", "__int__", "__index__"):
            patch.setattr(torch.Tensor, method, reject_host_read)
        metadata = inputs.packed_metadata
        assert inputs.packed_metadata is metadata
    cu_q, cu_k, max_q, max_k = metadata
    assert cu_q.dtype == cu_k.dtype == torch.int32
    assert cu_q.is_contiguous() and cu_k.is_contiguous() and cu_q.shape == cu_k.shape
    # Every original sequence keeps its addresses. Synthetic segments cover
    # all physical capacity, including tails larger than the real-sequence bound.
    prefix_size = len(physical) + 1
    torch.testing.assert_close(cu_q[:prefix_size], layout.cu_seqlens_padded)
    torch.testing.assert_close(cu_k[:prefix_size], compressed.cu_seqlens_padded)
    assert cu_q[-1] == inputs.q.shape[0] and cu_k[-1] == inputs.k.shape[0]
    assert (cu_q.diff() >= 0).all() and cu_q.diff().max() <= max_q
    assert (cu_k.diff() >= 0).all() and cu_k.diff().max() <= max_k
    assert max_q == max(bound, 1) and max_k == bound // ratio


@pytest.mark.parametrize("backend", ["adapter", "real"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize(
    "layout", ["sbhd", "thd", "empty", "short", "padding", "unused-capacity", "long-tail"]
)
def test_fused_indexer_topk_layout_and_precision(
    monkeypatch, backend, precision, heads, ratio, layout
):
    """r1/r2 odd tails, per-sequence padding, int64 metadata, H32 MXFP8 padding, and empty K."""
    real = backend == "real"
    if real:
        _require_sparse_kernels()
    device = "cuda" if real else "cpu"
    torch.manual_seed(412)
    indexer = _kernel_indexer(ratio, precision, heads, device)
    calls = []
    if not real:

        def adapter(q, k, w, **kwargs):
            calls.append(kwargs)
            if precision == "bf16" or heads == 64:
                assert q.data_ptr() == prepared.q.data_ptr(), "BSHD Q must be a view"
            if layout != "sbhd" or ratio == 1:
                assert k.data_ptr() == prepared.k.data_ptr(), "Only the odd r2 tail needs padded K"
            if precision == "mxfp8" and heads == 32:
                assert torch.count_nonzero(q[..., 32:, :]) == 0
                assert torch.count_nonzero(w[..., 32:]) == 0
            return _emulate_v4_indexer_topk_core(q, k, w, **kwargs)

        monkeypatch.setattr(
            "megatron.core.transformer.experimental_attention_variant.csa2._indexer_topk_core",
            adapter,
        )
    token_layout = compressed = None
    if layout == "sbhd":
        q = torch.randn(129, 2, heads, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(129 // ratio, 2, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(129, 2, heads, device=device, dtype=torch.bfloat16)
    else:
        lengths, padded, tail = {
            "thd": ([1, 0, 57, 129], [3, 0, 62, 133], 3),
            "empty": ([0, 0], [0, 0], 0),
            "short": ([1, 1], [1, 1], 0),
            "padding": ([0, 0], [3, 5], 3),
            "unused-capacity": ([1, 1], [1, 1], 5),
            "long-tail": ([3, 0, 5], [4, 0, 6], 31),
        }[layout]
        params, _, valid = _packed(lengths, padded, tail=tail, device=device)
        if layout == "unused-capacity":
            params.max_seqlen_q = params.max_seqlen_kv = 3
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ):
            setattr(params, name, getattr(params, name).long())
        token_layout = build_csa2_thd_layout(params, valid.numel())
        compressed = token_layout.for_compression(ratio)
        q = torch.randn(valid.numel(), heads, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(compressed.capacity, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(valid.numel(), heads, device=device, dtype=torch.bfloat16)
        q[~valid], w[~valid], k[~compressed.valid_groups] = 0, 0, 0
    q_before, k_before, w_before = q.clone(), k.clone(), w.clone()
    prepared = prepare_csa2_indexer_inputs(
        q, k, w, ratio, thd_layout=token_layout, compressed_layout=compressed
    )
    prepared_before = [t.clone() for t in (prepared.q, prepared.k, prepared.weights)]
    actual = indexer._fused_topk(prepared)
    if not real:
        # The CPU adapter checks precision dispatch, not the CUDA quantizer's arithmetic.
        indexer.precision = "bf16"
    expected = _reference_fused_indexer(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    torch.testing.assert_close(actual, expected)
    for original, before in ((q, q_before), (k, k_before), (w, w_before)):
        torch.testing.assert_close(original, before, rtol=0, atol=0)
    for tensor, before in zip((prepared.q, prepared.k, prepared.weights), prepared_before):
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
    assert actual.dtype == torch.int32 and actual.is_contiguous()
    if not real:
        assert len(calls) == (0 if q.shape[0] == 0 or k.shape[0] == 0 else 1)
    if real and precision == "mxfp8" and layout == "sbhd":
        indexer.precision = "bf16"
        unquantized = _reference_fused_indexer(indexer, q, k, w)
        assert not torch.equal(actual, unquantized), "MXFP8 must actually change quantized ranking"


@pytest.mark.parametrize("supervised", [False, True])
def test_fused_indexer_without_loss_avoids_dense_scores(monkeypatch, supervised):
    indexer = _kernel_indexer(2, "bf16", 32, "cpu")
    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.csa2._indexer_topk_core",
        _emulate_v4_indexer_topk_core,
    )

    def reject_dense(*args, **kwargs):
        pytest.fail(
            "Ordinary fused selection without supervision must not materialize dense scores"
        )

    monkeypatch.setattr(indexer, "_score_projected", reject_dense)
    k = indexer.project_keys(torch.randn(4, 2, 8, dtype=torch.bfloat16), _CPUFrequencyTable(4))
    k_flat = k.transpose(0, 1).reshape(-1, 128).contiguous()
    selected_inputs = []
    select = indexer._fused_topk

    def record_selection(prepared):
        assert not torch.is_grad_enabled()
        assert prepared.k is k_flat
        assert prepared.q.requires_grad == prepared.weights.requires_grad == supervised
        selected_inputs.append(prepared)
        return select(prepared)

    monkeypatch.setattr(indexer, "_fused_topk", record_selection)
    with torch.set_grad_enabled(supervised):
        indices, loss_inputs, candidate_scores = indexer.forward_with_scores(
            torch.randn(9, 2, 6, dtype=torch.bfloat16),
            torch.randn(9, 2, 4, dtype=torch.bfloat16),
            None,
            _CPUFrequencyTable(4),
            indexer_k=k,
            indexer_k_flat=k_flat,
            return_loss_inputs=supervised,
        )
    assert indices.shape == (2, 9, 4)
    assert candidate_scores is None
    assert len(selected_inputs) == 1
    assert loss_inputs is (selected_inputs[0] if supervised else None)


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("layout", ["sbhd", "thd", "padding"])
@pytest.mark.parametrize("candidates", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.usefixtures("reset_fused_indexer_loss_state")
def test_fused_indexer_stack_shared_gradients(monkeypatch, precision, layout, candidates, sparse):
    """Compare real kernels to dense rounded-input selection and native attention/loss."""
    _require_sparse_kernels()
    torch.manual_seed(413)
    options = dict(
        num_attention_heads=64,
        v_head_dim=512,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        candidates=candidates,
        coefficient=0.3,
        sparse=sparse,
    )
    reference = _packed_attention_cores(torch.bfloat16, **options).cuda()
    cores = _packed_attention_cores(
        torch.bfloat16, dsa_kernel_backend="cudnn", dsa_indexer_precision=precision, **options
    ).cuda()
    cores.load_state_dict(reference.state_dict())
    for core in reference:
        if core.indexer is not None:
            indexer = core.indexer
            indexer.use_fused_kernels, indexer.precision = True, precision
            indexer._fused_topk = lambda prepared, idx=indexer: _reference_prepared_indexer(
                idx, prepared
            )
            indexer._candidate_blocks = (
                lambda q, k, w, idx=indexer, **kwargs: _reference_candidate_blocks(
                    idx, q, k, w, **kwargs
                )
            )
            indexer._fused_topk_and_candidates = lambda prepared, idx=indexer: (
                _reference_prepared_indexer(idx, prepared, return_candidates=True)
            )
    inputs, params, valid = _fused_core_inputs(cores, layout, "cuda")
    ref_inputs = _copy_core_inputs(inputs)
    calls = []
    shared_buffers = {}
    prepared_inputs = {}
    loss_inputs_seen = []
    kernel_outputs = []
    compact_inputs = {}
    backward_calls = []
    window_lse_calls = []
    final_index_calls = []
    compressor_inputs = {}
    compressor_projections = []
    window_lse = csa2_indexer.fused_csa_window_lse
    attention_forward = csa2_indexer.fused_sparse_attention._csa_fwd_flash_mla
    precompute_loss = csa2_indexer._precompute_indexer_loss

    def record_loss_inputs(q, k, w, *args, **kwargs):
        prepared = next(reversed(prepared_inputs.values()))
        assert all(t.requires_grad for t in (prepared.q, prepared.k, prepared.weights))
        assert (q.data_ptr(), k.data_ptr(), w.data_ptr()) == tuple(
            t.data_ptr() for t in (prepared.q, prepared.k, prepared.weights)
        ), "Selection and loss must consume the same prepared projection storage"
        loss_inputs_seen.append(prepared)
        return precompute_loss(q, k, w, *args, **kwargs)

    monkeypatch.setattr(csa2_indexer, "_precompute_indexer_loss", record_loss_inputs)

    def record_attention_forward(*args, **kwargs):
        indices, lengths = args[2], kwargs["topk_length"]
        assert lengths.dtype == torch.int32 and lengths.is_contiguous()
        prefix = torch.arange(indices.shape[-1], device=indices.device)[None, :] < lengths[:, None]
        assert torch.equal(indices >= 0, prefix)
        if valid is not None:
            assert torch.equal(lengths == 0, ~valid)
        result = attention_forward(*args, **kwargs)
        kernel_outputs.append(result[0].data_ptr())
        compact_inputs[result[0].data_ptr()] = (indices, lengths, indices.clone(), lengths.clone())
        return result

    monkeypatch.setattr(
        csa2_indexer.fused_sparse_attention, "_csa_fwd_flash_mla", record_attention_forward
    )
    csa2_indexer.fused_sparse_attention._ensure_dsa_namespace()
    dsa = csa2_indexer.fused_sparse_attention._DSA
    attention_backward = dsa.sparse_attention_backward_wrapper

    def record_attention_backward(q, kv, out, grad_out, lse, sink, indices, **kwargs):
        original, lengths, saved_indices, saved_lengths = compact_inputs[out.data_ptr()]
        assert torch.equal(original, saved_indices)
        assert torch.equal(lengths, saved_lengths)
        assert torch.equal(indices, saved_indices.clamp_min(0))
        assert torch.equal(kwargs["topk_length"], saved_lengths.clamp_min(1))
        empty_rows = saved_lengths == 0
        assert torch.count_nonzero(grad_out[empty_rows]) == 0
        assert torch.count_nonzero(lse[empty_rows]) == 0
        backward_calls.append(1)
        return attention_backward(q, kv, out, grad_out, lse, sink, indices, **kwargs)

    monkeypatch.setattr(dsa, "sparse_attention_backward_wrapper", record_attention_backward)

    def record_window_lse(*args, **kwargs):
        window_lse_calls.append(1)
        return window_lse(*args, **kwargs)

    monkeypatch.setattr(csa2_indexer, "fused_csa_window_lse", record_window_lse)
    for layer_idx, core in enumerate(cores):
        if layout != "sbhd" and core.compressor is not None:

            def record_compressor_input(module, args, kwargs, layer=layer_idx):
                assert kwargs["input_is_sanitized"]
                compressor_inputs[layer] = args[0]

            def record_compressor_projection(module, args, layer=layer_idx):
                hidden = compressor_inputs[layer]
                assert (
                    args[0].data_ptr() == hidden.data_ptr()
                ), "Linear must reuse token-order hidden"
                assert args[0].stride() == hidden.stride()
                compressor_projections.append(layer)

            core.compressor.register_forward_pre_hook(record_compressor_input, with_kwargs=True)
            core.compressor.linear_wkv.register_forward_pre_hook(record_compressor_projection)
            if core.compressor.linear_wgate is not None:
                core.compressor.linear_wgate.register_forward_pre_hook(record_compressor_projection)
        original = core._fused_indices

        def record_indices(q, window, topk, state, fn=original, layer=layer_idx, **kwargs):
            indices = fn(q, window, topk, state, **kwargs)
            shared_buffers[layer] = (
                indices,
                state.global_kv_flat,
                state.indexer_k_flat,
                state.fused_topk_length,
                state.fused_q_padding_mask,
                state.fused_window_indices,
            )
            return indices

        monkeypatch.setattr(core, "_fused_indices", record_indices)
        build_indices = core._build_fused_thd_indices

        def record_build_indices(*args, fn=build_indices, layer=layer_idx, **kwargs):
            final_index_calls.append(layer)
            return fn(*args, **kwargs)

        monkeypatch.setattr(core, "_build_fused_thd_indices", record_build_indices)
        if core.indexer is not None:
            select_name = (
                "_fused_topk_and_candidates" if core.is_candidate_source else "_fused_topk"
            )
            select = getattr(core.indexer, select_name)

            def record_selection(prepared, fn=select, layer=layer_idx):
                assert not torch.is_grad_enabled()
                prepared_inputs[layer] = prepared
                before = [t.clone() for t in (prepared.q, prepared.k, prepared.weights)]
                result = fn(prepared)
                for value, saved in zip((prepared.q, prepared.k, prepared.weights), before):
                    torch.testing.assert_close(value, saved, atol=0, rtol=0)
                return result

            monkeypatch.setattr(core.indexer, select_name, record_selection)
    for core in cores:
        if core.indexer is not None:
            core.indexer.linear_wq_b.register_forward_hook(lambda *args: calls.append(1))
            monkeypatch.setattr(
                core.indexer,
                "_score_projected",
                lambda *a, **kw: pytest.fail("fused stack used native dense scores"),
            )
    run = _run_sbhd_cores if layout == "sbhd" else lambda c, x: _run_packed_cores(c, x, params)
    with monkeypatch.context() as fused_path:
        if layout != "sbhd":
            module = "megatron.core.transformer.experimental_attention_variant.csa2"

            def no_unfused_indices(*args, **kwargs):
                pytest.fail("Fused THD must build final indices without a separate window/compact")

            fused_path.setattr(f"{module}.get_window_topk_idxs_thd", no_unfused_indices)
            fused_path.setattr(f"{module}._compact_flat_topk_idxs", no_unfused_indices)
            fused_path.setattr(
                f"{module}.maybe_prepare_csa2_r2_fused",
                lambda *a, **kw: pytest.fail("BF16 THD must not gather hidden before projection"),
            )
            fused_path.setattr(
                csa2_indexer.fused_sparse_attention, "_compact_flat_topk_idxs", no_unfused_indices
            )
        outputs, state = run(cores, inputs)
    expected, ref_state = run(reference, ref_inputs)
    assert len(calls) == 3, "Full/Reindex must project Q once; Reuse must not run the indexer"
    assert len(prepared_inputs) == len(loss_inputs_seen) == 3
    for layer, prepared in prepared_inputs.items():
        assert prepared.k is shared_buffers[layer][2], "Selection must use the Full owner's K"
    assert prepared_inputs[3].k is prepared_inputs[4].k
    assert len(window_lse_calls) == (0 if sparse else 3)
    assert shared_buffers[1][0] is shared_buffers[2][0]
    assert shared_buffers[4][0] is shared_buffers[5][0]
    assert shared_buffers[1][3] is shared_buffers[2][3]
    assert shared_buffers[4][3] is shared_buffers[5][3]
    assert shared_buffers[3][0] is not shared_buffers[4][0], "Reindex must refresh Top-K"
    assert shared_buffers[3][3] is not shared_buffers[4][3], "Reindex must refresh lengths"
    assert shared_buffers[3][1] is shared_buffers[4][1] is shared_buffers[5][1]
    if layout != "sbhd":
        assert compressor_projections == [1, 1, 3]
        assert final_index_calls == [0, 1, 3, 4], "Reuse must not rebuild final THD indices"
        assert shared_buffers[1][4] is shared_buffers[2][4]
        assert shared_buffers[4][4] is shared_buffers[5][4]
        alignment = csa2_indexer.fused_sparse_attention.get_flash_mla_topk_alignment()
        for buffers in shared_buffers.values():
            assert buffers[0].shape[-1] % alignment == 0
            assert torch.equal(buffers[4], ~valid)
        windows = [shared_buffers[layer][5] for layer in (1, 2, 3, 4, 5)]
        if sparse:
            assert all(window is None for window in windows)
        else:
            assert all(window is windows[0] for window in windows)
            assert windows[0] is not None
    if layout == "sbhd":
        assert [
            out.data_ptr() for out in outputs
        ] == kernel_outputs, (
            "Restoring SBHD must not retain another full attention output allocation"
        )
    torch.testing.assert_close(state.global_indices, ref_state.global_indices)
    if candidates:
        torch.testing.assert_close(state.candidates.indices, ref_state.candidates.indices)
        torch.testing.assert_close(state.candidates.lengths, ref_state.candidates.lengths)
    loss = ref_loss = 0
    for out, ref in zip(outputs, expected):
        _assert_fused_close(out, ref, exact=False, name="indexer stack output")
        probe = torch.randn_like(out) * 0.01
        loss, ref_loss = loss + (out * probe).sum(), ref_loss + (ref * probe).sum()
    loss.backward()
    ref_loss.backward()
    assert len(backward_calls) == 6
    for indices, lengths, saved_indices, saved_lengths in compact_inputs.values():
        assert torch.equal(indices, saved_indices)
        assert torch.equal(lengths, saved_lengths)
    for row, ref_row in zip(inputs, ref_inputs):
        for actual, ref in zip(row, ref_row):
            _assert_fused_close(actual.grad, ref.grad, exact=False, name="indexer stack input")
            if valid is not None and actual.grad is not None:
                assert torch.count_nonzero(actual.grad[~valid]) == 0
    for (name, p), (_, ref) in zip(cores.named_parameters(), reference.named_parameters()):
        _assert_fused_close(p.grad, ref.grad, exact=False, name=name)
    if layout != "padding":
        for owner in (1, 3):
            assert cores[owner].indexer.linear_wk.weight.grad.abs().sum() > 0


# Compact candidate generation. Keep the oracle independent of fused scoring,
# block reduction, radix selection, and packed address conversion.


def _reference_block_ids(scores, visible, topk_blocks, block_size):
    width = scores.shape[-1]
    blocks = (width + block_size - 1) // block_size
    output = torch.full(
        (*scores.shape[:-1], min(topk_blocks, blocks)), -1, device=scores.device, dtype=torch.int32
    )
    counts = torch.zeros(scores.shape[:-1], device=scores.device, dtype=torch.int32)
    if width == 0:
        return CSA2CandidateBlocks(output, counts, block_size)
    cols = torch.arange(width, device=scores.device)
    masked = scores.masked_fill(cols >= visible.unsqueeze(-1), -torch.inf)
    maxima = torch.stack(
        [masked[..., start : start + block_size].amax(-1) for start in range(0, width, block_size)],
        -1,
    )
    newest = (visible - 1) // block_size
    for block in range(blocks):
        maxima[..., block] = torch.where(newest == block, torch.inf, maxima[..., block])
    ordered = maxima.argsort(dim=-1, descending=True, stable=True)[..., :topk_blocks]
    valid = maxima.gather(-1, ordered) > -torch.inf
    ordered = ordered.masked_fill(~valid, width).sort(-1).values
    return CSA2CandidateBlocks(
        ordered.masked_fill(ordered == width, -1).int(), valid.sum(-1).int(), block_size
    )


def _reference_candidate_blocks(
    indexer, q, k, weights, *, selection_scores=None, thd_layout=None, compressed_layout=None
):
    scores = _reference_fused_indexer_scores(
        indexer, q, k, weights, thd_layout=thd_layout, compressed_layout=compressed_layout
    )
    block_size = indexer.config.csa2_candidate_block_size
    topk_blocks = indexer.config.csa2_candidate_topk_blocks
    if thd_layout is None:
        visible = ((torch.arange(q.shape[0], device=q.device) + 1) // indexer.compress_ratio).clamp(
            max=k.shape[0]
        )
        return _reference_block_ids(scores, visible, topk_blocks, block_size)
    width = min(topk_blocks, (compressed_layout.max_seqlen + block_size - 1) // block_size)
    ids = torch.full((q.shape[0], width), -1, device=q.device, dtype=torch.int32)
    lengths = torch.zeros(q.shape[0], device=q.device, dtype=torch.int32)
    for q_start, q_length, k_start, k_length in zip(
        thd_layout.cu_seqlens_padded[:-1].tolist(),
        thd_layout.cu_seqlens.diff().tolist(),
        compressed_layout.cu_seqlens_padded[:-1].tolist(),
        compressed_layout.cu_seqlens.diff().tolist(),
    ):
        local = scores[q_start : q_start + q_length, k_start : k_start + k_length]
        visible = (torch.arange(q_length, device=q.device) + 1) // indexer.compress_ratio
        selected = _reference_block_ids(local, visible, topk_blocks, block_size)
        ids[q_start : q_start + q_length, : selected.indices.shape[-1]] = selected.indices
        lengths[q_start : q_start + q_length] = selected.lengths
    return CSA2CandidateBlocks(ids, lengths, block_size)


def _require_candidate_kernels(precision="bf16"):
    major = 10 if precision == "mxfp8" else 9
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < major:
        pytest.skip(f"Fused {precision} candidates require SM{major}0+")
    pytest.importorskip("triton")
    pytest.importorskip("cudnn")
    if precision == "mxfp8":
        pytest.importorskip("transformer_engine")
    csa2_candidates.fused_sparse_attention._ensure_dsa_namespace()


@pytest.mark.parametrize("backend", ["native", "real"])
@pytest.mark.parametrize(
    "block_size,topk,width", [(8, 3, 59), (3, 1, 37), (2, 8, 13), (8, 2048, 32771)]
)
def test_candidate_block_compaction_ties_newest_and_padding(backend, block_size, topk, width):
    if backend == "real":
        _require_candidate_kernels()
    device = "cuda" if backend == "real" else "cpu"
    torch.manual_seed(414)
    # Exact ties across the radix threshold, including negative weighted scores.
    scores = torch.randint(-3, 4, (5, width), device=device).float()
    visible = torch.tensor([0, 1, width // 2, width - 1, width], device=device, dtype=torch.int32)
    scores[4] = 0
    # Invisible high values must never enter the reduction. The latest block
    # has the worst ordinary score but must still be retained.
    scores[2, width // 2 - 1] = -100
    expected = _reference_block_ids(scores, visible, topk, block_size)
    if backend == "real":
        actual = csa2_candidates._select_blocks(scores, visible, topk, block_size)
    else:
        masked = scores.masked_fill(torch.arange(width)[None, :] >= visible[:, None], -torch.inf)
        actual = candidate_blocks_from_scores(masked, visible, topk, block_size)
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.lengths, expected.lengths)
    assert (actual.indices[0] == -1).all()
    assert (actual.indices[1:, :] == ((visible[1:] - 1) // block_size)[:, None]).any(-1).all()
    if backend == "real":
        again = csa2_candidates._select_blocks(
            F.pad(scores, (0, 19), value=1e10), visible, topk, block_size
        )
        torch.testing.assert_close(again.lengths, actual.lengths)
        torch.testing.assert_close(again.indices[:, : actual.indices.shape[-1]], actual.indices)
        assert (again.indices[:, actual.indices.shape[-1] :] == -1).all()


@pytest.mark.parametrize("shape", [(2, 0, 0), (2, 3, 0), (0, 7), (3, 0)])
def test_candidate_blocks_empty_native_storage(shape):
    scores = torch.empty(shape)
    selected = candidate_blocks_from_scores(
        scores, torch.zeros(shape[:-1], dtype=torch.int32), 3, 8
    )
    assert selected.indices.shape == (*shape[:-1], (shape[-1] + 7) // 8)
    assert torch.count_nonzero(selected.lengths) == 0
    assert selected.to_mask(shape[-1]).shape == shape


def test_fused_candidate_radix_rows_with_unaligned_block_counts():
    """Unaligned radix rows previously duplicated +inf and dropped a real winner."""
    _require_candidate_kernels()
    scores = torch.tensor(
        [[0.81289214, 0.430245, 0.707756, 1.1742834, -1.0] + [100.0] * 12] * 17, device="cuda"
    )
    visible = torch.full((17,), 5, dtype=torch.int32, device="cuda")
    actual = csa2_candidates._select_blocks(scores, visible, 3, 1)
    torch.testing.assert_close(
        actual.indices, torch.tensor([[0, 3, 4]] * 17, dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(actual.lengths, torch.full_like(visible, 3))


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("layout", ["sbhd", "thd", "long-tail"])
def test_fused_candidate_generation_matches_dense_oracle(
    monkeypatch, precision, heads, ratio, layout
):
    _require_candidate_kernels(precision)
    torch.manual_seed(415)
    indexer = _kernel_indexer(ratio, precision, heads, "cuda")
    indexer.config.csa2_candidate_topk_blocks = 3
    indexer.config.csa2_candidate_block_size = 8
    token_layout = compressed = None
    if layout == "sbhd":
        q = torch.randn(131, 2, heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(131 // ratio, 2, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(131, 2, heads, device="cuda", dtype=torch.bfloat16)
    else:
        params, _, valid = _packed(
            [1, 0, 43, 131],
            [3, 0, 48, 134],
            tail=511 if layout == "long-tail" else 7,
            device="cuda",
        )
        token_layout = build_csa2_thd_layout(params, valid.numel())
        compressed = token_layout.for_compression(ratio)
        q = torch.randn(valid.numel(), heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(compressed.capacity, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(valid.numel(), heads, device="cuda", dtype=torch.bfloat16)
        # Padding contains adversarial values, not a convenient all-zero input.
        q[~valid], w[~valid], k[~compressed.valid_groups] = 100, 100, 100
    snapshots = [x.clone() for x in (q, k, w)]
    expected = _reference_candidate_blocks(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    # Force chunks across sequences and compression/block boundaries. Bound the
    # score allocation by bytes as well as rows; neither depends on total Q.
    monkeypatch.setattr(csa2_candidates, "_QUERY_CHUNK_SIZE", 31)
    max_k = k.shape[0] if token_layout is None else compressed.max_seqlen
    monkeypatch.setattr(csa2_candidates, "_SCORE_CHUNK_MAX_BYTES", 17 * ((max_k + 3) // 4 * 4) * 4)
    scorer = csa2_candidates.fused_sparse_attention._DSA.indexer_forward_wrapper
    chunks = []

    def bounded_scorer(chunk_q, all_k, chunk_w, **kwargs):
        assert chunk_q.shape[0] <= 17
        if precision == "bf16":
            assert all_k.data_ptr() == prepared.k.data_ptr()
            offset = sum(chunks) * heads * 128 * prepared.q.element_size()
            assert chunk_q.data_ptr() == prepared.q.data_ptr() + offset
        assert kwargs["cu_seqlens_q"][-1] == chunk_q.shape[0]
        assert kwargs["cu_seqlens_k"][-1] == all_k.shape[0]
        assert kwargs["cu_seqlens_q"].diff().max() <= kwargs["max_seqlen_q"]
        assert kwargs["cu_seqlens_k"].diff().max() <= kwargs["max_seqlen_k"]
        chunks.append(chunk_q.shape[0])
        return scorer(chunk_q, all_k, chunk_w, **kwargs)

    monkeypatch.setattr(
        csa2_candidates.fused_sparse_attention._DSA, "indexer_forward_wrapper", bounded_scorer
    )
    expected_topk = _reference_fused_indexer(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    prepared = prepare_csa2_indexer_inputs(
        q, k, w, ratio, thd_layout=token_layout, compressed_layout=compressed
    )
    prepared_before = [t.clone() for t in (prepared.q, prepared.k, prepared.weights)]
    actual_topk, actual = indexer._fused_topk_and_candidates(prepared)
    torch.testing.assert_close(actual_topk, expected_topk)
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.lengths, expected.lengths)
    assert len(chunks) > 1 and sum(chunks) == q.shape[0] * (2 if layout == "sbhd" else 1)
    for x, before in zip((q, k, w), snapshots):
        torch.testing.assert_close(x, before, atol=0, rtol=0)
    for x, before in zip((prepared.q, prepared.k, prepared.weights), prepared_before):
        torch.testing.assert_close(x, before, atol=0, rtol=0)
    assert actual.indices.is_contiguous() and not actual.indices.requires_grad
    mask = actual.to_mask(k.shape[0], thd_layout=token_layout, compressed_layout=compressed)
    if token_layout is not None:
        assert not mask[~token_layout.valid_tokens].any()
        assert not mask[:, ~compressed.valid_groups].any()
        assert not mask[
            token_layout.sequence_ids[:, None] != compressed.sequence_ids[None, :]
        ].any()
    if precision == "mxfp8" and layout == "sbhd":
        indexer.precision = "bf16"
        unquantized = _reference_candidate_blocks(indexer, q, k, w)
        assert not torch.equal(
            actual.indices, unquantized.indices
        ), "MXFP8 must change candidate ranking"


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("layout", ["empty", "padding", "short", "unused-capacity"])
def test_fused_candidate_generation_empty_and_unused_capacity(precision, ratio, layout):
    _require_candidate_kernels(precision)
    real, physical, tail = {
        "empty": ([0, 0], [0, 0], 0),
        "padding": ([0, 0], [3, 5], 3),
        "short": ([1, 1], [1, 1], 0),
        "unused-capacity": ([1, 1], [1, 1], 7),
    }[layout]
    params, _, valid = _packed(real, physical, tail=tail, device="cuda")
    if layout == "unused-capacity":
        params.max_seqlen_q = params.max_seqlen_kv = 4
    token_layout = build_csa2_thd_layout(params, valid.numel())
    compressed = token_layout.for_compression(ratio)
    indexer = _kernel_indexer(ratio, precision, 32, "cuda")
    indexer.config.csa2_candidate_topk_blocks = 2
    indexer.config.csa2_candidate_block_size = 2
    q = torch.randn(valid.numel(), 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(compressed.capacity, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(valid.numel(), 32, device="cuda", dtype=torch.bfloat16)
    expected = _reference_candidate_blocks(
        indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    actual = indexer._candidate_blocks(
        q, k, w, selection_scores=None, thd_layout=token_layout, compressed_layout=compressed
    )
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.lengths, expected.lengths)
    expected_topk = _reference_fused_indexer(
        indexer, q, k, w, candidates=actual, thd_layout=token_layout, compressed_layout=compressed
    )
    topk = indexer._fused_topk(
        prepare_csa2_indexer_inputs(q, k, w, ratio, actual, token_layout, compressed)
    )
    torch.testing.assert_close(topk, expected_topk)
    q.requires_grad_(), k.requires_grad_(), w.requires_grad_()
    loss = fused_csa2_indexer_loss(
        prepare_csa2_indexer_inputs(q, k, w, ratio, actual, token_layout, compressed),
        torch.zeros((q.shape[0], 64, 512), dtype=q.dtype, device=q.device),
        torch.zeros((q.shape[0], 512), dtype=q.dtype, device=q.device),
        torch.zeros((k.shape[0], 1, 512), dtype=q.dtype, device=q.device),
        torch.zeros(64, device=q.device),
        torch.full((q.shape[0], 1), -1, dtype=torch.int32, device=q.device),
        topk,
        512**-0.5,
        0.3,
        False,
    )
    assert loss == 0 and loss.requires_grad
    loss.backward()
    for t in (q, k, w):
        assert t.grad is not None and torch.count_nonzero(t.grad) == 0


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_fused_candidate_source_without_loss_avoids_dense_scores(monkeypatch, precision):
    _require_candidate_kernels(precision)
    indexer = _kernel_indexer(2, precision, 32, "cuda")
    indexer.config.csa2_candidate_topk_blocks = 2
    indexer.config.csa2_candidate_block_size = 8

    def reject_native(*args, **kwargs):
        pytest.fail("Fused candidate-source selection must not materialize native scores")

    monkeypatch.setattr(indexer, "_score_projected", reject_native)
    monkeypatch.setattr(indexer, "_fused_topk", reject_native)
    monkeypatch.setattr(indexer, "_candidate_blocks", reject_native)
    indices, scores, candidates = indexer.forward_with_scores(
        torch.randn(65, 2, 6, dtype=torch.bfloat16, device="cuda"),
        torch.randn(65, 2, 4, dtype=torch.bfloat16, device="cuda"),
        torch.randn(32, 2, 8, dtype=torch.bfloat16, device="cuda"),
        _CPUFrequencyTable(4).cuda(),
        return_candidates=True,
    )
    assert scores is None and indices.shape == (2, 65, 7)
    assert candidates.indices.shape == (2, 65, 2)


def test_fused_candidate_generation_long_top2048_memory():
    """Exercise the real scorer/Top2048 together without a quadratic score oracle."""
    _require_candidate_kernels()
    seq_len, topk, block_size = 16393, 2048, 8
    indexer = _kernel_indexer(1, "bf16", 32, "cuda")
    indexer.config.csa2_candidate_topk_blocks = topk
    indexer.config.csa2_candidate_block_size = block_size
    # All scores tie at zero: exact winners are earlier blocks plus the latest.
    q = torch.zeros(seq_len, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(seq_len, 1, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(seq_len, 1, 32, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = indexer._candidate_blocks(q, k, w, selection_scores=None)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    # Output storage is linear in Q * topk_blocks. One full FP32 score matrix
    # alone would exceed 1 GiB here, well above this generous workspace bound.
    dense_score_bytes = seq_len * seq_len * 4
    assert peak < dense_score_bytes // 2
    print(
        f"candidate_top2048: incremental_peak_bytes={peak}, dense_score_bytes={dense_score_bytes}"
    )
    newest = torch.arange(seq_len, device="cuda", dtype=torch.int32) // block_size
    counts = (newest + 1).clamp(max=topk)
    expected = torch.arange(topk, device="cuda", dtype=torch.int32).expand(seq_len, -1).clone()
    expected.masked_fill_(expected >= counts[:, None], -1)
    expected[:, -1] = torch.where(newest >= topk - 1, newest, -1)
    torch.testing.assert_close(actual.indices[0], expected)
    torch.testing.assert_close(actual.lengths[0], counts)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    selected = indexer._fused_topk(prepare_csa2_indexer_inputs(q, k, w, 1, actual))
    torch.cuda.synchronize()
    reindex_peak = torch.cuda.max_memory_allocated() - before
    assert reindex_peak < dense_score_bytes // 2
    # Zero scores select the earliest causal keys, independently of latest-block retention.
    expected_topk = torch.arange(indexer.topk, device="cuda", dtype=torch.int32)[None].expand(
        seq_len, -1
    )
    expected_topk = expected_topk.masked_fill(
        expected_topk > torch.arange(seq_len, device="cuda")[:, None], -1
    )
    torch.testing.assert_close(selected[0], expected_topk)
    print(
        f"reindex_top2048: incremental_peak_bytes={reindex_peak}, dense_score_bytes={dense_score_bytes}"
    )


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_fused_candidate_reindex_matches_dense_oracle(monkeypatch, precision, heads, ratio, layout):
    _require_candidate_kernels(precision)
    torch.manual_seed(418)
    indexer = _kernel_indexer(ratio, precision, heads, "cuda")
    indexer.config.csa2_candidate_topk_blocks, indexer.config.csa2_candidate_block_size = 3, 8
    token_layout = compressed = None
    if layout == "sbhd":
        q = torch.randn(131, 2, heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(131 // ratio, 2, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(131, 2, heads, device="cuda", dtype=torch.bfloat16)
    else:
        params, _, valid = _packed([1, 0, 43, 131], [3, 0, 48, 134], tail=7, device="cuda")
        token_layout = build_csa2_thd_layout(params, valid.numel())
        compressed = token_layout.for_compression(ratio)
        q = torch.randn(valid.numel(), heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(compressed.capacity, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(valid.numel(), heads, device="cuda", dtype=torch.bfloat16)
        q[~valid], w[~valid], k[~compressed.valid_groups] = 100, 100, 100
    candidates = _reference_candidate_blocks(
        indexer, -q, k, w, thd_layout=token_layout, compressed_layout=compressed
    )
    kwargs = dict(candidates=candidates, thd_layout=token_layout, compressed_layout=compressed)
    expected = _reference_fused_indexer(indexer, q, k, w, **kwargs)
    tied_q = torch.zeros_like(q)
    tied_expected = _reference_fused_indexer(indexer, tied_q, k, w, **kwargs)
    before = [t.clone() for t in (q, k, w, candidates.indices, candidates.lengths)]
    monkeypatch.setattr(csa2_indexer, "_QUERY_CHUNK_SIZE", 17)
    scorer, chunks = csa2_indexer._score_chunk, []

    def bounded_score(*args, **kw):
        assert args[0].shape[0] <= 17 and kw["stride"] == 24
        assert (args[0].dtype == torch.float8_e4m3fn) == (precision == "mxfp8")
        chunks.append(args[0].shape[0])
        return scorer(*args, **kw)

    monkeypatch.setattr(csa2_indexer, "_score_chunk", bounded_score)
    if precision == "mxfp8" and torch.cuda.get_device_capability() == (10, 0):
        run = csa2_indexer._score_kernel.run

        def hardware_scaled_mma(*args, **kw):
            compiled = run(*args, **kw)
            assert any(
                "tcgen05.mma" in line and "mxf8f6f4" in line
                for line in compiled.asm["ptx"].splitlines()
            )
            return compiled

        monkeypatch.setattr(csa2_indexer._score_kernel, "run", hardware_scaled_mma)
    monkeypatch.setattr(
        CSA2CandidateBlocks, "to_mask", lambda *a, **kw: pytest.fail("dense candidate mask")
    )
    actual = indexer._fused_topk(prepare_csa2_indexer_inputs(q, k, w, ratio, **kwargs))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        indexer._fused_topk(prepare_csa2_indexer_inputs(tied_q, k, w, ratio, **kwargs)),
        tied_expected,
        atol=0,
        rtol=0,
    )
    assert len(chunks) > 1
    for tensor, old in zip((q, k, w, candidates.indices, candidates.lengths), before):
        torch.testing.assert_close(tensor, old, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["dense", "candidates", "sparse"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_fused_indexer_loss_matches_independent_teacher(
    monkeypatch, mode, ratio, per_token, layout
):
    _require_candidate_kernels()
    torch.manual_seed(419)
    indexer = _kernel_indexer(ratio, "bf16", 32, "cuda")
    indexer.config.csa2_candidate_topk_blocks, indexer.config.csa2_candidate_block_size = 2, 3
    token_layout = compressed = None
    if layout == "sbhd":
        q = torch.randn(37, 2, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(37 // ratio, 2, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(37, 2, 32, device="cuda", dtype=torch.bfloat16)
        teacher_q = torch.randn(37, 2, 64, 512, device="cuda", dtype=torch.bfloat16)
        teacher_k = torch.randn(37 // ratio, 2, 512, device="cuda", dtype=torch.bfloat16)
        local = torch.randn(37, 2, 512, device="cuda", dtype=torch.bfloat16)
        window = (
            torch.arange(37, device="cuda")[:, None] - torch.arange(4, device="cuda")[None, :]
        ).clamp_min(-1)
        window = window[None].expand(2, -1, -1).contiguous().int()
    else:
        params, _, valid = _packed([1, 0, 11, 37], [3, 0, 14, 40], tail=5, device="cuda")
        token_layout = build_csa2_thd_layout(params, valid.numel())
        compressed = token_layout.for_compression(ratio)
        q = torch.randn(valid.numel(), 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(compressed.capacity, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(valid.numel(), 32, device="cuda", dtype=torch.bfloat16)
        teacher_q = torch.randn(valid.numel(), 64, 512, device="cuda", dtype=torch.bfloat16)
        teacher_k = torch.randn(compressed.capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
        local = torch.randn(valid.numel(), 512, device="cuda", dtype=torch.bfloat16)
        window = (
            torch.arange(valid.numel(), device="cuda")[:, None]
            - torch.arange(4, device="cuda")[None, :]
        )
        keep = (
            valid[:, None]
            & (window >= 0)
            & (torch.arange(4, device="cuda")[None, :] <= token_layout.position_ids[:, None])
        )
        window = window.masked_fill(~keep, -1).int()
        q[~valid], w[~valid], k[~compressed.valid_groups] = 100, 100, 100
    candidates = (
        None
        if mode == "dense"
        else _reference_candidate_blocks(
            indexer, q, k, w, thd_layout=token_layout, compressed_layout=compressed
        )
    )
    q, k, w = [t.requires_grad_() for t in (q, k, w)]
    ref_q, ref_k, ref_w = [t.detach().clone().requires_grad_() for t in (q, k, w)]
    kwargs = dict(candidates=candidates, thd_layout=token_layout, compressed_layout=compressed)
    scores = indexer._score_projected(ref_q, ref_k, ref_w, **kwargs)
    topk = _sorted_valid_indices(scores.detach(), 7)
    sink = torch.linspace(-7, 9, 64, device="cuda", requires_grad=True)
    if mode == "candidates":
        # Global teacher mass far below 1e-10 must still renormalize to one.
        with torch.no_grad():
            sink.add_(60)
    teacher_q.requires_grad_(), teacher_k.requires_grad_(), local.requires_grad_()
    core = SimpleNamespace(
        attn_sink=sink,
        softmax_scale=512**-0.5,
        config=SimpleNamespace(
            dsa_indexer_use_sparse_loss=mode == "sparse",
            dsa_indexer_loss_coeff=0.3,
            calculate_per_token_loss=per_token,
        ),
    )
    if layout == "sbhd":
        expected = _teacher_oracle(core, teacher_q, local, teacher_k, window, topk, scores)
    else:
        expected = _teacher_oracle(
            core,
            teacher_q.unsqueeze(1),
            local.unsqueeze(1),
            teacher_k,
            window.unsqueeze(0),
            topk.unsqueeze(0),
            scores.unsqueeze(0),
        )
        if not per_token:
            expected = expected * q.shape[0] / valid.sum()
    monkeypatch.setattr(csa2_indexer, "_QUERY_CHUNK_SIZE", 17)
    monkeypatch.setattr(
        CSA2CandidateBlocks, "to_mask", lambda *a, **kw: pytest.fail("dense candidate mask")
    )
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t.shape) or t), lambda t: t
    ):
        actual = fused_csa2_indexer_loss(
            prepare_csa2_indexer_inputs(q, k, w, ratio, candidates, token_layout, compressed),
            teacher_q,
            local,
            teacher_k,
            sink,
            window,
            topk,
            core.softmax_scale,
            0.3,
            mode == "sparse",
            per_token,
        )
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    assert scores.shape not in saved
    assert len(saved) == 3, "Only the unit-loss dQ/dK/dW should survive forward"
    monkeypatch.setattr(
        csa2_indexer, "_score_chunk", lambda *a, **kw: pytest.fail("backward recomputed scores")
    )
    monkeypatch.setattr(
        csa2_indexer, "_target_chunk", lambda *a, **kw: pytest.fail("backward recomputed targets")
    )
    (actual * 0.7).backward()
    (expected * 0.7).backward()
    for value, ref in zip((q, k, w), (ref_q, ref_k, ref_w)):
        error = (value.grad.float() - ref.grad.float()).norm()
        assert error / ref.grad.float().norm().clamp_min(1e-10) < 0.01
        assert torch.isfinite(value.grad).all()
    assert all(t.grad is None for t in (teacher_q, teacher_k, local, sink))
    if layout == "thd":
        assert not q.grad[~valid].any() and not w.grad[~valid].any()
        assert not k.grad[~compressed.valid_groups].any()


@pytest.mark.parametrize("mode", ["dense", "candidates", "sparse"])
def test_fused_indexer_loss_spans_key_tiles_without_dense_saved_scores(mode):
    """Exercise H64 and reductions beyond 64/1024 slots, including backward accumulation."""
    _require_candidate_kernels()
    torch.manual_seed(420)
    length = 1099
    indexer = _kernel_indexer(1, "bf16", 64, "cuda")
    indexer.config.csa2_candidate_topk_blocks, indexer.config.csa2_candidate_block_size = 137, 8
    q = torch.randn(length, 1, 64, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(length, 1, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(length, 1, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    candidates = (
        None
        if mode == "dense"
        else _reference_candidate_blocks(indexer, q.detach(), k.detach(), w.detach())
    )
    rq, rk, rw = [t.detach().clone().requires_grad_() for t in (q, k, w)]
    scores = indexer._score_projected(rq, rk, rw, candidates=candidates)
    topk = _sorted_valid_indices(scores.detach(), 1027)
    teacher_q = torch.randn(length, 1, 16, 64, dtype=q.dtype, device=q.device)
    teacher_k = torch.randn(length, 1, 64, dtype=q.dtype, device=q.device)
    local = torch.randn(length, 1, 64, dtype=q.dtype, device=q.device)
    window = (
        (torch.arange(length, device=q.device)[:, None] - torch.arange(4, device=q.device)[None, :])
        .clamp_min(-1)[None]
        .int()
    )
    core = _math_core(sparse=mode == "sparse").cuda()
    core.attn_sink = nn.Parameter(torch.linspace(-7, 9, 16, device=q.device))
    core.softmax_scale = 64**-0.5
    expected = core._compute_indexer_loss(teacher_q, local, teacher_k, window, topk, scores)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t.shape) or t), lambda t: t
    ):
        actual = fused_csa2_indexer_loss(
            prepare_csa2_indexer_inputs(q, k, w, 1, candidates),
            teacher_q,
            local,
            teacher_k,
            core.attn_sink,
            window,
            topk,
            core.softmax_scale,
            0.3,
            mode == "sparse",
        )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    # Eager backward retains linear-size unit gradients instead of projections
    # and teacher activations. Account for those outputs plus bounded workspace.
    gradient_bytes = sum(t.numel() * t.element_size() for t in (q, k, w))
    assert peak < gradient_bytes + 8 * 1024 * 1024
    assert torch.Size((length, length)) not in saved and scores.shape not in saved
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    actual.backward()
    expected.backward()
    for value, ref in zip((q, k, w), (rq, rk, rw)):
        error = (value.grad.float() - ref.grad.float()).norm()
        assert error / ref.grad.float().norm().clamp_min(1e-10) < 0.01
    print(f"indexer_loss_{mode}: forward_incremental_peak_bytes={peak}")
