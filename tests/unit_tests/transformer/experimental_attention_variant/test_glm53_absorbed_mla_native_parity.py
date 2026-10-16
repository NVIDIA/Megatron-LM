# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real NoPE/KPool attention against independent released HF attention math.

Source: huggingface/transformers@93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py:
Glm5NextTextAttention, Glm5NextTextIndexer, Glm5NextTextRMSNorm and
eager_attention_forward. The reference expands K/V as HF does; it deliberately
does not copy MCore's matrix absorption or reuse its kernels/math helpers.

All hidden/head/latent dimensions are official. Sequence length is 17 and the
token budget is 8 instead of 2048 so this short test actually prunes pools.
This is single-layer BF16 coverage, not whole-checkpoint or training readiness.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_glm53_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_kernels
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.dsa_native_parity_utils import (
    assert_similarity,
)

_HIDDEN = 4096
_HEADS = 64
_Q_RANK = 1536
_KV_RANK = 512
_HEAD_DIM = 256
_INDEX_HEADS = 32
_INDEX_DIM = 128
_POOL = 4
_TOPK = 8
_EPS = 1e-5


def _config(backend):
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        kv_channels=_HEAD_DIM,
        q_lora_rank=_Q_RANK,
        kv_lora_rank=_KV_RANK,
        qk_head_dim=_HEAD_DIM,
        qk_pos_emb_head_dim=0,
        v_head_dim=_HEAD_DIM,
        qk_layernorm=True,
        normalization='RMSNorm',
        layernorm_epsilon=_EPS,
        attention_latent_norm_epsilon=_EPS,
        rmsnorm_cast_before_weight=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.unfused if backend == 'none' else AttnBackend.auto,
        dsa_kernel_backend=backend,
        experimental_attention_variant='dsa',
        dsa_indexer_n_heads=_INDEX_HEADS,
        dsa_indexer_head_dim=_INDEX_DIM,
        dsa_indexer_topk=_TOPK,
        dsa_indexer_kpool=_POOL,
        dsa_indexer_kpool_use_quantization=False,
        dsa_indexer_kpool_always_select_tail=True,
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_hf_scoring=True,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_qk_proj_use_quantization=False,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype='bf16',
        mla_proj_use_quantization=False,
        rope_type='rope',
        rotary_scaling_factor=1.0,
        mscale_all_dim=0.0,
        kda_two_stage_gates=True,
        enable_hyper_connections=True,
        mhc_norm_eps_inside_sqrt=True,
        mhc_keep_mappings_in_fp32=False,
        mhc_keep_pre_in_fp32=True,
        mhc_learned_output_contract=False,
    )


class _NativeRMSNorm(nn.Module):
    def __init__(self, width, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=torch.bfloat16))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + _EPS)
        return self.weight * normalized.to(x.dtype)


class _NativeFrozenIndexer(nn.Module):
    def __init__(self, device):
        super().__init__()
        kwargs = dict(device=device, dtype=torch.bfloat16, bias=False)
        self.wq_b = nn.Linear(_Q_RANK, _INDEX_HEADS * _INDEX_DIM, **kwargs)
        self.wk = nn.Linear(_HIDDEN, _INDEX_DIM, **kwargs)
        self.k_norm = nn.LayerNorm(_INDEX_DIM, eps=1e-6, device=device, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(_HIDDEN, _INDEX_HEADS, **kwargs)
        self.index_kpool_compress_gate = nn.Parameter(
            torch.randn(_INDEX_DIM, _HIDDEN, device=device, dtype=torch.bfloat16) * 0.01
        )
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(_POOL, _INDEX_DIM, device=device))
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, qr):
        batch, sequence, _ = x.shape
        query = self.wq_b(qr).view(batch, sequence, _INDEX_HEADS, _INDEX_DIM)
        keys = self.k_norm(self.wk(x))
        gates = F.linear(x, self.index_kpool_compress_gate)
        count = sequence // _POOL
        grouped_keys = keys[:, : count * _POOL].view(batch, count, _POOL, _INDEX_DIM)
        grouped_gates = gates[:, : count * _POOL].view_as(grouped_keys)
        probabilities = (
            (grouped_gates.float() + self.index_kpool_compress_ape.float()[None, None])
            .softmax(dim=2)
            .to(keys.dtype)
        )
        pooled = (probabilities * grouped_keys).sum(dim=2)
        scores = torch.matmul(query.float(), pooled.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * (_INDEX_DIM**-0.5))
        weights = self.weights_proj(x).float() * (_INDEX_HEADS**-0.5)
        scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        pool_tokens = torch.arange(count * _POOL, device=x.device).view(count, _POOL)
        valid = pool_tokens[:, -1][None, :] <= torch.arange(sequence, device=x.device)[:, None]
        selected = (
            scores.masked_fill(~valid[None], torch.finfo(scores.dtype).min)
            .topk(min(_TOPK // _POOL, count), dim=-1)
            .indices
        )
        selected_valid = valid[None].expand(batch, -1, -1).gather(-1, selected)
        tokens = pool_tokens[selected].masked_fill(~selected_valid[..., None], -1).flatten(-2)
        output = torch.full(
            (batch, sequence, _TOPK + _POOL - 1), -1, device=x.device, dtype=torch.int32
        )
        output[..., : tokens.size(-1)] = tokens.to(torch.int32)
        for query_index in range(sequence):
            tail_count = (query_index + 1) % _POOL
            output[:, query_index, _TOPK : _TOPK + tail_count] = torch.arange(
                query_index + 1 - tail_count, query_index + 1, device=x.device, dtype=torch.int32
            )
        return output

    @torch.no_grad()
    def forward_padded(self, x, qr, key_valid):
        """HF selection for a ``[batch, sequence]`` padding vector.

        Pools start at each sample's first valid key, a pool is a candidate only when
        every slot is valid and its final token is visible, and the visible tail starts
        at ``first_key + visible - visible % pool``. Padded query rows select nothing.
        """
        batch, sequence, _ = x.shape
        device = x.device
        query = self.wq_b(qr).view(batch, sequence, _INDEX_HEADS, _INDEX_DIM)
        keys = self.k_norm(self.wk(x))
        gates = F.linear(x, self.index_kpool_compress_gate)
        positions = torch.arange(sequence, device=device)
        visible = (positions[None, :] <= positions[:, None])[None] & key_valid[:, None, :]

        count = (sequence + _POOL - 1) // _POOL
        first_key = torch.where(
            key_valid.any(-1),
            key_valid.long().argmax(-1),
            torch.full((batch,), sequence, dtype=torch.long, device=device),
        )
        pool_tokens = first_key[:, None, None] + torch.arange(count * _POOL, device=device).view(
            1, count, _POOL
        )
        batch_index = torch.arange(batch, device=device)[:, None, None]
        safe_tokens = pool_tokens.clamp(0, sequence - 1)
        slot_valid = key_valid[batch_index, safe_tokens] & (pool_tokens < sequence)
        pool_valid = slot_valid.all(-1)
        pool_tokens = pool_tokens.masked_fill(~slot_valid, -1)
        logits = gates[batch_index, safe_tokens].float()
        logits = logits + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~slot_valid[..., None], float('-inf'))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(keys.dtype)
        pooled = (probabilities * keys[batch_index, safe_tokens]).sum(dim=2)
        scores = torch.matmul(query.float(), pooled.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * (_INDEX_DIM**-0.5))
        weights = self.weights_proj(x).float() * (_INDEX_HEADS**-0.5)
        scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        pool_end = pool_tokens[..., -1].clamp(0, sequence - 1)
        valid = visible.gather(-1, pool_end[:, None, :].expand(batch, sequence, -1))
        valid = valid & pool_valid[:, None]
        selected = (
            scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            .topk(min(_TOPK // _POOL, count), dim=-1)
            .indices
        )
        selected_valid = valid.gather(-1, selected)
        tokens = pool_tokens[batch_index, selected]
        tokens = tokens.masked_fill(~selected_valid[..., None], -1).flatten(-2)
        output = torch.full(
            (batch, sequence, _TOPK + _POOL - 1), -1, device=device, dtype=torch.int32
        )
        output[..., : tokens.size(-1)] = tokens.to(torch.int32)
        visible_count = visible.long().sum(-1)
        tail_count = visible_count.remainder(_POOL)
        offsets = torch.arange(_POOL - 1, device=device)
        tail = (first_key[:, None] + visible_count - tail_count)[..., None] + offsets
        tail_valid = (offsets < tail_count[..., None]) & (tail < sequence)
        tail_valid = tail_valid & visible.gather(-1, tail.clamp(0, sequence - 1))
        output[..., _TOPK:] = tail.masked_fill(~tail_valid, -1).to(torch.int32)
        return output.masked_fill(~key_valid[..., None], -1)


class _NativeHFNoPEAttention(nn.Module):
    def __init__(self, device):
        super().__init__()
        kwargs = dict(device=device, dtype=torch.bfloat16, bias=False)
        self.q_a_proj = nn.Linear(_HIDDEN, _Q_RANK, **kwargs)
        self.q_a_layernorm = _NativeRMSNorm(_Q_RANK, device)
        self.q_b_proj = nn.Linear(_Q_RANK, _HEADS * _HEAD_DIM, **kwargs)
        self.kv_a_proj_with_mqa = nn.Linear(_HIDDEN, _KV_RANK, **kwargs)
        self.kv_a_layernorm = _NativeRMSNorm(_KV_RANK, device)
        self.kv_b_proj = nn.Linear(_KV_RANK, _HEADS * 2 * _HEAD_DIM, **kwargs)
        self.o_proj = nn.Linear(_HEADS * _HEAD_DIM, _HIDDEN, **kwargs)
        self.indexer = _NativeFrozenIndexer(device)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(std=0.02)

    def forward(self, hidden, key_valid=None):
        x = hidden.transpose(0, 1)
        batch, sequence, _ = x.shape
        qr = self.q_a_layernorm(self.q_a_proj(x))
        query = self.q_b_proj(qr).view(batch, sequence, _HEADS, _HEAD_DIM).transpose(1, 2)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        kv = self.kv_b_proj(latent).view(batch, sequence, _HEADS, 2 * _HEAD_DIM).transpose(1, 2)
        keys, values = kv.split(_HEAD_DIM, dim=-1)
        if key_valid is None:
            selected = self.indexer(x, qr)
        else:
            selected = self.indexer.forward_padded(x, qr, key_valid)
        visible = torch.zeros(batch, sequence, sequence, device=x.device, dtype=torch.int32)
        visible.scatter_add_(-1, selected.clamp_min(0).long(), (selected >= 0).to(torch.int32))
        mask = torch.where(visible[:, None].bool(), 0.0, torch.finfo(query.dtype).min).to(
            query.dtype
        )
        # Preserve HF eager attention's BF16 QK GEMM and its BF16 score boundary.
        scores = torch.matmul(query, keys.transpose(-1, -2)) * (_HEAD_DIM**-0.5)
        probabilities = (scores + mask).softmax(dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, values).transpose(1, 2).contiguous()
        return self.o_proj(output.flatten(-2)).transpose(0, 1).contiguous(), selected


_PARAMETER_MAPPING = {
    'q_a_proj.weight': 'linear_q_down_proj.weight',
    'q_a_layernorm.weight': 'q_layernorm.weight',
    'q_b_proj.weight': 'linear_q_up_proj.weight',
    'kv_a_proj_with_mqa.weight': 'linear_kv_down_proj.weight',
    'kv_a_layernorm.weight': 'kv_layernorm.weight',
    'kv_b_proj.weight': 'linear_kv_up_proj.weight',
    'o_proj.weight': 'linear_proj.weight',
    'indexer.wq_b.weight': 'core_attention.indexer.linear_wq_b.weight',
    'indexer.wk.weight': 'core_attention.indexer.linear_wk.weight',
    'indexer.k_norm.weight': 'core_attention.indexer.k_norm.weight',
    'indexer.k_norm.bias': 'core_attention.indexer.k_norm.bias',
    'indexer.weights_proj.weight': 'core_attention.indexer.linear_weights_proj.weight',
    'indexer.index_kpool_compress_gate': 'core_attention.indexer.index_kpool_compress_gate',
    'indexer.index_kpool_compress_ape': 'core_attention.indexer.index_kpool_compress_ape',
}


@pytest.fixture(autouse=True)
def _model_parallel():
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(5353)
    torch.manual_seed(5353)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


def _run_real_layer(backend, monkeypatch, *, compare_native):
    config = _config(backend)
    spec = hybrid_glm53_stack_spec(config).submodules.dsa_layer.submodules.self_attention
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    real = build_module(spec, config=config, layer_number=1, pg_collection=groups).cuda()
    assert isinstance(real, AbsorbedMLASelfAttention)
    assert not real.use_rope
    selector = real.core_attention.indexer
    assert selector.index_kpool == _POOL
    assert all(not parameter.requires_grad for parameter in selector.parameters())
    native = _NativeHFNoPEAttention(torch.device('cuda')) if compare_native else None
    real_parameters = dict(real.named_parameters())
    if native is not None:
        native_parameters = dict(native.named_parameters())
        assert set(native_parameters) == set(_PARAMETER_MAPPING)
        assert set(real_parameters) == set(_PARAMETER_MAPPING.values())
        with torch.no_grad():
            for source, target in _PARAMETER_MAPPING.items():
                assert real_parameters[target].shape == native_parameters[source].shape
                real_parameters[target].copy_(native_parameters[source])

    selected_rows, fused_results = [], []
    sparse_attention = dsa_module._run_sparse_attention
    fused_attention = dsa_kernels.run_fused_absorbed_sparse_attention

    def capture_selection(**kwargs):
        selected_rows.append(kwargs['topk_indices'].detach().clone())
        return sparse_attention(**kwargs)

    def capture_fused_result(*args, **kwargs):
        result = fused_attention(*args, **kwargs)
        fused_results.append(result is not None)
        return result

    monkeypatch.setattr(dsa_module, '_run_sparse_attention', capture_selection)
    monkeypatch.setattr(dsa_kernels, 'run_fused_absorbed_sparse_attention', capture_fused_result)
    hidden = torch.randn(17, 1, _HIDDEN, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    actual, bias = real(hidden, attention_mask=None)
    assert bias is None
    assert actual.shape == hidden.shape and actual.dtype == hidden.dtype
    assert torch.isfinite(actual).all()
    assert len(selected_rows) == 1
    assert selected_rows[0].shape == (1, 17, _TOPK + _POOL - 1)
    assert (selected_rows[0][0, -1] >= 0).sum().item() == 9
    if backend == 'cudnn':
        assert fused_results == [True], 'cuDNN case must not silently exercise the unfused fallback'
    else:
        assert fused_results == []

    grad = torch.randn_like(actual)
    actual.backward(grad)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad).item() > 0
    for name, parameter in real_parameters.items():
        if name.startswith('core_attention.indexer.'):
            assert not parameter.requires_grad and parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad).item() > 0, name
    if native is None:
        return
    native_hidden = hidden.detach().clone().requires_grad_()
    expected, expected_selection = native(native_hidden)
    torch.testing.assert_close(selected_rows[0], expected_selection, rtol=0, atol=0)
    assert_similarity(actual.detach(), expected.detach(), eps=1e-3, label='attention output')
    expected.backward(grad)
    assert_similarity(hidden.grad, native_hidden.grad, eps=1e-3, label='hidden gradient')
    for source, target in _PARAMETER_MAPPING.items():
        if source.startswith('indexer.'):
            assert native_parameters[source].grad is None
        else:
            assert_similarity(
                real_parameters[target].grad, native_parameters[source].grad, eps=1e-3, label=target
            )


@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_real_absorbed_mla_frozen_selector_forward_backward(backend, monkeypatch):
    _run_real_layer(backend, monkeypatch, compare_native=False)


@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_absorbed_mla_matches_independent_hf_attention(backend, monkeypatch):
    _run_real_layer(backend, monkeypatch, compare_native=True)


_LEFT_PADDING = 3


def _run_real_layer_left_padded(backend, monkeypatch):
    """Explicit-mask DSAttention with non-aligned left padding against padded HF math.

    AbsorbedMLA forwards its causal spec mask type; clearing it routes the explicit
    padding mask to DSAttention, the module-level path that accepts one.
    """
    config = _config(backend)
    spec = hybrid_glm53_stack_spec(config).submodules.dsa_layer.submodules.self_attention
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    real = build_module(spec, config=config, layer_number=1, pg_collection=groups).cuda()
    assert isinstance(real, AbsorbedMLASelfAttention)
    native = _NativeHFNoPEAttention(torch.device('cuda'))
    real_parameters = dict(real.named_parameters())
    native_parameters = dict(native.named_parameters())
    assert set(native_parameters) == set(_PARAMETER_MAPPING)
    assert set(real_parameters) == set(_PARAMETER_MAPPING.values())
    with torch.no_grad():
        for source, target in _PARAMETER_MAPPING.items():
            assert real_parameters[target].shape == native_parameters[source].shape
            real_parameters[target].copy_(native_parameters[source])

    selected_rows, fused_results = [], []
    sparse_attention = dsa_module._run_sparse_attention
    fused_attention = dsa_kernels.run_fused_absorbed_sparse_attention

    def capture_selection(**kwargs):
        selected_rows.append(kwargs['topk_indices'].detach().clone())
        return sparse_attention(**kwargs)

    def capture_fused_result(*args, **kwargs):
        result = fused_attention(*args, **kwargs)
        fused_results.append(result is not None)
        return result

    monkeypatch.setattr(dsa_module, '_run_sparse_attention', capture_selection)
    monkeypatch.setattr(dsa_kernels, 'run_fused_absorbed_sparse_attention', capture_fused_result)
    monkeypatch.setattr(real, 'attn_mask_type', None)
    sequence = 17
    positions = torch.arange(sequence, device='cuda')
    key_valid = (positions >= _LEFT_PADDING)[None]
    visible = (positions[None, :] <= positions[:, None]) & key_valid & key_valid.T
    # Boolean attention masks mark hidden keys; padded query rows hide every key.
    attention_mask = ~visible[None, None]
    hidden = torch.randn(
        sequence, 1, _HIDDEN, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )
    actual, bias = real(hidden, attention_mask=attention_mask)
    assert bias is None
    assert actual.shape == hidden.shape and torch.isfinite(actual).all()
    assert len(selected_rows) == 1
    assert selected_rows[0].shape == (1, sequence, _TOPK + _POOL - 1)
    assert torch.all(selected_rows[0][0, :_LEFT_PADDING] == -1)
    # Two selected pools plus the tail of the 14 visible keys.
    assert (selected_rows[0][0, -1] >= 0).sum().item() == _TOPK + 2
    if backend == 'cudnn':
        assert fused_results == [True], 'cuDNN case must not silently exercise the unfused fallback'
    else:
        assert fused_results == []

    # Padded query rows have no HF-defined output, so they carry no upstream gradient.
    grad = torch.randn_like(actual)
    grad[:_LEFT_PADDING] = 0
    actual.backward(grad)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, parameter in real_parameters.items():
        if name.startswith('core_attention.indexer.'):
            assert not parameter.requires_grad and parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad).item() > 0, name
    native_hidden = hidden.detach().clone().requires_grad_()
    expected, expected_selection = native(native_hidden, key_valid)
    torch.testing.assert_close(selected_rows[0], expected_selection, rtol=0, atol=0)
    assert_similarity(
        actual.detach()[_LEFT_PADDING:],
        expected.detach()[_LEFT_PADDING:],
        eps=1e-3,
        label='attention output',
    )
    expected.backward(grad)
    assert_similarity(hidden.grad, native_hidden.grad, eps=1e-3, label='hidden gradient')
    for source, target in _PARAMETER_MAPPING.items():
        if source.startswith('indexer.'):
            assert native_parameters[source].grad is None
        else:
            assert_similarity(
                real_parameters[target].grad, native_parameters[source].grad, eps=1e-3, label=target
            )


@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_absorbed_mla_left_padding_matches_independent_hf_attention(backend, monkeypatch):
    _run_real_layer_left_padded(backend, monkeypatch)
