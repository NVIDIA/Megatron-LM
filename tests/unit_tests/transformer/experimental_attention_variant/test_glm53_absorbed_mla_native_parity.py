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

The trainable cases turn the indexer loss on (a positive dsa_indexer_loss_coeff
with the sparse pool-level KL). The selection and the trunk must not change: HF
still selects the same pools, and a frozen layer with the same parameters gives
the same output and trunk gradients. Every indexer parameter, the pool gate and
the slot bias included, gets a finite gradient, and the logged loss matches the
independent reference in kpool_trainable_reference.py. The frozen HF indexer has
no training rule, so the indexer gradients are not compared with HF.
"""

import os
from types import SimpleNamespace

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
from tests.unit_tests.transformer.experimental_attention_variant import (
    kpool_trainable_reference as reference,
)
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
# A positive coefficient (with the sparse pool-level KL) trains the KPool indexer.
_LOSS_COEFF = 0.1
_INDEXER_PREFIX = 'core_attention.indexer.'
_GATE = f'{_INDEXER_PREFIX}index_kpool_compress_gate'
_APE = f'{_INDEXER_PREFIX}index_kpool_compress_ape'


def _config(backend, kpool_scorer='torch', *, trainable=False):
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
        dsa_indexer_kpool_scorer_backend=kpool_scorer,
        dsa_indexer_hf_scoring=True,
        dsa_indexer_loss_coeff=_LOSS_COEFF if trainable else 0.0,
        dsa_indexer_use_sparse_loss=trainable,
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
        kda_beta_sigmoid_in_projection_dtype=True,
        enable_hyper_connections=True,
        mhc_norm_eps_inside_sqrt=True,
        mhc_normalize_before_projection=True,
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


def _require_cudnn_kpool_scorer():
    """The skips of the frozen cuDNN-scorer case."""
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    try:
        dsa_cudnn_kernels._ensure_dsa_namespace()
    except ImportError as exc:
        pytest.skip(f'cuDNN frontend DSA unavailable: {exc}')
    if not dsa_cudnn_kernels._supports_indexer_q_causal_offsets():
        # 17 tokens end off the pool grid; without q_causal_offsets the PyTorch scorer runs.
        pytest.skip('this cuDNN frontend has no q_causal_offsets')


def _expected_loss_path(backend, kpool_scorer):
    """Where the trainable loss runs at TP1 with HF scoring and the fused sparse attention.

    The cuDNN frontend kernels need both cuDNN backends and frontend >= 1.31; otherwise the
    PyTorch ``KPoolSparseIndexerLoss`` runs.
    """
    if backend != 'cudnn' or kpool_scorer != 'cudnn':
        return 'pytorch'
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    minimum = dsa_cudnn_kernels._KPOOL_SPARSE_LOSS_MIN_FE_VERSION
    return 'cudnn' if dsa_cudnn_kernels._cudnn_frontend_at_least(minimum) else 'pytorch'


def _trace_trainable_selector(selector, monkeypatch):
    """Record, without changing anything, what a trainable selector's layer logs and runs.

    ``losses`` holds each logged indexer loss, ``paths`` which loss Function ran (the
    PyTorch ``KPoolSparseIndexerLoss`` or the cuDNN frontend one) and ``indexer`` each
    ``forward_before_topk`` result of ``selector``: ``q``, ``k``, ``weights`` and the
    pool-gate score. The loss enters the backward with scale 1, and the test logs into a
    tracker of its own.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    trace = SimpleNamespace(losses=[], paths=[], indexer=[])
    helper = dsa_module.DSAIndexerLossLoggingHelper
    monkeypatch.setattr(helper, 'tracker', {})
    monkeypatch.setattr(
        dsa_module.DSAIndexerLossAutoScaler,
        'main_loss_backward_scale',
        torch.ones((), device='cuda'),
    )
    save_loss = helper.save_loss_to_tracker

    def record_loss(*args, **kwargs):
        record = dict(kwargs)
        for name in ('loss', 'raw_loss'):
            if isinstance(record.get(name), torch.Tensor):
                record[name] = record[name].detach().clone()
        trace.losses.append(record)
        return save_loss(*args, **kwargs)

    monkeypatch.setattr(helper, 'save_loss_to_tracker', staticmethod(record_loss))
    for path, function in (
        ('pytorch', dsa_module.KPoolSparseIndexerLoss),
        ('cudnn', dsa_cudnn_kernels.KPoolSparseIndexerLossFromScoresFunc),
    ):

        def apply(*args, _apply=function.apply, _path=path, **kwargs):
            trace.paths.append(_path)
            return _apply(*args, **kwargs)

        monkeypatch.setattr(function, 'apply', apply)
    forward_before_topk = selector.forward_before_topk

    def capture_indexer(*args, **kwargs):
        outputs = forward_before_topk(*args, **kwargs)
        trace.indexer.append(tuple(output.detach().clone() for output in outputs))
        return outputs

    monkeypatch.setattr(selector, 'forward_before_topk', capture_indexer)
    return trace


def _selected_pool_ids(selection):
    """Pool ids ``[b, sq, G]`` (``-1`` pads) of a token top-k's pool slots.

    Each slot must hold ``_POOL`` consecutive tokens that start on the pool grid; the slot
    order (score order or the fused selector's kernel order) does not matter to the loss.
    """
    tokens = selection[..., :_TOPK].detach().to(device='cpu', dtype=torch.int64)
    slots = tokens.unflatten(-1, (_TOPK // _POOL, _POOL))
    first = slots[..., 0]
    valid = first >= 0
    consecutive = first.unsqueeze(-1) + torch.arange(_POOL)
    expected = torch.where(valid.unsqueeze(-1), consecutive, torch.full_like(slots, -1))
    assert torch.equal(slots, expected), 'a selected pool slot does not hold consecutive tokens'
    assert bool((first[valid] % _POOL == 0).all()), 'a selected pool is off the pool grid'
    return torch.where(valid, first // _POOL, torch.full_like(first, -1))


def _assert_trainable_loss(trace, selector, attention, selection, expected_path):
    """One forward logged one loss, on ``expected_path``, equal to the independent reference.

    The reference rebuilds the loss from the indexer's own tensors, the main query and key
    the sparse attention received (``attention``) and the selection it attended, in the
    layer's key-dtype compression profile, divided by every row of the batch.
    """
    assert trace.paths == [expected_path], f'loss Functions run: {trace.paths}'
    assert len(trace.losses) == 1, f'{len(trace.losses)} indexer losses logged for one forward'
    record = trace.losses[0]
    assert record['layer_number'] == 1
    loss = record['loss']
    assert torch.isfinite(loss).all() and loss.item() > 0, f'indexer loss {loss.item()}'
    torch.testing.assert_close(record['raw_loss'] * _LOSS_COEFF, loss, rtol=1e-6, atol=0)
    assert len(trace.indexer) == 1
    q, k, weights, gate_score = trace.indexer[0]
    batch, rows = selection.shape[:2]
    expected = reference.sparse_indexer_loss(
        q=q.cpu(),
        weights=weights.cpu(),
        k=k.cpu(),
        gate_score=gate_score.cpu(),
        ape=selector.index_kpool_compress_ape.detach().cpu(),
        query=attention['query'].cpu(),
        key=attention['key'].cpu(),
        pool_ids=_selected_pool_ids(selection),
        pool_size=_POOL,
        softmax_scale=attention['softmax_scale'],
        hf_scoring=True,
        use_relu=True,
        profile='key_dtype',
        loss_coeff=_LOSS_COEFF,
        row_count=float(batch * rows),
    )
    torch.testing.assert_close(
        loss.double().cpu(), expected.loss.detach().double(), rtol=1e-4, atol=1e-8
    )


def _attention_inputs(kwargs):
    """The main query and key (copies) and the softmax scale of a sparse-attention call."""
    return {
        'query': kwargs['query'].detach().clone(),
        'key': kwargs['key'].detach().clone(),
        'softmax_scale': kwargs['softmax_scale'],
    }


def _run_real_layer(backend, monkeypatch, *, compare_native, kpool_scorer='torch', trainable=False):
    config = _config(backend, kpool_scorer, trainable=trainable)
    spec = hybrid_glm53_stack_spec(config).submodules.dsa_layer.submodules.self_attention
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    real = build_module(spec, config=config, layer_number=1, pg_collection=groups).cuda()
    assert isinstance(real, AbsorbedMLASelfAttention)
    assert not real.use_rope
    selector = real.core_attention.indexer
    assert selector.index_kpool == _POOL
    # The loss on trains the whole KPool indexer, gate and slot bias included; off, it is frozen.
    assert all(parameter.requires_grad == trainable for parameter in selector.parameters())
    trace = _trace_trainable_selector(selector, monkeypatch) if trainable else None
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

    selected_rows, fused_results, kpool_results, attention_inputs = [], [], [], []
    sparse_attention = dsa_module._run_sparse_attention
    fused_attention = dsa_kernels.run_fused_absorbed_sparse_attention
    kpool_pool_topk = dsa_kernels.run_kpool_pool_topk

    def capture_selection(**kwargs):
        selected_rows.append(kwargs['topk_indices'].detach().clone())
        if trainable:
            attention_inputs.append(_attention_inputs(kwargs))
        return sparse_attention(**kwargs)

    def capture_fused_result(*args, **kwargs):
        result = fused_attention(*args, **kwargs)
        fused_results.append(result is not None)
        return result

    def capture_kpool_result(*args, **kwargs):
        result = kpool_pool_topk(*args, **kwargs)
        kpool_results.append(result is not None)
        return result

    monkeypatch.setattr(dsa_module, '_run_sparse_attention', capture_selection)
    monkeypatch.setattr(dsa_kernels, 'run_fused_absorbed_sparse_attention', capture_fused_result)
    monkeypatch.setattr(dsa_kernels, 'run_kpool_pool_topk', capture_kpool_result)
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
    # The cuDNN KPool scorer must run, not fall back to the PyTorch scorer.
    assert kpool_results == ([True] if kpool_scorer == 'cudnn' else [])
    if trainable:
        assert {_GATE, _APE} <= set(real_parameters)
        _assert_trainable_loss(
            trace,
            selector,
            attention_inputs[0],
            selected_rows[0],
            _expected_loss_path(backend, kpool_scorer),
        )

    grad = torch.randn_like(actual)
    actual.backward(grad)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad).item() > 0
    for name, parameter in real_parameters.items():
        if name.startswith(_INDEXER_PREFIX) and not trainable:
            assert not parameter.requires_grad and parameter.grad is None, name
        else:
            # Trainable: every indexer parameter, the gate and slot bias included, from the KL.
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad).item() > 0, name
    if native is None:
        return
    native_hidden = hidden.detach().clone().requires_grad_()
    expected, expected_selection = native(native_hidden)
    if kpool_scorer == 'cudnn':
        # The fused selector returns pools in its top-k kernel's order, not by score.
        torch.testing.assert_close(
            torch.sort(selected_rows[0], dim=-1).values,
            torch.sort(expected_selection, dim=-1).values,
            rtol=0,
            atol=0,
        )
    else:
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


@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_real_absorbed_mla_trainable_selector_forward_backward(backend, monkeypatch):
    """The loss on: every indexer parameter trains and the logged loss is the reference's."""
    _run_real_layer(backend, monkeypatch, compare_native=False, trainable=True)


@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_absorbed_mla_trainable_selector_matches_independent_hf_attention(
    backend, monkeypatch
):
    """The loss on: HF's frozen selection, attention output and trunk gradients still hold."""
    _run_real_layer(backend, monkeypatch, compare_native=True, trainable=True)


def test_glm53_absorbed_mla_cudnn_kpool_scorer_matches_independent_hf_attention(monkeypatch):
    """``dsa_indexer_kpool_scorer_backend='cudnn'`` selects the HF pools through DSAttention."""
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    try:
        dsa_cudnn_kernels._ensure_dsa_namespace()
    except ImportError as exc:
        pytest.skip(f'cuDNN frontend DSA unavailable: {exc}')
    if not dsa_cudnn_kernels._supports_indexer_q_causal_offsets():
        # 17 tokens end off the pool grid; without q_causal_offsets the PyTorch scorer runs.
        pytest.skip('this cuDNN frontend has no q_causal_offsets')
    _run_real_layer('cudnn', monkeypatch, compare_native=True, kpool_scorer='cudnn')


def test_glm53_absorbed_mla_trainable_cudnn_kpool_scorer_matches_independent_hf_attention(
    monkeypatch,
):
    """The cuDNN-scorer case with the loss on, which then runs on the cuDNN frontend kernels
    where the frontend is at least 1.31 (the attention also returns its LSE there)."""
    _require_cudnn_kpool_scorer()
    _run_real_layer('cudnn', monkeypatch, compare_native=True, kpool_scorer='cudnn', trainable=True)


def _run_layer_pass(layer, hidden, grad, calls):
    """One forward and backward of ``layer`` on a fresh leaf copy of ``hidden``."""
    layer.zero_grad(set_to_none=True)
    leaf = hidden.detach().clone().requires_grad_()
    output, bias = layer(leaf, attention_mask=None)
    assert bias is None and torch.isfinite(output).all()
    output.backward(grad)
    return SimpleNamespace(
        call=calls[-1],
        output=output.detach().clone(),
        hidden_grad=leaf.grad.detach().clone(),
        grads={
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters()
        },
    )


def _assert_as_frozen(label, trainable, frozen, repeat, *, atomic):
    """The trainable layer's tensor against the frozen layer's.

    With ``atomic`` (the cuDNN backend, whose sparse-attention backward accumulates the
    latent gradient atomically), runs need not agree bit for bit even after one pair did:
    the trainable tensor and the frozen repeat must lie within a rounding-level bar of the
    frozen tensor (a cosine deviation of 1e-4 is about a 1.4% relative difference).
    Otherwise bit for bit when the frozen layer repeats the tensor bit for bit, else within
    the same bar.
    """
    assert trainable is not None and frozen is not None and repeat is not None, label
    if not atomic and torch.equal(frozen, repeat):
        assert torch.equal(trainable, frozen), f'{label} differs from the frozen layer'
        return
    if atomic:
        assert_similarity(repeat, frozen, eps=1e-4, label=f'{label} (frozen repeat)')
    assert_similarity(trainable, frozen, eps=1e-4, label=label)


def _run_trainable_against_frozen(backend, kpool_scorer, monkeypatch):
    """A trainable layer against a frozen one with the same parameters, input and cotangent.

    The frozen layer runs twice: what it repeats bit for bit, the trainable layer must give
    bit for bit. The slot bias is non-zero, as in a trained checkpoint.
    """
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    layers = []
    for trainable in (False, True):
        config = _config(backend, kpool_scorer, trainable=trainable)
        spec = hybrid_glm53_stack_spec(config).submodules.dsa_layer.submodules.self_attention
        layer = build_module(spec, config=config, layer_number=1, pg_collection=groups).cuda()
        assert isinstance(layer, AbsorbedMLASelfAttention) and not layer.use_rope
        assert layer.core_attention.indexer.index_kpool == _POOL
        layers.append(layer)
    frozen, trainable = layers
    frozen_parameters = dict(frozen.named_parameters())
    trainable_parameters = dict(trainable.named_parameters())
    assert set(frozen_parameters) == set(trainable_parameters) == set(_PARAMETER_MAPPING.values())
    with torch.no_grad():
        frozen_parameters[_APE].uniform_(-1.0, 1.0)
        for name, parameter in trainable_parameters.items():
            assert parameter.dtype == frozen_parameters[name].dtype, name
            parameter.copy_(frozen_parameters[name])
    for name, parameter in frozen_parameters.items():
        assert parameter.requires_grad != name.startswith(_INDEXER_PREFIX), name
    assert all(parameter.requires_grad for parameter in trainable_parameters.values())
    assert trainable_parameters[_GATE].dtype == torch.bfloat16
    assert trainable_parameters[_APE].dtype == torch.float32

    calls, fused_results, kpool_results = [], [], []
    sparse_attention = dsa_module._run_sparse_attention
    fused_attention = dsa_kernels.run_fused_absorbed_sparse_attention
    kpool_pool_topk = dsa_kernels.run_kpool_pool_topk

    def capture_call(**kwargs):
        calls.append(dict(_attention_inputs(kwargs), selection=kwargs['topk_indices'].clone()))
        return sparse_attention(**kwargs)

    def capture_fused_result(*args, **kwargs):
        result = fused_attention(*args, **kwargs)
        fused_results.append(result is not None)
        return result

    def capture_kpool_result(*args, **kwargs):
        result = kpool_pool_topk(*args, **kwargs)
        kpool_results.append(result is not None)
        return result

    monkeypatch.setattr(dsa_module, '_run_sparse_attention', capture_call)
    monkeypatch.setattr(dsa_kernels, 'run_fused_absorbed_sparse_attention', capture_fused_result)
    monkeypatch.setattr(dsa_kernels, 'run_kpool_pool_topk', capture_kpool_result)
    trace = _trace_trainable_selector(trainable.core_attention.indexer, monkeypatch)
    hidden = torch.randn(17, 1, _HIDDEN, device='cuda', dtype=torch.bfloat16)
    grad = torch.randn_like(hidden)
    frozen_run = _run_layer_pass(frozen, hidden, grad, calls)
    repeat = _run_layer_pass(frozen, hidden, grad, calls)
    assert trace.losses == [] and trace.paths == [], 'a frozen layer logged an indexer loss'
    trainable_run = _run_layer_pass(trainable, hidden, grad, calls)
    assert len(calls) == 3
    if backend == 'cudnn':
        assert fused_results == [True] * 3, 'cuDNN case must not exercise the unfused fallback'
    else:
        assert fused_results == []
    # The cuDNN KPool scorer must run, not fall back to the PyTorch scorer.
    assert kpool_results == ([True] * 3 if kpool_scorer == 'cudnn' else [])
    selection = trainable_run.call['selection']
    _assert_trainable_loss(
        trace,
        trainable.core_attention.indexer,
        trainable_run.call,
        selection,
        _expected_loss_path(backend, kpool_scorer),
    )

    # The fused selector returns pools in its top-k kernel's order, not by score.
    frozen_selection, repeat_selection = frozen_run.call['selection'], repeat.call['selection']
    torch.testing.assert_close(
        torch.sort(selection, dim=-1).values,
        torch.sort(frozen_selection, dim=-1).values,
        rtol=0,
        atol=0,
    )
    assert (selection[0, -1] >= 0).sum().item() == 9
    # The PyTorch scorer's top-k order is deterministic; the cuDNN kernel's is checked as a set.
    if kpool_scorer == 'torch':
        assert torch.equal(frozen_selection, repeat_selection), 'the frozen order changed'
        assert torch.equal(selection, frozen_selection), 'the selection order changed'
    atomic = backend == 'cudnn'
    _assert_as_frozen(
        'attention output', trainable_run.output, frozen_run.output, repeat.output, atomic=atomic
    )
    _assert_as_frozen(
        'hidden gradient',
        trainable_run.hidden_grad,
        frozen_run.hidden_grad,
        repeat.hidden_grad,
        atomic=atomic,
    )
    for name, parameter in trainable_parameters.items():
        trained = trainable_run.grads[name]
        if name.startswith(_INDEXER_PREFIX):
            assert frozen_run.grads[name] is None and repeat.grads[name] is None, name
            assert trained is not None and trained.dtype == parameter.dtype, name
            assert torch.isfinite(trained).all(), name
            assert torch.count_nonzero(trained).item() > 0, name
        else:
            _assert_as_frozen(
                name, trained, frozen_run.grads[name], repeat.grads[name], atomic=atomic
            )


@pytest.mark.parametrize(
    ('backend', 'kpool_scorer'),
    [('none', 'torch'), ('cudnn', 'torch'), ('cudnn', 'cudnn')],
    ids=['none', 'cudnn', 'cudnn-kpool-scorer'],
)
def test_glm53_absorbed_mla_trainable_selector_matches_the_frozen_layer(
    backend, kpool_scorer, monkeypatch
):
    """The loss on trains every indexer parameter, the gate and slot bias included, and
    leaves the selection, the attention output and every trunk gradient of the frozen
    layer unchanged; the logged loss is the independent reference's."""
    if kpool_scorer == 'cudnn':
        _require_cudnn_kpool_scorer()
    _run_trainable_against_frozen(backend, kpool_scorer, monkeypatch)


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


@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_absorbed_mla_trainable_selector_rejects_an_explicit_mask(backend, monkeypatch):
    """The left-padded explicit mask stays frozen-only: a trainable selector raises on it.

    DSAttention turns the boolean padding mask into its additive float mask, a layout the
    trainable KPool path does not cover; the frozen selector accepts it (the case above).
    """
    config = _config(backend, trainable=True)
    spec = hybrid_glm53_stack_spec(config).submodules.dsa_layer.submodules.self_attention
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    real = build_module(spec, config=config, layer_number=1, pg_collection=groups).cuda()
    assert isinstance(real, AbsorbedMLASelfAttention)
    assert all(parameter.requires_grad for parameter in real.core_attention.indexer.parameters())
    monkeypatch.setattr(real, 'attn_mask_type', None)
    sequence = 17
    positions = torch.arange(sequence, device='cuda')
    key_valid = (positions >= _LEFT_PADDING)[None]
    visible = (positions[None, :] <= positions[:, None]) & key_valid & key_valid.T
    attention_mask = ~visible[None, None]
    hidden = torch.randn(
        sequence, 1, _HIDDEN, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )
    with pytest.raises(ValueError, match='explicit attention mask'):
        real(hidden, attention_mask=attention_mask)
