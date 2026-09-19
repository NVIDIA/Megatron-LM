# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for Qwen Sparse Attention (QSA).

The indexer reference is a port of the HuggingFace ``Qwen4ExpTextQSAIndexer.forward`` loop.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant.qsa import (
    QwenSparseSelfAttention,
    build_qsa_dense_mask,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _make_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        use_cpu_initialization=False,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        qk_layernorm=True,
        attention_output_gate=True,
        add_bias_linear=False,
        qsa_indexer_n_heads=2,
        qsa_indexer_kv_heads=1,
        qsa_indexer_head_dim=16,
        qsa_indexer_budget=8,
        qsa_indexer_compress_ratio=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.unfused,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


ROTARY_PERCENT = 0.5  # rot_dim = 8 of the 16-dim heads (Qwen4-Exp: 64 of 256)


def _rotary(config, seq_len):
    rope = RotaryEmbedding(
        kv_channels=config.kv_channels, rotary_percent=ROTARY_PERCENT, rotary_base=10000
    )
    return rope(seq_len)  # [s, 1, 1, rot_dim]


def _hf_rope(x, freqs):
    """HF apply_rotary_pos_emb on the first rot_dim dims; x [..., D], freqs [..., rot_dim]."""
    rot = freqs.shape[-1]
    cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)


def _zero_centered_rmsnorm(x, weight, eps):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def _reference_selected_mask(attn: QwenSparseSelfAttention, hidden_sbh, freqs):
    """Port of the HF indexer loop for a single unpacked sequence; returns [b, s, s] bool."""
    indexer = attn.indexer
    config = attn.config
    s, b, _ = hidden_sbh.shape
    D, H, R = indexer.head_dim, indexer.n_heads, indexer.compress_ratio
    eps = config.layernorm_epsilon
    qk = F.linear(hidden_sbh.transpose(0, 1), indexer.index_qk_proj.weight)  # [b, s, (H+1)*D]
    q, raw_keys = torch.split(qk, [H * D, D], dim=-1)
    q = _zero_centered_rmsnorm(q.reshape(b, s, H, D), indexer.q_layernorm.weight, eps)
    freqs_bs = freqs.reshape(1, s, 1, -1)
    q = _hf_rope(q, freqs_bs)
    mask = torch.zeros(b, s, s, dtype=torch.bool, device=hidden_sbh.device)
    for bi in range(b):
        for t in range(s):
            visible = torch.arange(t + 1, device=hidden_sbh.device)
            n_complete = visible.numel() // R
            selected = []
            if n_complete > 0:
                block_tokens = visible[: n_complete * R].view(n_complete, R)
                pooled = (
                    raw_keys[bi][block_tokens.flatten()]
                    .view(n_complete, R, D)
                    .float()
                    .mean(1)
                    .to(raw_keys.dtype)
                )
                pooled = _zero_centered_rmsnorm(pooled, indexer.k_layernorm.weight, eps)
                pooled = _hf_rope(pooled, freqs.reshape(s, -1)[block_tokens[:, 0]])
                scores = torch.matmul(q[bi, t].float(), pooled.float().T).transpose(
                    -1, -2
                )  # [n_complete, H]
                scores = torch.relu(scores).sum(-1) / math.sqrt(D)
                top = scores.topk(min(indexer.block_topk, n_complete)).indices
                selected.append(block_tokens[top].flatten())
            selected.append(visible[n_complete * R :])
            mask[bi, t, torch.cat(selected)] = True
    return mask


class TestQwenSparseAttention:
    _NVTE_VARS = {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "1"}

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        # The dense reference path runs TE's unfused (pure PyTorch) attention: it is the
        # only TE backend that accepts every head size used here.
        self._saved_env = {k: os.environ.get(k) for k in self._NVTE_VARS}
        os.environ.update(self._NVTE_VARS)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        os.environ.pop("MCORE_QSA_SPARSE_BACKEND", None)
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _build(self, config):
        spec = get_qsa_module_spec_for_backend(config)
        attn = build_module(spec, config=config, layer_number=1).cuda()
        with torch.no_grad():  # non-trivial norms so the reference exercises the (1 + w) gain
            attn.indexer.q_layernorm.weight.normal_(0, 0.1)
            attn.indexer.k_layernorm.weight.normal_(0, 0.1)
        return attn

    def test_indexer_selection_matches_reference(self):
        config = _make_config()
        attn = self._build(config)
        s, b = 41, 2
        hidden = torch.randn(s, b, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        freqs = _rotary(config, s).cuda()

        selection = attn.indexer(hidden, freqs, None)
        assert not selection.all_selected  # 10 complete blocks > block_topk=2
        got = build_qsa_dense_mask(selection, s)
        ref = _reference_selected_mask(attn, hidden, freqs)

        # Ties in the fp32 scores can flip a block near the top-k boundary; require exact match
        # on the deterministic parts (causal + tail) and >= 99% agreement overall.
        assert torch.equal(got.tril(), got)
        agreement = (got == ref).float().mean().item()
        assert agreement > 0.99, agreement
        per_query = got.sum(-1)
        budget = config.qsa_indexer_budget
        assert (per_query <= budget + config.qsa_indexer_compress_ratio - 1).all()

    def test_short_sequence_selects_everything(self):
        config = _make_config()
        attn = self._build(config)
        s = config.qsa_indexer_budget + config.qsa_indexer_compress_ratio - 1
        hidden = torch.randn(s, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        selection = attn.indexer(hidden, _rotary(config, s).cuda(), None)
        assert selection.all_selected
        mask = build_qsa_dense_mask(selection, s)
        assert torch.equal(mask[0], torch.ones(s, s, dtype=torch.bool, device="cuda").tril())

    @pytest.mark.parametrize("backend", ["flex", "dense_masked"])
    def test_forced_sparse_matches_dense_on_short_sequence(self, backend):
        """With every block selected the sparse kernels must reproduce dense causal attention."""
        config = _make_config()
        attn = self._build(config)
        s, b = 11, 2
        hidden = torch.randn(s, b, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        freqs = _rotary(config, s).cuda()
        dense_out, _ = attn(hidden, attention_mask=None, rotary_pos_emb=freqs)

        os.environ["MCORE_QSA_SPARSE_BACKEND"] = backend
        attn.core_attention.sparse_backend = backend
        attn.config.qsa_force_sparse = True
        sparse_out, _ = attn(hidden, attention_mask=None, rotary_pos_emb=freqs)
        attn.config.qsa_force_sparse = False

        torch.testing.assert_close(sparse_out.float(), dense_out.float(), atol=2e-2, rtol=2e-2)

    def test_flex_matches_dense_masked_on_long_sequence(self):
        config = _make_config()
        attn = self._build(config)
        s, b = 45, 2
        hidden = torch.randn(s, b, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        freqs = _rotary(config, s).cuda()

        attn.core_attention.sparse_backend = "dense_masked"
        ref_out, _ = attn(hidden, attention_mask=None, rotary_pos_emb=freqs)
        attn.core_attention.sparse_backend = "flex"
        flex_out, _ = attn(hidden, attention_mask=None, rotary_pos_emb=freqs)

        torch.testing.assert_close(flex_out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)

    def test_packed_sequences_select_per_document(self):
        config = _make_config()
        attn = self._build(config)
        lens = [21, 30]
        t = sum(lens)
        hidden = torch.randn(t, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, lens[0], t], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(lens),
            max_seqlen_kv=max(lens),
        )
        freqs = _rotary(config, max(lens)).cuda()
        selection = attn.indexer(hidden, freqs, packed)
        mask = build_qsa_dense_mask(selection, t)[0]
        # No cross-document attention, causal within documents.
        assert not mask[: lens[0], lens[0] :].any()
        assert not mask[lens[0] :, : lens[0]].any()
        assert torch.equal(mask.tril(), mask)
        # Each document's selection equals the single-sequence selection of that document.
        for start, length in zip([0, lens[0]], lens):
            single = attn.indexer(
                hidden[start : start + length], _rotary(config, length).cuda(), None
            )
            single_mask = build_qsa_dense_mask(single, length)[0]
            assert torch.equal(mask[start : start + length, start : start + length], single_mask)

    def test_backward_through_sparse_attention(self):
        config = _make_config()
        attn = self._build(config)
        s, b = 45, 1
        hidden = torch.randn(
            s, b, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        freqs = _rotary(config, s).cuda()
        out, _ = attn(hidden, attention_mask=None, rotary_pos_emb=freqs)
        out.float().square().sum().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert attn.linear_qkv.weight.grad is not None
