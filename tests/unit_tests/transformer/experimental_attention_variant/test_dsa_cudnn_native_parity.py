# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

_SIMILARITY_EPS = 1e-3
# Fused dense-loss path: the cuDNN dense indexer backward consumes raw scores plus
# L1-norm/LSE separately, so its indexer param-grad parity has the same precision
# floor as the DSv4 fused dense-loss parity test.
_FUSED_DENSE_INDEXER_GRAD_SIMILARITY_EPS = 3e-3


def _skip_if_fused_dsa_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused DSA parity")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cudnn fused DSA path requires SM100+")
    missing = []
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        missing.append("cudnn-frontend DSA (nvidia-cudnn-frontend[cutedsl]>=1.24.1)")
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        missing.append("flash_mla")
    if missing:
        pytest.skip(f"fused DSA dependencies are not available: {', '.join(missing)}")


def _mock_rotate_activation(x: torch.Tensor) -> torch.Tensor:
    return x


def _make_config(
    use_sparse_loss: bool = True,
    calculate_per_token_loss: bool = False,
    dsa_kernel_backend: str = "none",
) -> MLATransformerConfig:
    return MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=1,
        hidden_size=7168,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=2048,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=use_sparse_loss,
        calculate_per_token_loss=calculate_per_token_loss,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_scaling_factor=40,
        mscale=1.0,
        mscale_all_dim=1.0,
        rotary_base=10000,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        rotary_interleaved=False,
        recompute_granularity=None,
        recompute_modules=[],
        fine_grained_activation_offloading=False,
        gradient_accumulation_fusion=False,
        fp8=False,
        fp4=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        kv_channels=128,
        num_query_groups=128,
        batch_invariant_mode=False,
        cache_mla_latents=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        symmetric_ar_type=None,
        disable_parameter_transpose_cache=False,
        init_model_with_meta_device=False,
        delay_wgrad_compute=False,
        tp_comm_overlap=False,
        softmax_scale=None,
        attention_backend=AttnBackend.unfused,
        dsa_kernel_backend=dsa_kernel_backend,
    )


def _precompute_freqs_cis(
    qk_pos_emb_head_dim: int,
    seqlen: int,
    device: torch.device,
    rotary_base: float,
    rotary_scaling_factor: float,
    original_max_position_embeddings: int,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    dim = qk_pos_emb_head_dim
    base = rotary_base
    factor = rotary_scaling_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_value, max_value, dim):
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_value) / (
            max_value - min_value
        )
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if seqlen > original_max_position_embeddings:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_max_position_embeddings
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


class NativeIndexer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        qk_pos_emb_head_dim: int,
        index_topk: int,
        use_sparse_loss: bool,
        layernorm_epsilon: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.index_topk = index_topk
        self.use_sparse_loss = use_sparse_loss
        self.softmax_scale = self.index_head_dim**-0.5

        self.linear_wq_b = nn.Linear(
            self.q_lora_rank, self.index_n_heads * self.index_head_dim, bias=False
        )
        self.linear_wk = nn.Linear(self.hidden_size, self.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.index_head_dim, eps=layernorm_epsilon)
        self.linear_weights_proj = nn.Linear(self.hidden_size, self.index_n_heads, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_wq_b(qr).view(
            qr.size(0), qr.size(1), self.index_n_heads, self.index_head_dim
        )
        q_pe, q_nope = torch.split(
            q, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        q_pe = _apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.linear_wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        k_pe = _apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        q = _mock_rotate_activation(q)
        k = _mock_rotate_activation(k)
        weights = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5)

        logits = torch.einsum("bthd,bsd->bths", q, k)
        logits = F.relu(logits)
        logits = logits * weights.unsqueeze(-1)
        logits = logits * self.softmax_scale
        logits = logits.sum(dim=2)
        logits = logits + attention_mask.squeeze(1)
        topk = min(self.index_topk, x.size(1))
        topk_logits, topk_indices = logits.topk(topk, dim=-1)
        log_topk_prob = F.log_softmax(topk_logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        return topk_indices, log_topk_prob, log_prob


class NativeDSA(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_head_dim: int,
        qk_pos_emb_head_dim: int,
        v_head_dim: int,
        dsa_indexer_n_heads: int,
        dsa_indexer_head_dim: int,
        dsa_indexer_topk: int,
        dsa_indexer_use_sparse_loss: bool,
        layernorm_epsilon: float,
        rotary_base: float,
        rotary_scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: int,
        beta_slow: int,
        mscale: float,
        rope_factor: float,
        calculate_per_token_loss: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.v_head_dim = v_head_dim
        self.rotary_base = rotary_base
        self.rotary_scaling_factor = rotary_scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.rope_factor = rope_factor
        self.use_sparse_loss = dsa_indexer_use_sparse_loss
        self.calculate_per_token_loss = calculate_per_token_loss

        self.q_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        self.linear_q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_layernorm = nn.RMSNorm(self.q_lora_rank, eps=layernorm_epsilon)
        self.linear_q_up_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.linear_kv_down_proj = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_pos_emb_head_dim, bias=False
        )
        self.kv_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=layernorm_epsilon)
        self.linear_kv_up_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_head_dim + self.v_head_dim), bias=False
        )
        self.linear_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        self.indexer = NativeIndexer(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            index_n_heads=dsa_indexer_n_heads,
            index_head_dim=dsa_indexer_head_dim,
            qk_pos_emb_head_dim=self.qk_pos_emb_head_dim,
            index_topk=dsa_indexer_topk,
            use_sparse_loss=dsa_indexer_use_sparse_loss,
            layernorm_epsilon=layernorm_epsilon,
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = hidden_states.transpose(0, 1).contiguous()  # [b, s, h]
        bsz, seqlen, _ = x.shape
        freqs_cis = _precompute_freqs_cis(
            qk_pos_emb_head_dim=self.qk_pos_emb_head_dim,
            seqlen=seqlen,
            device=x.device,
            rotary_base=self.rotary_base,
            rotary_scaling_factor=self.rotary_scaling_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
        )

        qr = self.q_layernorm(self.linear_q_down_proj(x))
        q = self.linear_q_up_proj(qr).view(bsz, seqlen, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1)
        q_pe = _apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

        kv = self.linear_kv_down_proj(x)
        kv_compressed, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_pos_emb_head_dim], dim=-1)
        kv_compressed = self.kv_layernorm(kv_compressed)
        k_pe = _apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)

        kv_up_weight = self.linear_kv_up_proj.weight.view(
            self.num_heads, self.qk_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        k_up_weight = kv_up_weight[:, : self.qk_head_dim, :]
        q_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, k_up_weight)
        q_absorbed = torch.cat([q_absorbed, q_pe], dim=-1)

        topk_indices, log_topk_prob, log_prob = self.indexer(
            x.detach(), qr.detach(), freqs_cis, attention_mask
        )

        keys = torch.cat([kv_compressed, k_pe.squeeze(2)], dim=-1)
        values = kv_compressed
        scores = torch.einsum("bshd,btd->bsht", q_absorbed, keys) * self.softmax_scale
        scores = scores + attention_mask.squeeze(1).unsqueeze(2)
        dense_scores = scores

        index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device)
        index_mask.scatter_(-1, topk_indices, 0)
        scores = scores + index_mask.unsqueeze(2)

        probs = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("bsht,btd->bshd", probs.to(values.dtype), values)

        if self.use_sparse_loss:
            gather_index = topk_indices.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            sparse_probs_topk = torch.gather(probs.detach(), dim=-1, index=gather_index)
            attn_score = sparse_probs_topk.sum(dim=2)
            attn_score = attn_score / attn_score.sum(dim=-1, keepdim=True)
            predict_log_prob = log_topk_prob
        else:
            dense_probs = F.softmax(dense_scores, dim=-1, dtype=torch.float32)
            attn_score = dense_probs.detach().sum(dim=2)
            attn_score = attn_score / attn_score.sum(dim=-1, keepdim=True)
            predict_log_prob = log_prob

        indexer_kl_loss = F.kl_div(
            predict_log_prob.clip(-100, 0),
            attn_score.log().clip(-100, 0),
            log_target=True,
            reduction="sum",
        )
        if not self.calculate_per_token_loss:
            indexer_kl_loss = indexer_kl_loss / (bsz * seqlen)

        v_up_weight = kv_up_weight[:, self.qk_head_dim :, :]
        attn_out = torch.einsum("bshc,hdc->bshd", attn_out, v_up_weight)

        out = self.linear_proj(attn_out.reshape(bsz, seqlen, -1))
        return out.transpose(0, 1).contiguous(), indexer_kl_loss


def _cosine_sim(a, b):
    return F.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a, b):
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_similarity(
    a: torch.Tensor, b: torch.Tensor, eps: float = _SIMILARITY_EPS, label: str = ""
):
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    c = _cosine_sim(a, b)
    t = _tensor_sim(a, b)
    prefix = f"{label}: " if label else ""
    assert c > 1 - eps, f"{prefix}cosine_sim={c:.6f}"
    assert t > 1 - eps, f"{prefix}tensor_sim={t:.6f}"


def test_cudnn_indexer_topk_varlen_uses_logical_query_positions(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            b, sq, _, _ = q_bshd.shape
            sk = k_bshd.size(1)
            key_ids = torch.arange(sk, dtype=torch.float32).view(1, 1, sk).expand(b, sq, sk)
            query_ids = torch.arange(sq).view(1, sq, 1)
            scores = key_ids.clone().masked_fill(
                torch.arange(sk).view(1, 1, sk) > query_ids, float("-inf")
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            return {"indices": scores.topk(top_k, dim=-1).indices.to(torch.int32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    q = torch.zeros((1, 2, 1, 1))
    k = torch.zeros((1, 8, 1))
    w = torch.zeros((1, 2, 1))
    starts = torch.tensor([3, 6], dtype=torch.int64)
    ends = torch.tensor([5, 8], dtype=torch.int64)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        q,
        k,
        w,
        topk=4,
        varlen_starts=starts,
        varlen_ends=ends,
        key_positions=torch.arange(8, dtype=torch.int64),
    )

    expected = torch.tensor([[[4, 3, -1, -1], [7, 6, -1, -1]]], dtype=torch.int32)
    torch.testing.assert_close(topk_indices, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        topk_length, torch.tensor([[2, 2]], dtype=torch.int32), rtol=0, atol=0
    )


def test_cudnn_indexer_topk_local_varlen_keeps_compact_query_rows(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            seen["score_sq"] = q_bshd.size(1)
            scores = (
                torch.arange(k_bshd.size(1), dtype=torch.float32)
                .view(1, 1, k_bshd.size(1))
                .expand(q_bshd.size(0), q_bshd.size(1), k_bshd.size(1))
                .clone()
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            return {"indices": scores.topk(top_k, dim=-1).indices.to(torch.int32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 2, 1, 1)),
        torch.zeros((1, 8, 1)),
        torch.zeros((1, 2, 1)),
        topk=4,
        varlen_starts=torch.tensor([3, 6], dtype=torch.int64),
        varlen_ends=torch.tensor([5, 8], dtype=torch.int64),
        key_positions=torch.arange(8, dtype=torch.int64),
        return_scores=False,
        use_local_indexer_varlen=True,
    )

    assert seen["score_sq"] == 2
    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[[3, 4, -1, -1], [6, 7, -1, -1]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        topk_length, torch.tensor([[2, 2]], dtype=torch.int32), rtol=0, atol=0
    )


def test_cudnn_indexer_topk_indices_only_filters_masked_varlen_scores(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            scores = (
                torch.arange(8, dtype=torch.float32)
                .view(1, 1, 8)
                .expand(q_bshd.size(0), q_bshd.size(1), 8)
                .clone()
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            assert return_val is False
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            return {"indices": scores.topk(top_k, dim=-1).indices.to(torch.int32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 2, 1, 1)),
        torch.zeros((1, 8, 1)),
        torch.zeros((1, 2, 1)),
        topk=4,
        varlen_starts=torch.tensor([3, 6], dtype=torch.int64),
        varlen_ends=torch.tensor([5, 8], dtype=torch.int64),
        key_positions=torch.arange(8, dtype=torch.int64),
        return_scores=False,
    )

    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[[3, 4, -1, -1], [6, 7, -1, -1]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        topk_length, torch.tensor([[2, 2]], dtype=torch.int32), rtol=0, atol=0
    )


def test_cudnn_indexer_topk_indices_only_compacts_invalid_prefix_entries(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            scores = (
                torch.arange(8, dtype=torch.float32)
                .view(1, 1, 8)
                .expand(q_bshd.size(0), q_bshd.size(1), 8)
                .clone()
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            assert return_val is False
            indices = torch.tensor([[4, 6, 3, 5]], dtype=torch.int32)
            return {"indices": indices.to(device=scores_flat.device)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 8, 1)),
        torch.zeros((1, 1, 1)),
        topk=4,
        varlen_starts=torch.tensor([3], dtype=torch.int64),
        varlen_ends=torch.tensor([5], dtype=torch.int64),
        key_positions=torch.arange(8, dtype=torch.int64),
        return_scores=False,
    )

    torch.testing.assert_close(
        topk_indices, torch.tensor([[[3, 4, -1, -1]]], dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(topk_length, torch.tensor([[2]], dtype=torch.int32), rtol=0, atol=0)


def test_cudnn_indexer_topk_can_return_topk_scores(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio, sm_scale):
            return {"scores": torch.tensor([[[0.5, float("-inf"), 2.0, 1.0]]], dtype=torch.float32)}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            seen["return_val"] = return_val
            values, indices = scores_flat.topk(top_k, dim=-1)
            if return_val:
                return {"indices": indices.to(torch.int32), "values": values}
            return {"indices": indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 4, 1)),
        torch.zeros((1, 1, 1)),
        topk=4,
        return_scores=False,
        return_topk_scores=True,
    )

    assert seen["return_val"] is True
    torch.testing.assert_close(topk_indices, torch.tensor([[[2, 3, 0, -1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[3]], dtype=torch.int32))
    torch.testing.assert_close(
        topk_scores, torch.tensor([[[2.0, 1.0, 0.5, torch.finfo(torch.float32).min]]])
    )


def test_cudnn_attention_topk_preparation_preserves_valid_prefix():
    topk_indices = torch.tensor([[[3, -1, 1, 2], [4, 0, -1, -1]]], dtype=torch.int32)

    prepared, topk_length = dsa_cudnn_kernels._prepare_attention_topk_indices(topk_indices, sk=5)

    torch.testing.assert_close(
        prepared, torch.tensor([[[1, 2, 3, -1], [0, 4, -1, -1]]], dtype=torch.int32)
    )
    torch.testing.assert_close(topk_length, torch.tensor([[3, 2]], dtype=torch.int32))


def test_flash_mla_topk_alignment_uses_sm100_block(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))

    assert dsa_cudnn_kernels._get_topk_alignment() == 512


def test_dsa_fwd_flash_mla_pads_topk_to_flashmla_block(monkeypatch):
    seen = {}

    def fake_flash_mla(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, indexer_topk):
        seen["indices"] = indices
        seen["topk_length"] = topk_length
        return (
            torch.zeros((q.size(0), q.size(1), d_v), dtype=q.dtype),
            torch.zeros((q.size(0), q.size(1)), dtype=torch.float32),
            torch.zeros((q.size(0), q.size(1)), dtype=torch.float32),
        )

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_flash_mla", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_flash_mla_sparse_fwd", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_get_topk_alignment", lambda: 512)
    monkeypatch.setattr(dsa_cudnn_kernels, "_get_head_padding", lambda num_heads: num_heads)

    topk_idxs = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32)
    topk_length = torch.tensor([3, 3], dtype=torch.int32)

    dsa_cudnn_kernels._dsa_fwd_flash_mla(
        torch.zeros((2, 2, 4), dtype=torch.bfloat16),
        torch.zeros((5, 4), dtype=torch.bfloat16),
        topk_idxs,
        1.0,
        d_v=4,
        attn_sink=torch.full((2,), float("-inf"), dtype=torch.float32),
        topk_length=topk_length,
    )

    assert seen["indices"].shape == (2, 1, 512)
    torch.testing.assert_close(seen["indices"][:, 0, :3], topk_idxs)
    torch.testing.assert_close(
        seen["indices"][:, 0, 3:], torch.full((2, 509), -1, dtype=torch.int32)
    )
    torch.testing.assert_close(seen["topk_length"], topk_length)


def test_cudnn_indexer_topk_scores_varlen_uses_bounds_for_length(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio, sm_scale):
            seen["score_sq"] = q.size(1)
            key_ids = torch.arange(k.size(1), dtype=torch.float32).view(1, 1, k.size(1))
            scores = key_ids.expand(q.size(0), q.size(1), k.size(1)).clone()
            scores.masked_fill_(
                torch.arange(k.size(1)).view(1, 1, k.size(1))
                > torch.arange(q.size(1)).view(1, q.size(1), 1),
                float("-inf"),
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            values, indices = scores_flat.topk(top_k, dim=-1)
            if return_val:
                return {"indices": indices.to(torch.int32), "values": values}
            return {"indices": indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 4, 1)),
        torch.zeros((1, 1, 1)),
        topk=4,
        varlen_starts=torch.tensor([2], dtype=torch.int64),
        varlen_ends=torch.tensor([4], dtype=torch.int64),
        key_positions=torch.arange(4, dtype=torch.int64),
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
    )

    assert seen["score_sq"] == 4
    torch.testing.assert_close(topk_indices, torch.tensor([[[3, 2, -1, -1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[2]], dtype=torch.int32))
    torch.testing.assert_close(
        topk_scores,
        torch.tensor(
            [[[3.0, 2.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]]]
        ),
    )


def test_cudnn_sparse_attn_target_uses_frontend_wrapper(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attn_score_recompute_wrapper(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            softmax_scale,
            qhead_per_kv_head=None,
            topk_indices_global=True,
        ):
            seen["q_attn"] = q_attn
            seen["k_attn"] = k_attn
            seen["lse_is_contiguous"] = lse.is_contiguous()
            seen["topk_indices_is_contiguous"] = topk_indices.is_contiguous()
            seen["softmax_scale"] = softmax_scale
            seen["qhead_per_kv_head"] = qhead_per_kv_head
            seen["topk_indices_global"] = topk_indices_global
            return {"target": torch.ones_like(topk_indices, dtype=torch.float32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    q = torch.randn((1, 2, 8, 4), dtype=torch.bfloat16)
    k = torch.randn((1, 3, 4), dtype=torch.bfloat16)
    lse = torch.randn((1, 2, 8), dtype=torch.float32).transpose(1, 2).transpose(1, 2)
    topk_indices = torch.tensor([[[0, 1, -1], [2, 0, 1]]], dtype=torch.int32)

    target = dsa_cudnn_kernels._compute_attn_target(
        q, k, lse, topk_indices, softmax_scale=0.5, qhead_per_kv_head=8
    )

    torch.testing.assert_close(target, torch.ones_like(target))
    assert seen["q_attn"] is q
    assert seen["k_attn"] is k
    assert seen["lse_is_contiguous"] is True
    assert seen["topk_indices_is_contiguous"] is True
    assert seen["softmax_scale"] == 0.5
    assert seen["qhead_per_kv_head"] == 8
    assert seen["topk_indices_global"] is False


def test_cudnn_sparse_attn_target_pads_small_local_head_count(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attn_score_recompute_wrapper(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            softmax_scale,
            qhead_per_kv_head=None,
            topk_indices_global=True,
        ):
            seen["q_attn_shape"] = q_attn.shape
            seen["lse_shape"] = lse.shape
            seen["q_real"] = q_attn[:, :, :4, :].clone()
            seen["q_pad_abs_sum"] = q_attn[:, :, 4:, :].abs().sum()
            seen["lse_real"] = lse[:, :, :4].clone()
            seen["lse_pad_is_inf"] = torch.isinf(lse[:, :, 4:]).all()
            seen["qhead_per_kv_head"] = qhead_per_kv_head
            seen["topk_indices_global"] = topk_indices_global
            return {"target": torch.ones_like(topk_indices, dtype=torch.float32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    q = torch.randn((1, 2, 4, 4), dtype=torch.bfloat16)
    k = torch.randn((1, 3, 4), dtype=torch.bfloat16)
    lse = torch.randn((1, 2, 4), dtype=torch.float32)
    topk_indices = torch.tensor([[[0, 1, -1], [2, 0, 1]]], dtype=torch.int32)

    target = dsa_cudnn_kernels._compute_attn_target(
        q, k, lse, topk_indices, softmax_scale=0.5, qhead_per_kv_head=4
    )

    torch.testing.assert_close(target, torch.ones_like(target))
    assert seen["q_attn_shape"] == (1, 2, 8, 4)
    assert seen["lse_shape"] == (1, 2, 8)
    torch.testing.assert_close(seen["q_real"], q)
    assert seen["q_pad_abs_sum"].item() == 0
    torch.testing.assert_close(seen["lse_real"], lse)
    assert seen["lse_pad_is_inf"].item() is True
    assert seen["qhead_per_kv_head"] == 8
    assert seen["topk_indices_global"] is False


def test_cudnn_sparse_loss_uses_selected_topk_scores(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_indexer_score_recompute_wrapper(*args, **kwargs):
            raise AssertionError("sparse indexer predict should use selected top-k scores")

        @staticmethod
        def indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            index_score,
            topk_indices,
            sm_scale,
            loss_coeff,
            grad_loss,
            block_I,
        ):
            seen["index_score"] = index_score
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    def fake_indexer_topk(
        q_bshd,
        k_bsd,
        w_bsh,
        topk,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        return_scores=True,
        return_topk_scores=False,
        use_local_indexer_varlen=False,
    ):
        seen["return_scores"] = return_scores
        seen["return_topk_scores"] = return_topk_scores
        return (
            torch.tensor([[[3, 1, 2, -1]]], dtype=torch.int32),
            torch.tensor([[3]], dtype=torch.int32),
            torch.tensor([[[3.0, 2.0, 0.0, torch.finfo(torch.float32).min]]]),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        seen["flash_topk"] = topk_idxs
        seen["flash_topk_length"] = topk_length
        return torch.zeros((1, 1, d_v), dtype=q.dtype), torch.zeros((1, 1), dtype=torch.float32)

    def fake_attn_target(q_attn, k_attn, lse, topk_indices, *args, **kwargs):
        seen["loss_topk"] = topk_indices
        return torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_attn_target)

    dsa_cudnn_kernels.fused_indexer_sparse_attn(
        torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=True,
        d_v=1,
    )

    assert seen["return_scores"] is False
    assert seen["return_topk_scores"] is True
    torch.testing.assert_close(seen["flash_topk"], torch.tensor([[1, 2, 3, -1]], dtype=torch.int32))
    torch.testing.assert_close(seen["flash_topk_length"], torch.tensor([3], dtype=torch.int32))
    torch.testing.assert_close(
        seen["loss_topk"], torch.tensor([[[3, 1, 2, -1]]], dtype=torch.int32)
    )
    expected = torch.softmax(torch.tensor([3.0, 2.0, 0.0]), dim=0)
    torch.testing.assert_close(seen["index_score"][0, 0, :3], expected)
    torch.testing.assert_close(seen["index_score"][0, 0, 3:], torch.zeros(125))


def test_cudnn_sparse_loss_masks_invalid_query_rows_for_backward(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            index_score,
            topk_indices,
            sm_scale,
            loss_coeff,
            grad_loss,
            block_I,
        ):
            seen["attn_score"] = attn_score
            seen["index_score"] = index_score
            seen["topk_indices"] = topk_indices
            seen["loss_coeff"] = loss_coeff
            seen["grad_loss"] = grad_loss
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    def fake_indexer_topk(*args, **kwargs):
        return (
            torch.tensor([[[3, 1, 2, -1], [2, 0, 1, -1]]], dtype=torch.int32),
            torch.tensor([[3, 3]], dtype=torch.int32),
            torch.tensor(
                [[[3.0, 2.0, 0.0, torch.finfo(torch.float32).min], [4.0, 1.0, 0.0, -1.0]]],
                dtype=torch.float32,
            ),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    def fake_attn_target(q_attn, k_attn, lse, topk_indices, *args, **kwargs):
        return torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.25, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_attn_target)

    dsa_cudnn_kernels.fused_indexer_sparse_attn(
        torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=False,
        d_v=1,
        query_valid_rows=torch.tensor([[True, False]], dtype=torch.bool),
    )

    torch.testing.assert_close(
        seen["topk_indices"][0, 0, :4], torch.tensor([3, 1, 2, 0], dtype=torch.int32)
    )
    torch.testing.assert_close(seen["topk_indices"][0, 1, :4], torch.zeros(4, dtype=torch.int32))
    torch.testing.assert_close(
        seen["topk_indices"][0, :, 4:], torch.zeros((2, 124), dtype=torch.int32)
    )
    torch.testing.assert_close(seen["attn_score"][0, 1], torch.zeros(128))
    torch.testing.assert_close(seen["index_score"][0, 1], torch.zeros(128))
    assert seen["loss_coeff"] == 0.01
    torch.testing.assert_close(seen["grad_loss"], torch.tensor(2.0))


def test_cudnn_sparse_backward_uses_batch_major_topk_indices_for_batched_kv(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            index_score,
            topk_indices,
            sm_scale,
            loss_coeff,
            grad_loss,
            block_I,
        ):
            seen["topk_indices"] = topk_indices
            seen["loss_coeff"] = loss_coeff
            seen["grad_loss"] = grad_loss
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    def fake_indexer_topk(*args, **kwargs):
        return (
            torch.tensor([[[3, 1, -1]], [[0, 2, -1]]], dtype=torch.int32),
            torch.tensor([[2], [2]], dtype=torch.int32),
            torch.tensor([[[3.0, 2.0, -1.0]], [[4.0, 1.0, -1.0]]], dtype=torch.float32),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    def fake_attn_target(q_attn, k_attn, lse, topk_indices, *args, **kwargs):
        return torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_attn_target)

    dsa_cudnn_kernels.fused_indexer_sparse_attn(
        torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 2, 1), dtype=torch.bfloat16),
        torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 2, 1), dtype=torch.bfloat16),
        torch.zeros((1, 2, 1), dtype=torch.bfloat16),
        indexer_topk=3,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=False,
        d_v=1,
    )

    torch.testing.assert_close(
        seen["topk_indices"][:, 0, :3], torch.tensor([[3, 1, 0], [4, 6, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        seen["topk_indices"][:, 0, 3:], torch.zeros((2, 125), dtype=torch.int32)
    )
    assert seen["loss_coeff"] == 0.01
    torch.testing.assert_close(seen["grad_loss"], torch.tensor(1.0))


def test_cudnn_attention_backward_sanitizes_ignored_topk_slots(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attention_backward_wrapper(
            q, kv, out, dO, lse, attn_sink, topk_indices, softmax_scale, topk_length
        ):
            seen["bwd_q_shape"] = q.shape
            seen["bwd_topk"] = topk_indices.detach().clone()
            seen["bwd_topk_length"] = topk_length.detach().clone()
            return {"dq": torch.ones_like(q), "dkv": torch.zeros_like(kv)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DSA", FakeDSA)

    def fake_indexer_topk(*args, **kwargs):
        return (
            torch.tensor([[[-1, -1, -1, -1], [2, -1, 1, -1]]], dtype=torch.int32),
            torch.tensor([[0, 2]], dtype=torch.int32),
            None,
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        seen["fwd_topk"] = topk_idxs.detach().clone()
        seen["fwd_topk_length"] = topk_length.detach().clone()
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)

    query = torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16, requires_grad=True)
    kv_full = torch.zeros((4, 1, 1), dtype=torch.bfloat16, requires_grad=True)
    output, _indexer_loss = dsa_cudnn_kernels.fused_indexer_sparse_attn(
        query,
        kv_full,
        torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=True,
        calculate_per_token_loss=False,
        d_v=1,
    )

    output.float().sum().backward()

    torch.testing.assert_close(
        seen["fwd_topk"], torch.tensor([[-1, -1, -1, -1], [1, 2, -1, -1]], dtype=torch.int32)
    )
    torch.testing.assert_close(seen["fwd_topk_length"], torch.tensor([0, 2], dtype=torch.int32))
    assert seen["bwd_q_shape"] == (1, 1, 1)
    torch.testing.assert_close(seen["bwd_topk"], torch.tensor([[1, 2, 0, 0]], dtype=torch.int32))
    torch.testing.assert_close(seen["bwd_topk_length"], torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(query.grad[0], torch.zeros_like(query.grad[0]))
    torch.testing.assert_close(query.grad[1], torch.ones_like(query.grad[1]))


def test_cudnn_attention_backward_pads_small_local_head_count():
    _skip_if_fused_dsa_unavailable()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    sq = 64
    batch_size = 1
    num_heads = 8
    attn_dim = 576
    latent_v_channels = 512
    indexer_heads = 64
    indexer_dim = 128

    query = torch.randn(
        (sq, batch_size, num_heads, attn_dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    kv_full = torch.randn(
        (sq, batch_size, attn_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    q_indexer = torch.randn(
        (sq, batch_size, indexer_heads, indexer_dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k_indexer = torch.randn(
        (sq, batch_size, indexer_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weights = torch.randn(
        (sq, batch_size, indexer_heads), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    output, indexer_loss = dsa_cudnn_kernels.fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk=2048,
        softmax_scale=attn_dim**-0.5,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=latent_v_channels,
    )

    assert torch.isfinite(output).all()
    assert indexer_loss.item() == 0.0
    output.float().mul(torch.randn_like(output).float()).sum().backward()

    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(kv_full.grad).all()
    torch.testing.assert_close(q_indexer.grad, torch.zeros_like(q_indexer.grad))
    torch.testing.assert_close(k_indexer.grad, torch.zeros_like(k_indexer.grad))
    torch.testing.assert_close(weights.grad, torch.zeros_like(weights.grad))


def test_cudnn_indexer_backward_head_padding_slices_to_actual_heads():
    q = torch.randn(1, 2, 32, 4)
    w = torch.randn(1, 2, 32)

    padded_q, padded_w, actual_heads = dsa_cudnn_kernels._pad_indexer_heads_for_backward(q, w)

    assert actual_heads == 32
    assert padded_q.shape == (1, 2, 64, 4)
    assert padded_w.shape == (1, 2, 64)
    torch.testing.assert_close(padded_q[:, :, :32], q)
    torch.testing.assert_close(padded_w[:, :, :32], w)
    torch.testing.assert_close(padded_q[:, :, 32:], torch.zeros(1, 2, 32, 4))
    torch.testing.assert_close(padded_w[:, :, 32:], torch.zeros(1, 2, 32))

    grad_q, grad_w = dsa_cudnn_kernels._slice_indexer_backward_head_grads(
        padded_q, padded_w, actual_heads
    )
    torch.testing.assert_close(grad_q, q)
    torch.testing.assert_close(grad_w, w)


def test_cudnn_sparse_backward_topk_padding_aligns_to_block_size():
    attn_score = torch.ones(1, 2, 3)
    index_score = torch.full((1, 2, 3), 2.0)
    topk_indices = torch.tensor([[[0, 1, 2], [2, 1, -1]]], dtype=torch.int32)

    padded_attn, padded_index, padded_topk = dsa_cudnn_kernels._pad_sparse_backward_topk(
        attn_score, index_score, topk_indices, block_size=4
    )

    assert padded_attn.shape == (1, 2, 4)
    assert padded_index.shape == (1, 2, 4)
    assert padded_topk.shape == (1, 2, 4)
    torch.testing.assert_close(padded_attn[..., :3], attn_score)
    torch.testing.assert_close(padded_index[..., :3], index_score)
    torch.testing.assert_close(
        padded_topk[..., :3], torch.tensor([[[0, 1, 2], [2, 1, 0]]], dtype=torch.int32)
    )
    torch.testing.assert_close(padded_attn[..., 3], torch.zeros(1, 2))
    torch.testing.assert_close(padded_index[..., 3], torch.zeros(1, 2))
    torch.testing.assert_close(padded_topk[..., 3], torch.zeros((1, 2), dtype=torch.int32))


def test_cudnn_full_fusion_accepts_varlen_when_indexer_loss_disabled(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 4
        calculate_per_token_loss = True

    seen = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        softmax_scale,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=0,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        use_local_indexer_varlen=False,
    ):
        seen["sparse_loss"] = sparse_loss
        seen["loss_coeff"] = loss_coeff
        seen["varlen_starts"] = varlen_starts
        seen["use_local_indexer_varlen"] = use_local_indexer_varlen
        return torch.zeros(
            (query.size(0), query.size(1), query.size(2) * d_v),
            dtype=query.dtype,
            device=query.device,
        ), torch.zeros((), dtype=torch.float32, device=query.device)

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    sq = 2
    sk = 4
    batch = 1
    heads = 2
    hidden = 8
    idx_heads = 2
    idx_hidden = 4
    output = dsa_cudnn_kernels.run_fused_dsa_attention(
        config=Config(),
        query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
        key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
        value=None,
        up_v_weight=None,
        q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
        k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
        indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=True,
        absorbed_mla=True,
        cp_size=2,
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=object(),
        varlen_starts=torch.tensor([0, 2], dtype=torch.int64),
        varlen_ends=torch.tensor([1, 4], dtype=torch.int64),
        key_positions=torch.arange(sk, dtype=torch.int64),
        query_valid_rows=None,
        use_relu=True,
        use_local_indexer_varlen=True,
    )

    assert output is not None
    assert seen["sparse_loss"] is False
    assert seen["loss_coeff"] == 0.0
    assert seen["use_local_indexer_varlen"] is True
    torch.testing.assert_close(seen["varlen_starts"], torch.tensor([0, 2], dtype=torch.int64))


def test_cudnn_full_fusion_accepts_local_varlen_for_sparse_indexer_loss(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 4
        calculate_per_token_loss = True

    seen = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        softmax_scale,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=0,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        use_local_indexer_varlen=False,
    ):
        seen["sparse_loss"] = sparse_loss
        seen["loss_coeff"] = loss_coeff
        seen["use_local_indexer_varlen"] = use_local_indexer_varlen
        return torch.zeros(
            (query.size(0), query.size(1), query.size(2) * d_v),
            dtype=query.dtype,
            device=query.device,
        ), torch.ones((), dtype=torch.float32, device=query.device)

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    sq = 2
    sk = 4
    batch = 1
    heads = 2
    hidden = 8
    idx_heads = 2
    idx_hidden = 4
    output = dsa_cudnn_kernels.run_fused_dsa_attention(
        config=Config(),
        query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
        key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
        value=None,
        up_v_weight=None,
        q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
        k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
        indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=True,
        absorbed_mla=True,
        cp_size=2,
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=object(),
        varlen_starts=torch.tensor([0, 2], dtype=torch.int64),
        varlen_ends=torch.tensor([1, 4], dtype=torch.int64),
        key_positions=torch.arange(sk, dtype=torch.int64),
        query_valid_rows=None,
        use_relu=True,
        use_local_indexer_varlen=True,
    )

    assert output is not None
    assert seen["sparse_loss"] is True
    assert seen["loss_coeff"] == 0.01
    assert seen["use_local_indexer_varlen"] is True


def test_cudnn_full_fusion_rejects_varlen_dense_indexer_loss():
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 4
        calculate_per_token_loss = True

    sq = 2
    sk = 4
    batch = 1
    heads = 2
    hidden = 8
    idx_heads = 2
    idx_hidden = 4

    with pytest.raises(RuntimeError, match="packed-varlen dense indexer loss is unsupported"):
        dsa_cudnn_kernels.run_fused_dsa_attention(
            config=Config(),
            query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
            key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
            value=None,
            up_v_weight=None,
            q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
            k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
            indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
            indexer_topk=4,
            softmax_scale=1.0,
            loss_coeff=0.01,
            sparse_loss=False,
            calculate_per_token_loss=True,
            absorbed_mla=True,
            cp_size=2,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=object(),
            varlen_starts=torch.tensor([0, 2], dtype=torch.int64),
            varlen_ends=torch.tensor([1, 4], dtype=torch.int64),
            key_positions=torch.arange(sk, dtype=torch.int64),
            query_valid_rows=None,
            use_relu=True,
            use_local_indexer_varlen=True,
        )


@pytest.mark.parametrize("seqlen", [1024, 2048, 4096])
@pytest.mark.parametrize("attention_backend", [AttnBackend.unfused, AttnBackend.auto])
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize("use_sparse_loss", [False, True], ids=["dense_loss", "sparse_loss"])
def test_absorbed_mla_dsa(
    seqlen: int, attention_backend, calculate_per_token_loss: bool, use_sparse_loss: bool
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DSA parity")
    if attention_backend != AttnBackend.unfused:
        _skip_if_fused_dsa_unavailable()

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

        config = _make_config(
            use_sparse_loss=use_sparse_loss,
            calculate_per_token_loss=calculate_per_token_loss,
            dsa_kernel_backend="cudnn" if attention_backend != AttnBackend.unfused else "none",
        )
        object.__setattr__(config, "attention_backend", attention_backend)
        backend = TESpecProvider()
        spec = get_dsa_module_spec_for_backend(config=config, backend=backend)

        real_layer = build_module(
            spec, config=config, layer_number=1, cp_comm_type=None, pg_collection=None
        ).cuda()
        baseline = (
            NativeDSA(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_head_dim=config.qk_head_dim,
                qk_pos_emb_head_dim=config.qk_pos_emb_head_dim,
                v_head_dim=config.v_head_dim,
                dsa_indexer_n_heads=config.dsa_indexer_n_heads,
                dsa_indexer_head_dim=config.dsa_indexer_head_dim,
                dsa_indexer_topk=config.dsa_indexer_topk,
                dsa_indexer_use_sparse_loss=config.dsa_indexer_use_sparse_loss,
                layernorm_epsilon=config.layernorm_epsilon,
                rotary_base=config.rotary_base,
                rotary_scaling_factor=config.rotary_scaling_factor,
                original_max_position_embeddings=config.original_max_position_embeddings,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
                mscale=config.mscale,
                rope_factor=config.rotary_scaling_factor,
                calculate_per_token_loss=config.calculate_per_token_loss,
            )
            .bfloat16()
            .cuda()
        )

        name_mapping = {}
        for name, _ in baseline.named_parameters():
            name_mapping[name] = "core_attention." + name if "indexer" in name else name

        real_params = dict(real_layer.named_parameters())
        for baseline_name, baseline_param in baseline.named_parameters():
            real_param = real_params[name_mapping[baseline_name]]
            real_param.data.copy_(baseline_param.data)

        original_rotate_activation = dsa_module.rotate_activation
        dsa_module.rotate_activation = _mock_rotate_activation
        try:
            bsz = 1
            attention_mask = torch.triu(
                torch.full((bsz, 1, seqlen, seqlen), float("-inf"), device="cuda"), diagonal=1
            )
            for _ in range(10):
                hidden_states = torch.randn(
                    (seqlen, bsz, config.hidden_size),
                    dtype=torch.bfloat16,
                    device="cuda",
                    requires_grad=True,
                )
                hidden_states_baseline = hidden_states.detach().clone().requires_grad_(True)
                grad = torch.randn(
                    (seqlen, bsz, config.hidden_size), dtype=torch.bfloat16, device="cuda"
                )

                real_out, _ = real_layer(hidden_states, attention_mask=attention_mask)
                baseline_out, kl_loss = baseline(
                    hidden_states_baseline, attention_mask=attention_mask
                )

                _assert_similarity(real_out.detach(), baseline_out.detach())

                real_out.backward(grad)
                baseline_out.backward(grad)
                loss_coeff = torch.tensor(
                    config.dsa_indexer_loss_coeff, device=kl_loss.device, dtype=kl_loss.dtype
                )
                kl_loss.backward(loss_coeff)

                _assert_similarity(hidden_states.grad, hidden_states_baseline.grad)
        finally:
            dsa_module.rotate_activation = original_rotate_activation

        is_fused_dense = attention_backend != AttnBackend.unfused and not use_sparse_loss
        for baseline_name, baseline_param in baseline.named_parameters():
            real_param = real_params[name_mapping[baseline_name]]
            assert baseline_param.grad is not None
            assert real_param.grad is not None
            eps = (
                _FUSED_DENSE_INDEXER_GRAD_SIMILARITY_EPS
                if is_fused_dense and baseline_name.startswith("indexer.")
                else _SIMILARITY_EPS
            )
            _assert_similarity(real_param.grad, baseline_param.grad, eps=eps, label=baseline_name)
    finally:
        Utils.destroy_model_parallel()
