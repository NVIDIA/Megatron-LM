# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils


def _mock_rotate_activation(x: torch.Tensor) -> torch.Tensor:
    return x


def _make_config(use_sparse_loss: bool) -> MLATransformerConfig:
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


class NaiveIndexer(nn.Module):

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        logits = torch.einsum("bthd,bsd->bths", q, k) * self.softmax_scale
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=2)
        logits = logits + attention_mask.squeeze(1)
        topk = min(self.index_topk, x.size(1))
        if not self.use_sparse_loss:
            topk = x.size(1)
        topk_logits, topk_indices = logits.topk(topk, dim=-1)
        log_topk_prob = F.log_softmax(topk_logits, dim=-1)
        return topk_indices, log_topk_prob


class NaiveDSA(nn.Module):

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
        self.linear_k_up_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * self.qk_head_dim, bias=False
        )
        self.linear_v_up_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False
        )
        self.linear_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        self.indexer = NaiveIndexer(
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

        k_up_weight = self.linear_k_up_proj.weight.view(
            self.num_heads, self.qk_head_dim, self.kv_lora_rank
        )
        q_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, k_up_weight)
        q_absorbed = torch.cat([q_absorbed, q_pe], dim=-1)

        topk_indices, log_topk_prob = self.indexer(
            x.detach(), qr.detach(), freqs_cis, attention_mask
        )

        keys = torch.cat([kv_compressed, k_pe.squeeze(2)], dim=-1)
        values = kv_compressed
        scores = torch.einsum("bshd,btd->bsht", q_absorbed, keys) * self.softmax_scale
        scores = scores + attention_mask.squeeze(1).unsqueeze(2)

        index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device)
        index_mask.scatter_(-1, topk_indices, 0)
        scores = scores + index_mask.unsqueeze(2)

        probs = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("bsht,btd->bshd", probs.to(values.dtype), values)

        gather_index = topk_indices.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        sparse_probs_topk = torch.gather(probs.detach(), dim=-1, index=gather_index)
        attn_score = sparse_probs_topk.sum(dim=2)
        attn_score = attn_score / attn_score.sum(dim=-1, keepdim=True)

        indexer_kl_loss = F.kl_div(
            log_topk_prob.clip(-100, 0),
            attn_score.log().clip(-100, 0),
            log_target=True,
            reduction="sum",
        )

        v_up_weight = self.linear_v_up_proj.weight.view(
            self.num_heads, self.v_head_dim, self.kv_lora_rank
        )
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


def _assert_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3):
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    c = _cosine_sim(a, b)
    t = _tensor_sim(a, b)
    assert c > 1 - eps, f"cosine_sim={c:.6f}"
    assert t > 1 - eps, f"tensor_sim={t:.6f}"


@pytest.mark.parametrize("seqlen", [1024, 2048, 4096])
@pytest.mark.parametrize("use_sparse_loss", [True, False])
def test_absorbed_mla_dsa(seqlen: int, use_sparse_loss: bool):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

        config = _make_config(use_sparse_loss=use_sparse_loss)
        backend = TESpecProvider()
        spec = get_dsa_module_spec_for_backend(config=config, backend=backend)

        real_layer = build_module(
            spec, config=config, layer_number=1, cp_comm_type=None, pg_collection=None
        ).cuda()
        baseline = (
            NaiveDSA(
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

        for baseline_name, baseline_param in baseline.named_parameters():
            real_param = real_params[name_mapping[baseline_name]]
            assert baseline_param.grad is not None
            assert real_param.grad is not None
            _assert_similarity(real_param.grad, baseline_param.grad)
    finally:
        Utils.destroy_model_parallel()
