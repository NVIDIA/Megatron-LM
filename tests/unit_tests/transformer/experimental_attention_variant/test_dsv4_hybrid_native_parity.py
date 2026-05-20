# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsv4_hybrid_module_spec_for_backend,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import compute_dsa_indexer_loss
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
_FUSED_SIMILARITY_EPS = 1.5e-4
_UNFUSED_SIMILARITY_EPS = 2.3e-5


@torch.compile
def _native_q_rms_norm(query: torch.Tensor, eps: float) -> torch.Tensor:
    return query * torch.rsqrt(query.square().mean(-1, keepdim=True) + eps)

_DSV4_VARIANTS = {
    "flash": {
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "q_lora_rank": 1024,
        "v_head_dim": 512,
        "qk_pos_emb_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "csa_compress_rotary_base": 40000,
        "dsa_indexer_topk": 512,
    },
    "pro": {
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "q_lora_rank": 1536,
        "v_head_dim": 512,
        "qk_pos_emb_head_dim": 64,
        "o_groups": 16,
        "o_lora_rank": 1024,
        "csa_compress_rotary_base": 160000,
        "dsa_indexer_topk": 1024,
    },
}

_DSA_BACKENDS = [
    pytest.param("fused", False, id="fused"),
    pytest.param("unfused", True, id="unfused"),
]

_CASE_SEQLENS = [2048, 4096, 8192]


def _make_config(
    variant: str, compress_ratio: int, force_unfused_dsa: bool = False
) -> MLATransformerConfig:
    shape = _DSV4_VARIANTS[variant]
    mcore_ratio = 0 if compress_ratio == 1 else compress_ratio
    qk_head_dim = shape["v_head_dim"] - shape["qk_pos_emb_head_dim"]
    config = MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        num_layers=1,
        hidden_size=shape["hidden_size"],
        num_attention_heads=shape["num_attention_heads"],
        q_lora_rank=shape["q_lora_rank"],
        kv_lora_rank=qk_head_dim,
        qk_head_dim=qk_head_dim,
        qk_pos_emb_head_dim=shape["qk_pos_emb_head_dim"],
        v_head_dim=shape["v_head_dim"],
        o_groups=shape["o_groups"],
        o_lora_rank=shape["o_lora_rank"],
        csa_compress_ratios=[mcore_ratio],
        csa_window_size=128,
        csa_dense_mode=False,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=shape["dsa_indexer_topk"],
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
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
        rotary_base=10000,
        rotary_percent=1.0,
        csa_compress_rotary_base=shape["csa_compress_rotary_base"],
        recompute_granularity=None,
        recompute_modules=[],
        fine_grained_activation_offloading=False,
        gradient_accumulation_fusion=False,
        fp8=False,
        fp4=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        kv_channels=shape["v_head_dim"],
        num_query_groups=shape["num_attention_heads"],
        batch_invariant_mode=False,
        cache_mla_latents=False,
        use_cpu_initialization=True,
        perform_initialization=True,
        symmetric_ar_type=None,
        disable_parameter_transpose_cache=False,
        init_model_with_meta_device=False,
        delay_wgrad_compute=False,
        tp_comm_overlap=False,
        softmax_scale=None,
    )
    config.force_unfused_dsa = force_unfused_dsa
    return config


def _precompute_freqs_cis(dim: int, seqlen: int, device, base: float) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    return torch.cat((freqs, freqs), dim=-1)[:, None, None, :]


def _apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    if x.numel() == 0:
        return x
    freqs = freqs_cis.to(x.device)
    if freqs.dim() == x.dim() + 1 and freqs.size(-2) == 1:
        freqs = freqs.squeeze(-2)

    rot_dim = freqs.size(-1)
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    x_rot = torch.cat((x1, x2), dim=-1)

    cos = torch.cos(freqs).to(x_rot.dtype)
    sin = torch.sin(freqs).to(x_rot.dtype)
    if inverse:
        sin = -sin

    rot_half_1, rot_half_2 = torch.chunk(x_rot, 2, dim=-1)
    x_rotated = torch.cat((-rot_half_2, rot_half_1), dim=-1)
    out = (x_rot * cos) + (x_rotated * sin)

    x1, x2 = torch.chunk(out, 2, dim=-1)
    out = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)
    return torch.cat((out, x_pass), dim=-1)


def _native_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    n = x.size(-1)
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard transform requires power-of-two last dim, got {n}")
    dtype = x.dtype
    y = x.float()
    shape = y.shape
    h = 1
    while h < n:
        y = y.reshape(*shape[:-1], -1, 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.cat((a + b, a - b), dim=-1)
        h *= 2
    return (y.reshape(shape) * (n**-0.5)).to(dtype)


def _get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    offsets = torch.arange(window_size, device=device)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def _get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def _native_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    sq, batch_size, num_heads, head_dim = query.size()
    kv_t = kv_full.permute(1, 0, 2)
    safe_indices = topk_indices.clamp(min=0).long()
    gather_index = safe_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_gathered = torch.gather(
        kv_t.unsqueeze(1).expand(-1, sq, -1, -1), dim=2, index=gather_index
    )

    q = query.permute(1, 2, 0, 3).float()
    scores = torch.einsum("bnsh,bskh->bnsk", q, kv_gathered.float()) * softmax_scale
    scores = scores.masked_fill((topk_indices < 0).unsqueeze(1), float("-inf"))

    sink = attn_sink.view(1, num_heads, 1, 1).float()
    scores_max = torch.max(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max)
    exp_sink = torch.exp(sink - scores_max)
    attn_weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink)

    output = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_gathered.float())
    output = output.to(query.dtype).permute(2, 0, 1, 3).contiguous()
    return output.reshape(sq, batch_size, num_heads * head_dim)


def _native_fused_sparse_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
) -> torch.Tensor:
    batch_size, seqlen, topk = topk_indices.size()
    num_heads, head_dim = query.size(2), query.size(3)
    safe_indices = topk_indices.clamp(min=0).long()
    valid = topk_indices >= 0
    row_valid = valid.any(dim=-1, keepdim=True)

    predict_logits = torch.gather(index_scores, dim=-1, index=safe_indices)
    predict_logits = predict_logits.masked_fill(~valid, float("-inf"))
    predict_logits = predict_logits.masked_fill(~row_valid, 0.0)
    predict = F.softmax(predict_logits, dim=-1, dtype=torch.float32)
    predict = predict * row_valid.float()

    compressed_kv_t = compressed_kv.detach().permute(1, 0, 2)
    selected_kv = torch.gather(
        compressed_kv_t.unsqueeze(1).expand(-1, seqlen, -1, -1),
        dim=2,
        index=safe_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim),
    )
    q = query.detach().permute(1, 2, 0, 3).float()
    attn_scores = torch.einsum("bhsd,bskd->bhsk", q, selected_kv.float())
    attn_scores = attn_scores * softmax_scale
    attn_scores = attn_scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    sink = attn_sink.detach().view(1, num_heads, 1, 1).float()
    score_max = torch.max(attn_scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(attn_scores - score_max)
    exp_sink = torch.exp(sink - score_max)
    attn_probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink)
    target = attn_probs.sum(dim=1)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    target = target * row_valid.float()

    eps = torch.finfo(torch.float32).tiny
    target = target.clamp(min=eps)
    predict = predict.clamp(min=eps)
    kl_per_row = (target * (torch.log(target) - torch.log(predict))).sum(dim=-1)
    kl_per_row = torch.where(row_valid.squeeze(-1), kl_per_row, torch.zeros_like(kl_per_row))
    return loss_coeff * kl_per_row.mean()


class NativeCompressor(nn.Module):
    def __init__(self, config: MLATransformerConfig, compress_ratio: int, head_dim: int, rotate: bool):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.rope_base = config.csa_compress_rotary_base if compress_ratio > 1 else config.rotary_base

        self.linear_wkv = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.linear_wgate = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * head_dim, dtype=torch.float32))
        self.norm = nn.RMSNorm(head_dim, eps=config.layernorm_epsilon)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        n_groups, ratio, batch_size, _ = tensor.size()
        new_tensor = tensor.new_full((n_groups, 2 * ratio, batch_size, self.head_dim), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, self.head_dim :]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, : self.head_dim]
        return new_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        sq, batch_size, _ = x.size()
        ratio = self.compress_ratio
        if sq < ratio:
            return None

        kv = self.linear_wkv(x)
        score = self.linear_wgate(x)

        cutoff = (sq // ratio) * ratio
        kv = kv[:cutoff]
        score = score[:cutoff]
        n_compressed = cutoff // ratio

        kv = kv.view(n_compressed, ratio, batch_size, -1)
        score = score.view(n_compressed, ratio, batch_size, -1)
        score = score + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        kv = (kv * torch.softmax(score, dim=1)).sum(dim=1)
        kv = self.norm(kv.to(x.dtype))

        pos_dim = self.qk_pos_emb_head_dim
        content, rotary = torch.split(kv, [self.head_dim - pos_dim, pos_dim], dim=-1)
        freqs_cis = _precompute_freqs_cis(
            pos_dim, n_compressed * ratio, device=x.device, base=self.rope_base
        )
        freqs_cis = freqs_cis[: n_compressed * ratio : ratio][:n_compressed]
        rotary = _apply_rotary_emb(rotary, freqs_cis)
        kv = torch.cat([content, rotary], dim=-1)

        if self.rotate:
            kv = _native_hadamard_transform(kv)
        return kv


class NativeCSAIndexer(nn.Module):
    def __init__(self, config: MLATransformerConfig, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.softmax_scale = self.index_head_dim**-0.5
        self.force_unfused_dsa = config.force_unfused_dsa
        self.rope_base = config.csa_compress_rotary_base

        self.linear_wq_b = nn.Linear(
            config.q_lora_rank, self.index_n_heads * self.index_head_dim, bias=False
        )
        self.linear_weights_proj = nn.Linear(
            config.hidden_size, self.index_n_heads, bias=False
        )
        self.compressor = NativeCompressor(
            config=config, compress_ratio=compress_ratio, head_dim=self.index_head_dim, rotate=True
        )

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sq, batch_size, _ = x.size()
        q = self.linear_wq_b(qr).view(sq, batch_size, self.index_n_heads, self.index_head_dim)
        pos_dim = self.qk_pos_emb_head_dim
        q_content, q_rotary = torch.split(q, [self.index_head_dim - pos_dim, pos_dim], dim=-1)
        freqs_cis = _precompute_freqs_cis(pos_dim, sq, device=x.device, base=self.rope_base)
        q_rotary = _apply_rotary_emb(q_rotary, freqs_cis)
        q = _native_hadamard_transform(torch.cat([q_content, q_rotary], dim=-1))

        k = self.compressor(x)
        weights = self.linear_weights_proj(x) * (self.index_n_heads**-0.5)
        return q, k, weights

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, weights = self.forward_before_topk(x, qr)
        weights_scaled = weights.float() * self.softmax_scale
        if not self.force_unfused_dsa:
            weights_scaled = weights_scaled.to(weights.dtype).float()
        scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
        scores = torch.relu(scores) * weights_scaled.unsqueeze(-1)
        scores = scores.sum(dim=2).transpose(0, 1)

        sq = x.size(0)
        n_compressed = k.size(0)
        valid_per_query = (
            torch.arange(1, sq + 1, device=x.device).unsqueeze(0) // self.compress_ratio
        ).clamp(max=n_compressed)
        invalid = torch.arange(n_compressed, device=x.device).view(1, 1, -1) >= valid_per_query.unsqueeze(-1)
        scores = scores.masked_fill(invalid.expand_as(scores), float("-inf"))

        topk = min(self.index_topk, n_compressed)
        topk_scores, topk_indices = scores.topk(topk, dim=-1)
        topk_indices = torch.where(topk_scores.isneginf(), -1, topk_indices)
        return q, k, weights, scores, topk_indices


class NativeCompressedSparseAttention(nn.Module):
    def __init__(self, config: MLATransformerConfig, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.v_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.indexer_loss_coeff = config.dsa_indexer_loss_coeff
        self.indexer_use_sparse_loss = config.dsa_indexer_use_sparse_loss
        self.force_unfused_dsa = config.force_unfused_dsa

        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.compressor = (
            NativeCompressor(config=config, compress_ratio=compress_ratio, head_dim=self.head_dim, rotate=False)
            if compress_ratio > 1
            else None
        )
        self.indexer = NativeCSAIndexer(config, compress_ratio) if compress_ratio == 4 else None

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, x: torch.Tensor, qr: torch.Tensor, pg_collection
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sq, batch_size, _, _ = query.size()
        kv = key.squeeze(-2)
        n_compressed = 0

        if self.compressor is not None:
            compressed_kv = self.compressor(x)
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
        else:
            compressed_kv = None
            kv_full = kv

        window_idxs = _get_window_topk_idxs(self.window_size, batch_size, sq, query.device)
        indexer_loss = None
        if self.compress_ratio > 1 and n_compressed > 0:
            offset = sq
            if self.indexer is not None:
                q_idx, k_idx, weights_idx, index_scores, topk_compressed = self.indexer(x.detach(), qr.detach())
                topk_compressed_for_attn = torch.where(topk_compressed >= 0, topk_compressed + offset, -1)

                if self.force_unfused_dsa:
                    key_for_loss = compressed_kv.unsqueeze(2).expand(
                        -1, -1, self.num_heads, -1
                    )
                    causal_mask = (
                        torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                    )
                    positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                    causal_mask = (
                        torch.where(
                            causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0
                        )
                        .unsqueeze(0)
                        .expand(batch_size, -1, -1)
                    )
                    indexer_loss = compute_dsa_indexer_loss(
                        index_scores,
                        topk_compressed.clamp(min=0),
                        query.detach(),
                        key_for_loss.detach(),
                        self.softmax_scale,
                        self.indexer_loss_coeff,
                        self.indexer_use_sparse_loss,
                        pg_collection,
                        causal_mask_override=causal_mask,
                    )
                else:
                    indexer_loss = _native_fused_sparse_indexer_loss(
                        index_scores,
                        topk_compressed,
                        query,
                        compressed_kv,
                        self.attn_sink,
                        self.softmax_scale,
                        self.indexer_loss_coeff,
                    )
            else:
                topk_compressed_for_attn = _get_compress_topk_idxs(
                    self.compress_ratio, batch_size, sq, offset, query.device
                )
            if self.indexer is not None and not self.force_unfused_dsa:
                topk_idxs = torch.cat([topk_compressed_for_attn, window_idxs], dim=-1)
            else:
                topk_idxs = torch.cat([window_idxs, topk_compressed_for_attn], dim=-1)
        else:
            topk_idxs = window_idxs

        output = _native_sparse_attn(query, kv_full, self.attn_sink, topk_idxs, self.softmax_scale)
        return output, indexer_loss


class NativeDSv4HybridAttention(nn.Module):
    def __init__(self, config: MLATransformerConfig, compress_ratio: int):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.num_heads = config.num_attention_heads
        self.head_dim = config.v_head_dim
        self.pos_dim = config.qk_pos_emb_head_dim
        self.nope_dim = config.v_head_dim - config.qk_pos_emb_head_dim
        self.rope_base = config.csa_compress_rotary_base if compress_ratio > 1 else config.rotary_base

        self.linear_q_down_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.layernorm_epsilon)
        self.linear_q_up_proj = nn.Linear(
            config.q_lora_rank, config.num_attention_heads * config.v_head_dim, bias=False
        )
        self.linear_kv_proj = nn.Linear(config.hidden_size, config.v_head_dim, bias=False)
        self.kv_layernorm = nn.RMSNorm(config.v_head_dim, eps=config.layernorm_epsilon)
        self.core_attention = NativeCompressedSparseAttention(config, compress_ratio)
        group_in = (config.num_attention_heads * config.v_head_dim) // config.o_groups
        self.linear_o_group_proj = nn.Parameter(
            torch.empty(config.o_groups * config.o_lora_rank, group_in)
        )
        self.linear_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, pg_collection) -> tuple[torch.Tensor, torch.Tensor | None]:
        sq, batch_size, _ = hidden_states.size()
        freqs_cis = _precompute_freqs_cis(self.pos_dim, sq, hidden_states.device, self.rope_base)

        qr = self.q_layernorm(self.linear_q_down_proj(hidden_states))
        query = self.linear_q_up_proj(qr).view(sq, batch_size, self.num_heads, self.head_dim)
        query = _native_q_rms_norm(query, self.config.layernorm_epsilon)
        q_content, q_rotary = torch.split(query, [self.nope_dim, self.pos_dim], dim=-1)
        query = torch.cat([q_content, _apply_rotary_emb(q_rotary, freqs_cis)], dim=-1)

        key = self.kv_layernorm(self.linear_kv_proj(hidden_states))
        k_content, k_rotary = torch.split(key, [self.nope_dim, self.pos_dim], dim=-1)
        key = torch.cat([k_content, _apply_rotary_emb(k_rotary, freqs_cis)], dim=-1)
        key = key.unsqueeze(-2)

        core_out, indexer_loss = self.core_attention(
            query=query, key=key, x=hidden_states, qr=qr, pg_collection=pg_collection
        )

        core_out = core_out.view(sq, batch_size, self.num_heads, self.head_dim)
        out_content, out_rotary = torch.split(core_out, [self.nope_dim, self.pos_dim], dim=-1)
        core_out = torch.cat([out_content, _apply_rotary_emb(out_rotary, freqs_cis, inverse=True)], dim=-1)
        core_out = core_out.view(sq, batch_size, -1)

        core_out = core_out.view(sq, batch_size, self.config.o_groups, -1)
        wo_a = self.linear_o_group_proj.view(self.config.o_groups, self.config.o_lora_rank, -1)
        core_out = torch.einsum("...gd,grd->...gr", core_out, wo_a)
        core_out = core_out.reshape(sq, batch_size, -1)
        return self.linear_proj(core_out), indexer_loss


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_similarity(
    a: torch.Tensor, b: torch.Tensor, label: str, eps: float
):
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    cosine_sim = _cosine_sim(a, b)
    tensor_sim = _tensor_sim(a, b)
    assert cosine_sim > 1 - eps, f"{label}: cosine_sim={cosine_sim:.10f}, eps={eps}"
    assert tensor_sim > 1 - eps, f"{label}: tensor_sim={tensor_sim:.10f}, eps={eps}"


def _copy_real_params_to_native(real_layer: nn.Module, native_layer: nn.Module):
    real_params = dict(real_layer.named_parameters())
    for name, native_param in native_layer.named_parameters():
        assert name in real_params, f"Missing real parameter for native parameter {name}"
        real_param = real_params[name]
        assert native_param.shape == real_param.shape, (
            f"Shape mismatch for {name}: native={native_param.shape}, real={real_param.shape}"
        )
        native_param.data = real_param.data.to(
            device=native_param.device, dtype=real_param.dtype
        ).clone()
    return real_params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
@pytest.mark.parametrize(("backend", "force_unfused_dsa"), _DSA_BACKENDS)
@pytest.mark.parametrize("variant", ["flash", "pro"])
@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
@pytest.mark.parametrize("seqlen", _CASE_SEQLENS)
def test_dsv4_hybrid_attention_matches_native_reference(
    variant: str,
    compress_ratio: int,
    seqlen: int,
    backend: str,
    force_unfused_dsa: bool,
):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config(
            variant, compress_ratio, force_unfused_dsa=force_unfused_dsa
        )
        similarity_eps = (
            _UNFUSED_SIMILARITY_EPS if force_unfused_dsa else _FUSED_SIMILARITY_EPS
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        spec = get_dsv4_hybrid_module_spec_for_backend(config=config, backend=TESpecProvider())

        mcore_ratio = 0 if compress_ratio == 1 else compress_ratio
        real_layer = build_module(
            spec, config=config, layer_number=1, cp_comm_type=None, pg_collection=pg_collection
        ).cuda()
        native_layer = NativeDSv4HybridAttention(config, mcore_ratio).cuda()
        real_params = _copy_real_params_to_native(real_layer, native_layer)

        bsz = 1
        for _ in range(1):
            hidden_states = torch.randn(
                seqlen,
                bsz,
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            hidden_states_native = hidden_states.detach().clone().requires_grad_(True)
            grad = torch.randn_like(hidden_states)

            real_out, _ = real_layer(hidden_states=hidden_states, attention_mask=None)
            native_out, native_indexer_loss = native_layer(hidden_states_native, pg_collection)

            _assert_similarity(
                real_out.detach(),
                native_out.detach(),
                f"{backend}-{variant}-{compress_ratio}-{seqlen}:out",
                eps=similarity_eps,
            )

            real_out.backward(grad)
            native_out.backward(grad)
            if native_indexer_loss is not None:
                native_indexer_loss.backward()

            _assert_similarity(
                hidden_states.grad,
                hidden_states_native.grad,
                f"{backend}-{variant}-{compress_ratio}-{seqlen}:hidden_grad",
                eps=similarity_eps,
            )

        for name, native_param in native_layer.named_parameters():
            real_param = real_params[name]
            if compress_ratio != 4 and ".indexer." in name:
                continue
            assert native_param.grad is not None, f"Missing native grad for {name}"
            assert real_param.grad is not None, f"Missing real grad for {name}"
            _assert_similarity(
                real_param.grad,
                native_param.grad,
                f"{backend}-{variant}-{compress_ratio}-{seqlen}:param_grad:{name}",
                eps=similarity_eps,
            )
        del real_layer, native_layer, real_params
        del hidden_states, hidden_states_native, grad, real_out, native_out, native_indexer_loss
    finally:
        Utils.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
