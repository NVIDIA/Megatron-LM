# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]


def _make_dsa():
    pytest.importorskip("cudnn", reason="GLM5 DSA accept-with-proof needs cudnn DSA.")
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention

    return DynamicSparseAttention(
        hidden_size=128,
        num_attention_heads=64,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        rms_norm_eps=1e-5,
        layer_number=1,
    )


def _run_once(module, x, cos, sin, position_ids, *, fused_training: bool):
    module.zero_grad(set_to_none=True)
    module.train(fused_training)
    local_x = x.detach().clone().requires_grad_(True)
    out = module(local_x, cos=cos, sin=sin, position_ids=position_ids)
    loss = out.float().square().mean()
    loss.backward()
    param_grads = {
        name: param.grad.detach().float().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": loss.detach().float().clone(),
        "out": out.detach().float().clone(),
        "x_grad": local_x.grad.detach().float().clone(),
        "param_grads": param_grads,
    }


def _causal_mask(seq_q: int, seq_k: int, *, ratio: int, device: torch.device) -> torch.Tensor:
    q_idx = torch.arange(seq_q, device=device)
    k_idx = torch.arange(seq_k, device=device)
    valid_per_q = ((q_idx + 1) // ratio).clamp(max=seq_k)
    return k_idx.unsqueeze(0) < valid_per_q.unsqueeze(1)


def _torch_indexer_scores(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    *,
    ratio: int,
    indexer_softmax_scale: float,
) -> torch.Tensor:
    q_bshd = q_indexer.permute(1, 0, 2, 3).float()
    k_bsd = k_indexer.permute(1, 0, 2).float()
    w_bsh = weights.permute(1, 0, 2).float()
    scores = torch.einsum("bqhd,bkd->bqhk", q_bshd, k_bsd)
    scores = torch.relu(scores).mul(w_bsh.unsqueeze(-1)).sum(dim=2)
    scores = scores * float(indexer_softmax_scale)
    causal = _causal_mask(scores.shape[1], scores.shape[2], ratio=ratio, device=scores.device)
    return torch.where(causal.unsqueeze(0), scores, torch.full_like(scores, -torch.inf))


def _torch_topk_from_scores(scores: torch.Tensor, topk: int) -> torch.Tensor:
    effective_topk = min(topk, scores.shape[-1])
    values, indices = torch.topk(scores, k=effective_topk, dim=-1)
    indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    if effective_topk < topk:
        pad = torch.full(
            (*indices.shape[:-1], topk - effective_topk),
            -1,
            device=indices.device,
            dtype=indices.dtype,
        )
        indices = torch.cat([indices, pad], dim=-1)
    return indices.int()


def _gather_sequence(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe_indices = indices.clamp(min=0).long()
    expanded = source.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
    gathered = torch.gather(
        expanded,
        dim=2,
        index=safe_indices.unsqueeze(-1).expand(-1, -1, -1, source.shape[-1]),
    )
    return torch.where(indices.unsqueeze(-1) >= 0, gathered, torch.zeros_like(gathered))


def _torch_sparse_attention(
    query_states: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    value_dim: int,
) -> torch.Tensor:
    query_bshd = query_states.permute(1, 0, 2, 3).float()
    kv_bsd = kv_full.permute(1, 0, 2).float()
    selected_keys = _gather_sequence(kv_bsd, topk_indices)
    selected_values = _gather_sequence(kv_bsd[..., :value_dim], topk_indices)
    scores = torch.einsum("bqhd,bqtd->bqht", query_bshd, selected_keys)
    scores = scores * float(softmax_scale)
    scores = torch.where(
        topk_indices.unsqueeze(2) >= 0, scores, torch.full_like(scores, -torch.inf)
    )
    sink = attn_sink.float().view(1, 1, -1, 1).expand(scores.shape[0], scores.shape[1], -1, -1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("bqht,bqtr->bqhr", probs, selected_values)
    return out.permute(1, 0, 2, 3).reshape(
        query_states.shape[0], query_states.shape[1], query_states.shape[2] * value_dim
    )


def _torch_unfused_dsa_forward(module, x, cos, sin, position_ids):
    from megatron.lite.primitive.modules.attention.dsa import (
        _rotary_embeddings_from_cache,
        apply_rotary_pos_emb,
    )

    batch, seq_len, _ = x.shape
    q_resid = module.q_a_layernorm(module.q_a_proj(x))
    q = module.q_b_proj(q_resid).view(batch, seq_len, module.num_heads, module.qk_head_dim)
    q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    cos, sin = _rotary_embeddings_from_cache(
        cos, sin, position_ids, device=x.device, dtype=x.dtype, dim=module.qk_rope_head_dim
    )
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)

    k_up_weight, v_up_weight = module._split_kv_b_weights()
    q_nope = torch.einsum("bshd,hdr->bshr", q_nope, k_up_weight)
    query_states = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1).contiguous()

    kv_latent, k_pe = torch.split(
        module.kv_a_proj_with_mqa(x), [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1
    )
    kv_latent = module.kv_a_layernorm(kv_latent)
    k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
    kv_full = torch.cat([kv_latent, k_pe], dim=-1).transpose(0, 1).contiguous()

    assert module.indexer is not None
    q_indexer, k_indexer, weights_indexer = module.indexer.forward_before_topk(
        x, q_resid, cos, sin, position_ids
    )
    indexer_scores = _torch_indexer_scores(
        q_indexer,
        k_indexer,
        weights_indexer,
        ratio=1,
        indexer_softmax_scale=module.indexer_softmax_scale,
    )
    topk_indices = _torch_topk_from_scores(
        indexer_scores, min(module.index_topk, indexer_scores.shape[-1])
    )
    out = _torch_sparse_attention(
        query_states,
        kv_full,
        module.attn_sink,
        topk_indices,
        softmax_scale=module.softmax_scale,
        value_dim=module.kv_lora_rank,
    )
    out = out.to(x.dtype).view(seq_len, batch, module.num_heads, module.kv_lora_rank)
    out = out.permute(1, 0, 2, 3).contiguous()
    out = torch.einsum("bshr,hvr->bshv", out, v_up_weight)
    out = out.reshape(batch, seq_len, module.num_heads * module.v_head_dim)
    return module.o_proj(out)


def _run_once_torch_unfused(module, x, cos, sin, position_ids):
    module.zero_grad(set_to_none=True)
    module.train(True)
    local_x = x.detach().clone().requires_grad_(True)
    out = _torch_unfused_dsa_forward(module, local_x, cos, sin, position_ids)
    loss = out.float().square().mean()
    loss.backward()
    param_grads = {
        name: param.grad.detach().float().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": loss.detach().float().clone(),
        "out": out.detach().float().clone(),
        "x_grad": local_x.grad.detach().float().clone(),
        "param_grads": param_grads,
    }


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _max_param_grad_abs(a: dict, b: dict) -> float:
    common = set(a["param_grads"]) & set(b["param_grads"])
    if not common:
        return 0.0
    return max(_max_abs(a["param_grads"][name], b["param_grads"][name]) for name in common)


def test_glm5_dsa_run_to_run_accept_with_proof():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLM5 DSA accept-with-proof smoke.")

    from megatron.lite.primitive.modules.attention import build_rope_cache

    device = torch.device("cuda", int(torch.cuda.current_device()))
    torch.manual_seed(20260626)
    fused = _make_dsa().to(device=device, dtype=torch.bfloat16)
    unfused = copy.deepcopy(fused).to(device=device, dtype=torch.bfloat16)

    batch, seq, hidden = 1, 512, 128
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=64,
        max_position_embeddings=seq,
        rope_theta=1_000_000.0,
        device=device,
    )
    position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)

    fused_a = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    fused_b = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    unfused_a = _run_once_torch_unfused(unfused, x, cos, sin, position_ids)
    unfused_b = _run_once_torch_unfused(unfused, x, cos, sin, position_ids)

    fused_r2r_out = _max_abs(fused_a["out"], fused_b["out"])
    fused_r2r_x_grad = _max_abs(fused_a["x_grad"], fused_b["x_grad"])
    fused_r2r_param_grad = _max_param_grad_abs(fused_a, fused_b)
    unfused_r2r_out = _max_abs(unfused_a["out"], unfused_b["out"])
    unfused_r2r_x_grad = _max_abs(unfused_a["x_grad"], unfused_b["x_grad"])
    unfused_r2r_param_grad = _max_param_grad_abs(unfused_a, unfused_b)
    fused_vs_unfused_out = _max_abs(fused_a["out"], unfused_a["out"])
    fused_vs_unfused_x_grad = _max_abs(fused_a["x_grad"], unfused_a["x_grad"])
    fused_vs_unfused_param_grad = _max_param_grad_abs(fused_a, unfused_a)
    loss_diff = abs(float(fused_a["loss"].item()) - float(unfused_a["loss"].item()))

    noise_floor = max(
        fused_r2r_x_grad,
        fused_r2r_param_grad,
        unfused_r2r_x_grad,
        unfused_r2r_param_grad,
    )
    assert torch.isfinite(fused_a["loss"])
    assert torch.isfinite(unfused_a["loss"])
    assert unfused_r2r_out == 0.0
    assert unfused_r2r_x_grad == 0.0
    assert unfused_r2r_param_grad == 0.0
    assert fused_vs_unfused_out <= 5.0e-2
    assert fused_vs_unfused_x_grad <= max(5.0e-1, 16.0 * noise_floor)
    assert fused_vs_unfused_param_grad <= max(5.0e-1, 16.0 * noise_floor)

    print(
        "NON_SKIP_GLM5_DSA_RUN_TO_RUN_ACCEPT_WITH_PROOF "
        f"fused_loss={float(fused_a['loss'].item()):.6e} "
        f"unfused_loss={float(unfused_a['loss'].item()):.6e} "
        f"loss_diff={loss_diff:.6e} "
        f"fused_r2r_out_max_abs={fused_r2r_out:.6e} "
        f"fused_r2r_x_grad_max_abs={fused_r2r_x_grad:.6e} "
        f"fused_r2r_param_grad_max_abs={fused_r2r_param_grad:.6e} "
        f"unfused_r2r_out_max_abs={unfused_r2r_out:.6e} "
        f"unfused_r2r_x_grad_max_abs={unfused_r2r_x_grad:.6e} "
        f"unfused_r2r_param_grad_max_abs={unfused_r2r_param_grad:.6e} "
        f"fused_vs_unfused_out_max_abs={fused_vs_unfused_out:.6e} "
        f"fused_vs_unfused_x_grad_max_abs={fused_vs_unfused_x_grad:.6e} "
        f"fused_vs_unfused_param_grad_max_abs={fused_vs_unfused_param_grad:.6e} "
        f"noise_floor={noise_floor:.6e}"
    )
